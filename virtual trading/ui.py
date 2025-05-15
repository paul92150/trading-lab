import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QLineEdit, QPushButton, QRadioButton, QButtonGroup,
    QPlainTextEdit, QMessageBox
)
from PyQt5.QtCore import QTimer
from portfolio import save_portfolios


class MainWindow(QMainWindow):
    def __init__(self, simulation_engine, portfolios):
        super().__init__()
        self.simulation_engine = simulation_engine
        self.portfolios = portfolios
        self.initUI()
        self.startTimer()

    def initUI(self):
        self.setWindowTitle("Simulateur de Portefeuille - Données Réelles")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Sélection du portefeuille
        portfolio_layout = QHBoxLayout()
        portfolio_label = QLabel("Sélectionner le portefeuille:")
        self.portfolio_combo = QComboBox()
        self.portfolio_combo.addItems(self.portfolios.keys())
        self.portfolio_combo.currentIndexChanged.connect(self.refresh_portfolio_info)
        portfolio_layout.addWidget(portfolio_label)
        portfolio_layout.addWidget(self.portfolio_combo)
        main_layout.addLayout(portfolio_layout)

        self.portfolio_info_label = QLabel()
        main_layout.addWidget(self.portfolio_info_label)

        # Zone pour modifier le cash
        cash_layout = QHBoxLayout()
        cash_label = QLabel("Modifier le cash (USD) :")
        self.cash_input = QLineEdit()
        self.cash_input.setPlaceholderText("Nouveau montant")
        self.cash_button = QPushButton("Mettre à jour Cash")
        self.cash_button.clicked.connect(self.update_cash)
        cash_layout.addWidget(cash_label)
        cash_layout.addWidget(self.cash_input)
        cash_layout.addWidget(self.cash_button)
        main_layout.addLayout(cash_layout)

        # Affichage des prix du marché
        market_layout = QVBoxLayout()
        market_label = QLabel("Prix du marché (USD):")
        self.market_info = QPlainTextEdit()
        self.market_info.setReadOnly(True)
        market_layout.addWidget(market_label)
        market_layout.addWidget(self.market_info)
        main_layout.addLayout(market_layout)

        # Saisie des transactions
        trans_layout = QHBoxLayout()
        asset_label = QLabel("Actif:")
        self.asset_combo = QComboBox()
        available_assets = list(self.simulation_engine.prices.keys())
        self.asset_combo.addItems(available_assets)
        self.asset_combo.setEditable(True)
        trans_layout.addWidget(asset_label)
        trans_layout.addWidget(self.asset_combo)

        qty_label = QLabel("Quantité:")
        self.qty_input = QLineEdit()
        trans_layout.addWidget(qty_label)
        trans_layout.addWidget(self.qty_input)

        self.buy_radio = QRadioButton("Achat")
        self.buy_radio.setChecked(True)
        self.sell_radio = QRadioButton("Vente")
        self.trans_type_group = QButtonGroup()
        self.trans_type_group.addButton(self.buy_radio)
        self.trans_type_group.addButton(self.sell_radio)
        trans_layout.addWidget(self.buy_radio)
        trans_layout.addWidget(self.sell_radio)

        self.execute_button = QPushButton("Exécuter")
        self.execute_button.clicked.connect(self.execute_transaction)
        trans_layout.addWidget(self.execute_button)
        main_layout.addLayout(trans_layout)

        # Historique des transactions
        history_label = QLabel("Historique des transactions:")
        self.history_info = QPlainTextEdit()
        self.history_info.setReadOnly(True)
        main_layout.addWidget(history_label)
        main_layout.addWidget(self.history_info)

        self.refresh_portfolio_info()

    def startTimer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(2000)

    def update_ui(self):
        market_text = ""
        for asset, price in self.simulation_engine.prices.items():
            market_text += f"{asset}: {price:.2f} USD\n"
        self.market_info.setPlainText(market_text)
        self.refresh_portfolio_info()

    def refresh_portfolio_info(self):
        current_portfolio = self.portfolio_combo.currentText()
        portfolio = self.portfolios[current_portfolio]
        total_value = portfolio.total_value(self.simulation_engine.prices)
        info = f"Portefeuille: {portfolio.name}\nCash: {portfolio.cash:.2f} USD\nTotal (Cash + Positions): {total_value:.2f} USD\nPositions:\n"
        for asset, qty in portfolio.positions.items():
            asset_value = qty * self.simulation_engine.prices.get(asset, 100)
            info += f"  {asset}: {qty} (valeur: {asset_value:.2f} USD)\n"
        self.portfolio_info_label.setText(info)

        history_text = "\n".join(portfolio.transaction_history)
        self.history_info.setPlainText(history_text)

    def execute_transaction(self):
        asset = self.asset_combo.currentText().upper()
        try:
            quantity = float(self.qty_input.text())
        except ValueError:
            QMessageBox.critical(self, "Erreur", "La quantité doit être un nombre.")
            return

        price = self.simulation_engine.prices.get(asset, 100.0)
        current_portfolio = self.portfolio_combo.currentText()
        portfolio = self.portfolios[current_portfolio]

        if self.buy_radio.isChecked():
            success, msg = portfolio.buy(asset, quantity, price)
        else:
            success, msg = portfolio.sell(asset, quantity, price)

        if success:
            QMessageBox.information(self, "Succès", msg)
        else:
            QMessageBox.critical(self, "Erreur", msg)
        self.refresh_portfolio_info()
        save_portfolios(self.portfolios)

    def update_cash(self):
        current_portfolio = self.portfolio_combo.currentText()
        portfolio = self.portfolios[current_portfolio]
        try:
            new_cash = float(self.cash_input.text())
        except ValueError:
            QMessageBox.critical(self, "Erreur", "Le montant du cash doit être un nombre.")
            return
        portfolio.cash = new_cash
        portfolio.transaction_history.append(f"Modification du cash à {new_cash:.2f} USD")
        self.cash_input.clear()
        self.refresh_portfolio_info()
        save_portfolios(self.portfolios)
        QMessageBox.information(self, "Succès", "Cash mis à jour.")
