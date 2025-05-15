import sys
import threading
from PyQt5.QtWidgets import QApplication
from engine import SimulationEngine
from portfolio import load_portfolios
from ui import MainWindow

def main():
    # Charger les portefeuilles depuis le fichier JSON
    portfolios = load_portfolios()

    # Créer et démarrer le moteur de simulation (mise à jour des prix)
    sim_engine = SimulationEngine()
    threading.Thread(target=sim_engine.update_prices, daemon=True).start()

    # Lancer l'application PyQt
    app = QApplication(sys.argv)
    window = MainWindow(sim_engine, portfolios)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
