# portfolio.py

import json
import os

DATA_FILE = "portfolio_data.json"

class Portfolio:
    def __init__(self, name, cash=10000, positions=None, transaction_history=None):
        self.name = name
        self.cash = cash
        self.positions = positions if positions is not None else {}
        self.transaction_history = transaction_history if transaction_history is not None else []

    def buy(self, asset, quantity, price):
        total_cost = quantity * price
        if self.cash < total_cost:
            return False, "Cash insuffisant"
        self.cash -= total_cost
        self.positions[asset] = self.positions.get(asset, 0) + quantity
        self.transaction_history.append(f"Achat de {quantity} {asset} à {price:.2f} USD")
        return True, "Achat effectué"

    def sell(self, asset, quantity, price):
        if self.positions.get(asset, 0) < quantity:
            return False, "Position insuffisante"
        self.positions[asset] -= quantity
        self.cash += quantity * price
        self.transaction_history.append(f"Vente de {quantity} {asset} à {price:.2f} USD")
        return True, "Vente effectuée"

    def total_value(self, current_prices):
        total = self.cash
        for asset, qty in self.positions.items():
            total += qty * current_prices.get(asset, 100)
        return total

    def to_dict(self):
        return {
            "cash": self.cash,
            "positions": self.positions,
            "transaction_history": self.transaction_history
        }

    @classmethod
    def from_dict(cls, name, data):
        return cls(
            name,
            cash=data.get("cash", 10000),
            positions=data.get("positions", {}),
            transaction_history=data.get("transaction_history", [])
        )


def load_portfolios():
    portfolios = {}
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        for name, pdata in data.items():
            portfolios[name] = Portfolio.from_dict(name, pdata)
    else:
        portfolios = {
            "PEA": Portfolio("PEA", cash=20000),
            "Crypto": Portfolio("Crypto", cash=10000)
        }
    return portfolios


def save_portfolios(portfolios):
    data = {name: p.to_dict() for name, p in portfolios.items()}
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)
