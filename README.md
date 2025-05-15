# 🧠 Trading Lab — A modular Python research environment for exploring, backtesting, and simulating algorithmic trading strategies.

This project includes classical strategies, machine learning baselines (like SVM), market cycle analysis using Fourier transforms, and an interactive virtual trading interface. A reinforcement learning agent (PPO-based) is also included but currently ignored from Git for flexibility.

---

## 📁 Project Structure

- `backtesting/` : Clean modular backtesting engine with several prebuilt strategies (mean reversion, momentum, breakout, trend following, etc.)
- `virtual_trading/` : Interactive virtual portfolio with basic UI for simulating trades
- `research/cycles/` : Tools for cycle detection using FFT and related techniques
- `research/ML/SVM/` : Baseline classification using SVM (with simple backtesting logic)
- `rl_trading/` : Reinforcement learning trading agent using Stable-Baselines3 (currently ignored in `.gitignore`)
- `LICENSE`, `README.md` : Standard project metadata

---

## 🚀 How to Run

Install dependencies (in a virtual environment):

```
pip install -r requirements.txt
```

Run a full backtest:

```
python backtesting/main.py
```

Try the virtual trading simulator:

```
python virtual_trading/main.py
```

Train and backtest the SVM baseline:

```
python research/ML/SVM/svm_backtest.py
```

---

## 📌 Notes

- This repo is meant for research and experimentation, not live trading.
- RL models are trained on 2000 days of BTC/USD and can be fine-tuned or extended to other assets.
- All modules are designed to be easily replaceable or extendable for rapid strategy development.

---

## 📜 License

MIT — Free to use, modify, and distribute. If you build something interesting on top of it, feel free to reach out!
