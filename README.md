# Trading Lab — A modular Python research environment for exploring, backtesting, and simulating algorithmic trading strategies

This project provides a clean, extensible framework for developing and evaluating trading strategies. It includes classical techniques (momentum, mean reversion), machine learning baselines (e.g., SVM), market cycle analysis via Fourier transforms, and an interactive virtual trading simulator. A reinforcement learning (PPO-based) agent is also part of the repository structure but excluded from version control for flexibility and modularity.

## Project Structure

```text
trading-lab/
├── backtesting/            ← Modular backtesting engine and strategy implementations
├── virtual_trading/        ← Virtual trading simulator with interactive UI
├── research/
│   ├── cycles/             ← Market cycle detection using FFT
│   └── ML/
│       └── SVM/            ← SVM baseline classification and backtesting
├── rl_trading/             ← Reinforcement learning agent (ignored by Git)
├── requirements.txt        ← Project dependencies
├── LICENSE                 ← MIT License
└── README.md               ← This file
```

## Installation

Install dependencies (preferably in a virtual environment):

```bash
pip install -r requirements.txt
```

## How to Run

### Run a full backtest

```bash
python backtesting/main.py
```

### Launch the virtual trading simulator

```bash
python virtual_trading/main.py
```

### Train and backtest the SVM baseline

```bash
python research/ML/SVM/svm_backtest.py
```

## Notes

- This repository is intended for research and educational use. It is not designed for live trading or production deployment.
- Reinforcement learning models are pre-trained on 2000 days of BTC/USD data and can be extended to other assets or fine-tuned.
- All components are modular and interchangeable, allowing for rapid experimentation with different strategies and model architectures.

## License

This project is licensed under the MIT License. You are free to use, modify, and distribute the code. If you build something valuable or interesting based on this work, feel free to reach out.

