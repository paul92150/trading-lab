# Trading Lab

A modular Python research environment for exploring, backtesting, and simulating algorithmic trading strategies.

This project provides a framework for developing and evaluating trading strategies. It includes classical techniques (momentum, mean reversion), machine learning baselines (e.g., SVM), market cycle analysis via Fourier transforms, and an interactive virtual trading simulator. A reinforcement learning (PPO-based) agent is also part of the repository but excluded from version control for flexibility and modularity.

The new `alphas/` module introduces tools for alpha signal research, including symbolic alpha generation via Genetic Programming (GP), deep learning models (MLP), rank-based fitness functions (Spearman IC), and fast simulation for rapid iteration.

## Project Structure

```text
trading-lab/
├── backtesting/              ← Modular backtesting engine and strategy implementations
├── virtual_trading/          ← Virtual trading simulator with interactive UI
├── research/
│   ├── cycles/               ← Market cycle detection using FFT
│   ├── ML/                   ← Classical ML models (SVM, etc.)
│   └── alphas/               ← Alpha signal research (GP, MLP, rank-based signals)
│       ├── scripts/          ← Executable training/eval scripts
│       └── src/              ← Modular utils: simulation, metrics, preprocessing, fitness
├── rl_trading/               ← Reinforcement learning agent (ignored by Git)
├── models/                   ← Exported models and scalers (ignored by Git)
├── requirements.txt          ← Project dependencies
├── LICENSE                   ← MIT License
└── README.md                 ← This file
```

## Features of the `alphas/` Module

- Symbolic alpha generation using Genetic Programming (`gplearn`)
- MLP-based alpha prediction with Optuna hyperparameter tuning
- Cross-sectional Spearman rank correlation (IC) fitness
- Fast vectorized portfolio simulation for daily alpha testing
- Modular design with reusable simulation, preprocessing, and metrics components

## Usage

### Train and evaluate symbolic alphas with Genetic Programming

```bash
python research/alphas/scripts/alpha_train_gp_sharpe.py     # GP using Sharpe ratio fitness
python research/alphas/scripts/alpha_train_gp_rank.py       # GP using Spearman rank correlation (IC)
```

### Train MLP model with Optuna hyperparameter search

```bash
python research/alphas/scripts/alpha_train_optuna_mlp.py
```

### Test a manually defined alpha signal

```bash
python research/alphas/scripts/alpha_test_manual.py
```

### Run a classical strategy backtest

```bash
python backtesting/main.py
```

### Launch the virtual trading simulator

```bash
python virtual_trading/main.py
```

## Notes

- The alpha module is intended for research purposes and is not designed for production use.
- Reinforcement learning models are trained on BTC/USD data and can be extended to other markets.
- The architecture emphasizes modularity and reusability, enabling rapid testing of new strategies, fitness metrics, and asset universes.

## License

This project is licensed under the MIT License.
