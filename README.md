# Blackjack Reinforcement Learning

Exploring the effectiveness of various reinforcement learning methods when applied to blackjack.

## Overview

This project implements and compares different reinforcement learning algorithms for learning optimal blackjack strategies. The environment is built using a high-performance C++ backend with Python bindings, enabling fast training over millions of episodes.

## Features

- **Custom C++ Blackjack Environment**: High-performance blackjack simulator with Python bindings via pybind11
- **Multiple RL Algorithms**: Implementation of Q-Learning, SARSA, Expected SARSA, and Monte Carlo Methods
- **Strategy Visualization**: Visual comparison of learned policies vs. basic strategy
- **Research Paper**: Small research paper explaining the process and analysing the results

## Project Structure

```
Blackjack-Reinforcement-Learning/
├── src/                      # C++ source files for blackjack environment
│   ├── main.cpp             # BlackjackEnv implementation
│   ├── main.hpp
│   ├── hand.cpp             # Hand logic and state management
│   └── hand.hpp
├── blackjack/               # Python RL implementation
│   ├── agent.py            # Agent training and evaluation
│   ├── algorithms.py       # RL algorithm implementations (Q-Learning, SARSA)
│   ├── policy.py           # Policy functions (greedy, random)
│   ├── state_space.py      # State and action definitions
│   ├── basic_strategy.py   # Standard blackjack basic strategy
│   ├── func.py             # Decay functions for learning rates
│   └── visualizer.py       # Strategy visualization tools
├── tests/                   # Test suite
├── databases/               # SQLite experiment results
├── plots/                   # Generated visualizations
├── paper/                   # Small research paper
├── trained_agents/          # Saved Q-tables
├── compare_algos.py        # Algorithm comparison experiments
├── train_agent.py          # Training script
├── evaluate_agent.py       # Evaluation script
├── plot_results.py         # Results visualization
├── setup.py                # C++ extension build configuration
├── requirements.txt        # Python dependencies
└── blackjack_env.pyi      # Type hints for C++ environment
```

## State Space

The state space includes:
- **Hand Value**: 4-21
- **Dealer Upcard**: 2-11 (Ace)
- **Useable Ace**: Boolean
- **Can Double**: Boolean (only with 2 cards)
- **Can Split**: Boolean

## Actions

- **HIT (0)**: Take another card
- **STAND (1)**: End turn
- **DOUBLE (2)**: Double bet and take exactly one more card
- **SPLIT (3)**: Split pair into two hands

## Experiment Tracking

Results are stored in SQLite databases with:
- Hyperparameter registry
- Experiment configurations
- Per-trial results (algorithm, decay factor, mean return)

## Performance

The C++ implementation enables:
- Fast episode simulation (millions of hands per minute)
- Efficient state representation
- Optimized reward calculations

## Acknowledgments

This project implements standard reinforcement learning algorithms applied to the classic blackjack problem, drawing inspiration from Sutton & Barto's "Reinforcement Learning: An Introduction."
