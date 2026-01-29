# Blackjack Reinforcement Learning

Exploring the effectiveness of various reinforcement learning methods when applied to blackjack.

## Overview

This project implements and compares different reinforcement learning algorithms for learning optimal blackjack strategies. The environment is built using a high-performance C++ backend with Python bindings, enabling fast training over millions of episodes.

## Features

- **Custom C++ Blackjack Environment**: High-performance blackjack simulator with Python bindings via pybind11
- **Multiple RL Algorithms**: Implementation of Q-Learning and SARSA
- **Hyperparameter Optimization**: Tools for systematic algorithm comparison and hyperparameter tuning
- **Strategy Visualization**: Visual comparison of learned policies vs. basic strategy
- **Interactive Plotting**: Plotly-based visualizations for strategies and algorithm performance
- **Comprehensive State Space**: Handles hard hands, soft hands, pairs, doubling, and splitting

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
├── trained_agents/          # Saved Q-tables
├── compare_algos.py        # Algorithm comparison experiments
├── train_agent.py          # Training script
├── evaluate_agent.py       # Evaluation script
├── plot_results.py         # Results visualization
├── setup.py                # C++ extension build configuration
├── requirements.txt        # Python dependencies
└── blackjack_env.pyi      # Type hints for C++ environment
```

## Installation

### Prerequisites

- Python 3.8+
- C++ compiler with C++17 support
- pybind11

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Uzu-Tech/Blackjack-Reinforcement-Learning.git
cd Blackjack-Reinforcement-Learning
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Build the C++ extension:
```bash
python setup.py build_ext --inplace
```

## Usage

### Training an Agent

```python
from blackjack.agent import Agent

# Create and train an agent
agent = Agent(algo_name="Q Learning", Q_init=0, seed=42)
agent.train(num_episodes=1_000_000)

# Evaluate performance
mean_return = agent.evaluate(num_episodes=100_000)
print(f"Mean return: {mean_return}")
```

### Comparing Algorithms

Run comprehensive algorithm comparison experiments:

```bash
python compare_algos.py
```

This will:
- Train agents using different algorithms (Q-Learning, SARSA)
- Test various hyperparameters (decay factors)
- Save results to SQLite database for analysis

## Visualization & Plotting

### Strategy Visualization

The project includes powerful visualization tools to understand learned strategies:

#### Visualizing Q-Tables

```python
import numpy as np
from blackjack.visualizer import plot_strategy_hard, plot_strategy_soft, plot_strategy_pair

# Load trained Q-table
Q = np.load("trained_agents/Q_Learning__200000000.npy")

# Visualize learned strategies
plot_strategy_hard(Q)  # Hard hands (no useable ace)
plot_strategy_soft(Q)  # Soft hands (with useable ace)
plot_strategy_pair(Q)  # Pairs (when splitting is possible)
```

Each visualization shows:
- **X-axis**: Dealer's upcard (2-11, where 11 is Ace)
- **Y-axis**: Player's hand total
- **Color**: Optimal action (HIT=0, STAND=1, DOUBLE=2, SPLIT=3)

#### Strategy Plots Explained

**Hard Hands Plot** (`plot_strategy_hard`):
- Shows optimal actions for hard totals (5-20)
- Useful for comparing learned strategy vs. basic strategy
- Hard hands have no useable ace

**Soft Hands Plot** (`plot_strategy_soft`):
- Shows optimal actions for soft totals (13-20)
- Soft 13 = Ace + 2, Soft 14 = Ace + 3, etc.
- Critical for understanding when to hit soft 17-18

**Pairs Plot** (`plot_strategy_pair`):
- Shows when to split pairs
- Covers all possible pairs (2,2) through (Ace,Ace)
- Helps identify optimal splitting strategy

### Algorithm Performance Comparison

Use `plot_results.py` to visualize experiment results:

```python
from pathlib import Path
from plot_results import plot_compare_algos, plot_Q_table

# Plot algorithm comparison from experiment
plot_compare_algos(experiment_id=1, save_path=Path("plots"))

# Plot Q-table strategies
agent_path = Path("trained_agents")
agent_name = "Q_Learning__200000000"
save_path = Path("plots") / agent_name

plot_Q_table(agent_path / f"{agent_name}.npy", save_path)
```

#### Performance Comparison Plot

The `plot_compare_algos` function generates:
- **Line plot** comparing different algorithms
- **X-axis**: Exploration rate (decay factor)
- **Y-axis**: Expected return (average reward per game)
- **Multiple traces**: One for each algorithm (SARSA variants)
- **Horizontal line**: Q-Learning baseline performance

This helps identify:
- Which algorithms perform best
- Optimal exploration rates
- Trade-offs between exploration and exploitation

### Training Progress Visualization

Monitor training with sliding window averages:

```python
from blackjack.visualizer import plot_sliding_window
import numpy as np

# Assuming you have episode rewards
episode_rewards = np.array([...])  # Your training rewards

# Plot smoothed learning curve
plot_sliding_window(episode_rewards, window_size=0.01)  # 1% window
```

This creates an interactive line plot showing:
- Learning progress over episodes
- Smoothed average to reduce noise
- Convergence behavior

### Saving Plots

All plotting functions support saving to disk:

```python
# Save high-resolution images
fig.write_image("plots/strategy_hard.png", scale=4)
```

Plots are saved to the `plots/` directory organized by agent name and experiment ID.

### Interactive Features

All Plotly visualizations include:
- **Zoom**: Click and drag to zoom
- **Pan**: Shift + drag to pan
- **Hover**: See exact values on mouseover
- **Download**: Save as PNG using camera icon
- **Reset**: Double-click to reset view

## Algorithms

### Q-Learning
Off-policy temporal difference learning that learns the optimal policy by updating Q-values based on the maximum future reward.

### SARSA
On-policy temporal difference learning that updates Q-values based on the action actually taken by the current policy.

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

Database location: `databases/compare_algos.sqlite3`

## Dependencies

Key dependencies include:
- **numpy**: Numerical computations
- **plotly**: Interactive visualizations
- **polars**: Fast dataframe operations for plotting
- **optuna**: Hyperparameter optimization
- **tensorboard**: Training monitoring
- **pytest**: Testing framework
- **pybind11**: Python-C++ bindings

See `requirements.txt` for complete list.

## Performance

The C++ implementation enables:
- Fast episode simulation (millions of hands per minute)
- Efficient state representation
- Optimized reward calculations

## Example Workflow

Complete workflow from training to visualization:

```bash
# 1. Run algorithm comparison
python compare_algos.py

# 2. Plot results
python plot_results.py

# 3. Analyze specific agent
python -c "
from pathlib import Path
from plot_results import plot_Q_table
import numpy as np

Q = np.load('trained_agents/Q_Learning__200000000.npy')
from blackjack.visualizer import plot_strategy_hard
plot_strategy_hard(Q)
"
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

[Add your license information here]

## Acknowledgments

This project implements standard reinforcement learning algorithms applied to the classic blackjack problem, drawing inspiration from Sutton & Barto's "Reinforcement Learning: An Introduction."