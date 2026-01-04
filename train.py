import argparse

from blackjack.agent import mc_epsilon_greedy, test_agent
from blackjack.env import initialize_Q
from blackjack.func import build_decay, get_epsilon_schedule

import numpy as np


def get_hyperparameters():
    parser = argparse.ArgumentParser(
        description="Train MC epsilon-greedy blackjack agent"
    )

    # Epsilon arguments
    parser.add_argument("--epsilon-function", type=str, default='rational')
    parser.add_argument("--epsilon-c", type=float, default=1000)
    parser.add_argument("--epsilon-p", type=float, default=1)
    parser.add_argument("--epsilon-per-state", action="store_true", default=True)

    # Alpha arguments
    parser.add_argument(
        "--alpha-function",
        choices=["polynomial", "log", "rational"],
        default="polynomial",
    )
    parser.add_argument("--alpha-c", type=float, default=1.0)
    parser.add_argument("--alpha-p", type=float, default=1.0)

    # Other arguments
    parser.add_argument("--train-episodes", type=int, default=200_000_000)
    parser.add_argument("--Q-init", type=float, default=1)
    parser.add_argument("--test-episodes", type=int, default=10_000_000)

    return parser.parse_args()


if __name__ == "__main__":
    hparams = get_hyperparameters()

    Q = initialize_Q(hparams.Q_init)

    epsilon_decay = build_decay(
        hparams.epsilon_function, p=hparams.epsilon_p, c=hparams.epsilon_c
    )
    epsilon_schedule = get_epsilon_schedule(epsilon_decay, hparams.epsilon_per_state)

    alpha_func = build_decay(
        hparams.alpha_function, p=hparams.alpha_p, c=hparams.alpha_c
    )

    mc_epsilon_greedy(
        Q,
        num_episodes=hparams.train_episodes,
        epsilon_func=epsilon_schedule,
        alpha_func=alpha_func
    )

    np.save("Q_opt.npy", Q)
    print(f"Reward: {test_agent(Q, hparams.test_episodes)}")
