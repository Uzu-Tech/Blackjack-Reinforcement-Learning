from pathlib import Path
from typing import Optional

import numpy as np

from blackjack.agent import Agent

SEED = 42
NUM_TRAIN_EPISODES = 200_000_000
SAVEFILE = Path("trained_agents")


def train_agent(algo_name: str, decay_factor: Optional[int] = None):
    agent = Agent(algo_name=algo_name, Q_init=0, decay_factor=decay_factor, seed=SEED)
    agent.train(num_episodes=NUM_TRAIN_EPISODES)
    np.save(
        SAVEFILE / f"{algo_name.replace(' ', '_')}__{NUM_TRAIN_EPISODES}.npy", agent.Q
    )

if __name__ == "__main__":
    train_agent(algo_name="Q Learning")
    print("Agent 1 Trained")
    train_agent(algo_name="Expected SARSA", decay_factor=1)
    print("Agent 2 Trained")