from functools import partial
from typing import Callable, Optional

import numpy as np

from blackjack.algorithms import ALGORITHMS_MAP, EPSILON_ALGORITHMS, EpisodeRunner
from blackjack.policy import greedy
from blackjack.state_space import flatten_Q, initialize_Q
from blackjack_env import BlackjackEnv


class Agent:
    def __init__(
        self, algo_name: str, Q_init: float, decay_factor: Optional[int], seed: int = 42
    ) -> None:
        self.Q = initialize_Q(Q_init)
        self.N = np.zeros_like(self.Q)
        self.run_episode: EpisodeRunner = ALGORITHMS_MAP[algo_name]
        if self.run_episode in EPSILON_ALGORITHMS:
            if decay_factor is None:
                raise ValueError(
                    "Decay factor must be specified when using an epsilon algorithm"
                )

            self.run_episode = partial(self.run_episode, decay_factor=decay_factor)

        self.seed = seed
        self.train_returns = None
        self.test_returns = None
        np.random.seed(seed)

    def train(self, num_episodes: int):
        flat_Q = flatten_Q(self.Q)
        flat_N = flatten_Q(self.N)
        self.train_returns = np.zeros(num_episodes)
        env = BlackjackEnv(self.seed)
        for episode in range(num_episodes):
            self.train_returns[episode] = self.run_episode(flat_Q, flat_N, env)
        return self.train_returns

    def evaluate(self, num_episodes: int):
        return evaluate_Q(self.Q, num_episodes, self.seed)
    

def evaluate_Q(
    Q: np.ndarray, num_episodes: int, seed: int
):
    flat_Q = flatten_Q(Q)
    test_returns = evaluate_policy(
        partial(greedy, Q=flat_Q), num_episodes, seed
    )
    return test_returns


def evaluate_policy(
    policy: Callable[[int], int], num_episodes: int, seed: int
) -> np.ndarray:
    env = BlackjackEnv(seed)
    np.random.seed(seed)
    returns = np.zeros(num_episodes, dtype=np.float32)
    for episode in range(num_episodes):
        env.new_game()
        hand_terminated = False
        while not hand_terminated:
            state = env.get_state()
            action = policy(state)
            result = env.play_hand(action)
            hand_terminated = result.terminated
            returns[episode] += result.reward

    return returns
