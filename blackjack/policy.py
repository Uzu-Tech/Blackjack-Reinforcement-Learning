from typing import Callable

import numpy as np

from blackjack.state_space import Action

Policy = Callable[[int, np.ndarray], Action]

# Assuming Q is flattened for all these functions


def greedy(state: int, Q: np.ndarray) -> Action:
    return Action(np.argmax(Q[state]))


def random(state: int, Q: np.ndarray) -> Action:
    legal = Q[state] != -np.inf
    return Action(np.random.choice(np.flatnonzero(legal)))


def epsilon_func(num_visits: int, decay_factor: int):
    return decay_factor / (decay_factor + num_visits)


def epsilon_greedy(
    state: int, Q: np.ndarray, num_visits: int, decay_factor: int
) -> Action:
    epsilon = epsilon_func(decay_factor, num_visits)

    if np.random.rand() < epsilon:
        return random(state, Q)

    return greedy(state, Q)


def expected_epsilon_greedy_return(
    state: int, Q: np.ndarray, num_visits: int, decay_factor: int
):
    epsilon = epsilon_func(num_visits, decay_factor)
    legal = Q[state] != -np.inf
    return (1 - epsilon) * np.max(Q[state]) + epsilon * np.mean(Q[state][legal])
