from functools import partial
from typing import Callable, Optional

import numpy as np

from blackjack.env import State_Action, epsilon_greedy_policy, greedy_policy, State, Action
from blackjack.game import run_episode


def mc_epsilon_greedy(
    Q: np.ndarray,
    num_episodes: int,
    epsilon_func: Callable[[int, int], float],
    alpha_func: Callable[[int], float],
    N: Optional[np.ndarray] = None,
):
    # Keeps track of number of times each state action return is calculated
    if N is None:
        N = np.zeros(shape=Q.shape)

    policy = partial(epsilon_greedy_policy, Q=Q, N=N, epsilon_func=epsilon_func)
    for episode in range(num_episodes):
        _, state_action_returns = run_episode(policy)
        update_Q(Q, N, alpha_func, state_action_returns)

    return N


def update_Q(
    Q: np.ndarray,
    N: np.ndarray,
    alpha_func: Callable[[int], float],
    state_action_returns: dict[State_Action, float],
):
    for s_a, episode_return in state_action_returns.items():
        N[s_a] += 1
        # Monte Carlo increment rule for calculating the sample mean of the return
        Q[s_a] += alpha_func(N[s_a]) * (episode_return - Q[s_a])


def test_agent(Q: np.ndarray, num_episodes: int) -> float:
    policy = partial(greedy_policy, Q=Q)
    return test_policy(policy, num_episodes)

def test_policy(policy: Callable[[State], Action], num_episodes: int) ->float:
    total_reward_sum = 0
    for episode in range(num_episodes):
        reward, _ = run_episode(policy)
        total_reward_sum += reward
    return total_reward_sum / num_episodes
