import numpy as np
from policy import random

from blackjack_env import BlackjackEnv


def q_learning_episode(N: np.ndarray, Q: np.ndarray, env: BlackjackEnv):
    env.new_game()
    game_terminated = False

    while not game_terminated:
        state = env.get_state()
        # Behaviour policy
        action = random(state, Q)
        N[state, action] += 1
        result = env.play_hand(action)

        expected_return = result.reward
        # Using greedy as the target policy
        if result.next_state != -1:
            expected_return += np.max(Q[result.next_state])

        if result.split_state != -1:
            expected_return += np.max(Q[result.split_state])

        Q[state, action] += (1 / N[state, action]) * (
            expected_return - Q[state, action]
        )
        game_terminated = result.terminated


def sarsa_episode(episode: int, Q: np.ndarray, env: BlackjackEnv):
    env.new_game()
    game_terminated = False
    state = env.get_state()

    while not game_terminated:
        # Behaviour policy
        action = random(state, Q)
        result = env.play_hand(action)

        expected_return = result.reward
        # Using greedy as the target policy
        if result.next_state != -1:
            expected_return += np.max(Q[result.next_state])

        if result.split_state != -1:
            expected_return += np.max(Q[result.split_state])

        Q[state, action] += (1 / episode) * (expected_return - Q[state, action])
        game_terminated = result.terminated
