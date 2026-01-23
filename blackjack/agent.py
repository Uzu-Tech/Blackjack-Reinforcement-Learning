from typing import Callable

from blackjack_env import BlackjackEnv


def test_policy(policy: Callable[[int], int], num_episodes: int) -> float:
    env = BlackjackEnv(seed=42)
    avg_reward = 0
    for episode in range(num_episodes):
        env.new_game()
        episode_reward = 0
        hand_terminated = False
        while not hand_terminated:
            state = env.get_state()
            action = policy(state)
            result = env.play_hand(action)
            hand_terminated = result.terminated
            episode_reward += result.reward

        avg_reward += (1 / (episode + 1)) * (episode_reward - avg_reward)

    return avg_reward
