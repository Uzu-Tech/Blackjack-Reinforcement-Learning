import numpy as np

from blackjack.policy import epsilon_greedy, expected_epsilon_greedy_return, random
from blackjack.state_space import Action
from blackjack_env import BlackjackEnv
from typing import Callable

EpisodeRunner = Callable[[np.ndarray, np.ndarray, BlackjackEnv], float]

def q_learning_episode(Q: np.ndarray, N: np.ndarray, env: BlackjackEnv) -> float:
    env.new_game()
    game_terminated = False
    episode_return = 0

    while not game_terminated:
        state = env.get_state()
        # Q-learning uses random behavior policy (explores uniformly)
        action = random(state, Q)
        N[state, action] += 1
        result = env.play_hand(action)

        expected_return = result.reward
        # Q-learning uses greedy target policy (always picks best action)
        if result.next_state != -1:
            expected_return += np.max(Q[result.next_state])
        if result.split_state != -1:
            expected_return += np.max(Q[result.split_state])

        Q[state, action] += (1 / N[state, action]) * (
            expected_return - Q[state, action]
        )
        game_terminated = result.terminated
        episode_return += result.reward

    return episode_return


def sarsa_episode(Q: np.ndarray, N: np.ndarray, env: BlackjackEnv, decay_factor: int) -> float:
    env.new_game()
    game_terminated = False
    episode_return = 0

    while not game_terminated:
        state = env.get_state()
        num_visits = N[state].sum() + 1
        # SARSA uses the same policy for both behavior and target (on-policy learning)
        action = epsilon_greedy(state, Q, num_visits, decay_factor)
        N[state, action] += 1
        result = env.play_hand(action)

        expected_return = result.reward
        # Returns 0 if there's no next state or no split state
        expected_return += _get_next_state_return(
            Q, N, result.next_state, decay_factor, epsilon_greedy
        )
        expected_return += _get_next_state_return(
            Q, N, result.split_state, decay_factor, epsilon_greedy
        )

        Q[state, action] += (1 / N[state, action]) * (
            expected_return - Q[state, action]
        )
        game_terminated = result.terminated
        episode_return += result.reward

    return episode_return


def expected_sarsa_episode(
    Q: np.ndarray, N: np.ndarray, env: BlackjackEnv, decay_factor: int
) -> float:
    env.new_game()
    game_terminated = False
    episode_return = 0

    while not game_terminated:
        state = env.get_state()
        num_visits = N[state].sum() + 1
        action = epsilon_greedy(state, Q, num_visits, decay_factor)
        N[state, action] += 1
        result = env.play_hand(action)

        expected_return = result.reward
        expected_return += _get_expected_return(Q, N, result.next_state, decay_factor)
        expected_return += _get_expected_return(Q, N, result.split_state, decay_factor)

        Q[state, action] += (1 / N[state, action]) * (
            expected_return - Q[state, action]
        )
        game_terminated = result.terminated
        episode_return += result.reward

    return episode_return


def monte_carlo_episode(
    Q: np.ndarray, N: np.ndarray, env: BlackjackEnv, decay_factor: int
) -> float:
    env.new_game()
    game_terminated = False

    # Tracks pending split hands: [hand_return] or [-split_idx, hand_return]
    split_stack = []
    current_sa = set()
    visited_sa = []
    episode_returns = np.full(shape=len(Q) * len(Action), fill_value=np.nan)
    final_return = 0

    while not game_terminated:
        state = env.get_state()
        num_visits = N[state].sum() + 1
        action = epsilon_greedy(state, Q, num_visits, decay_factor)
        N[state, action] += 1

        state_action_idx = _state_action_idx(state, action)

        if action == Action.SPLIT:
            # Mark this split state as pending completion (negative index as marker)
            split_stack.append(mark_idx(state_action_idx))

            if np.isnan(episode_returns[state_action_idx]):
                visited_sa.append(state_action_idx)

            env.play_hand(action)
        else:
            # Track first-visit for this hand
            if state_action_idx not in current_sa:
                current_sa.add(state_action_idx)

            result = env.play_hand(action)
            game_terminated = result.terminated

            # Hand completed (received a reward)
            if result.reward != 0:
                final_return += result.reward
                _finalize_hand_returns(
                    episode_returns,
                    current_sa,
                    result.reward,
                    split_stack,
                    visited_sa,
                )
                current_sa.clear()

    # Update Q-values from episode returns
    _update_q_from_returns(Q, N, episode_returns, visited_sa)
    return final_return


def _finalize_hand_returns(
    episode_returns: np.ndarray,
    state_actions: set,
    hand_return: float,
    split_stack: list,
    visited_sa: list,
):
    # Assign return to all state action pairs visited in this hand
    for idx in state_actions:
        if np.isnan(episode_returns[idx]):
            episode_returns[idx] = hand_return
            visited_sa.append(idx)

    # Process split hand combinations
    if split_stack:
        # Check if we can combine this hand with a previous split hand
        previous_hand_return = 0
        while split_stack and (previous_hand_return := split_stack.pop()) >= -REWARD_OFFSET: # Anything above -2 must be a reward
            # Combine current hand with previous completed hand
            combined_return = hand_return + previous_hand_return
            split_idx = demark_idx(split_stack.pop()) # Negative marker becomes positive index
            episode_returns[split_idx] = combined_return
            hand_return = combined_return

        # If split not complete yet, push back marker and this hand's return
        if split_stack or previous_hand_return < -REWARD_OFFSET:
            split_stack.append(previous_hand_return)  # The negative marker
            split_stack.append(hand_return)

REWARD_OFFSET = 100 # |Reward| should never go above 100 chosen just as a large value

def mark_idx(idx: int):
    return -(idx + REWARD_OFFSET)

def demark_idx(marked_idx: int):
    return -marked_idx - REWARD_OFFSET


def _update_q_from_returns(
    Q: np.ndarray, N: np.ndarray, episode_returns: np.ndarray, visited_sa: list
):
    for idx in visited_sa:
        state, action = divmod(idx, len(Action))
        Q[state, action] += (1 / N[state, action]) * (
            episode_returns[idx] - Q[state, action]
        )


def _state_action_idx(state: int, action: int):
    return state * len(Action) + action


def _get_next_state_return(
    Q: np.ndarray, N: np.ndarray, state: int, decay_factor: int, policy_fn
) -> float:
    # state == -1 means the game terminated (no next state to evaluate)
    if state == -1:
        return 0.0
    num_visits = N[state].sum()
    action = policy_fn(state, Q, num_visits, decay_factor)
    return Q[state, action]


def _get_expected_return(
    Q: np.ndarray, N: np.ndarray, state: int, decay_factor: int
) -> float:
    # state == -1 means the game terminated (no next state to evaluate)
    if state == -1:
        return 0.0
    num_visits = N[state].sum()
    # expected_epsilon_greedy_return returns the expected value directly
    return expected_epsilon_greedy_return(state, Q, num_visits, decay_factor)

ALGORITHMS_MAP = {
    "Q Learning": q_learning_episode,
    "SARSA": sarsa_episode,
    "Expected SARSA": expected_sarsa_episode,
    "Monte Carlo": monte_carlo_episode
}

EPSILON_ALGORITHMS = {sarsa_episode, expected_sarsa_episode, monte_carlo_episode}

