from enum import Enum, IntEnum, auto
from typing import TypeAlias, Callable

import numpy as np

np.random.seed(30)

State_Action: TypeAlias = tuple[int, int, int, int, int]
State: TypeAlias = tuple[int, int, int, int]
Hand: TypeAlias = list["Card"]

BLACKJACK_VALUE = 21
MAX_HARD_VALUE = 21
MIN_HARD_VALUE = 4
MAX_SOFT_VALUE = 21
MIN_SOFT_VALUE = 13
MAX_SPLIT_VALUE = 11
MIN_SPLIT_VALUE = 2


class HandType(IntEnum):
    HARD = 0
    SOFT = 1
    PAIR = 2


class Action(IntEnum):
    HIT = 0
    STAND = 1
    DOUBLE_DOWN = 2
    SPLIT = 3


class Card(IntEnum):
    ACE = 0
    TWO = auto()
    THREE = auto()
    FOUR = auto()
    FIVE = auto()
    SIX = auto()
    SEVEN = auto()
    EIGHT = auto()
    NINE = auto()
    TEN = auto()
    JACK = auto()
    QUEEN = auto()
    KING = auto()
    

def epsilon_greedy_policy(
    state: State,
    Q: np.ndarray,
    N: np.ndarray,
    epsilon_func: Callable[[int, int], float]
) -> Action:
    actions = Q[state]

    episode = np.sum(N)
    num_state_visits = np.sum(N[state])
    epsilon = epsilon_func(episode, num_state_visits)

    if np.random.rand() < epsilon:
        valid = np.where(actions != -np.inf)[0]
        return Action(np.random.choice(valid))

    return Action(np.argmax(actions))


def greedy_policy(state: State, Q: np.ndarray):
    return Action(np.argmax(Q[state]))


def initialize_Q(value: float) -> np.ndarray:
    # Q(s, a) is the expected reward after taking action a in state s
    # Each state measures 4 variables (hand type, hand value, upcard, has 2 cards)
    # Create a fixed array with (0, max_value) possible indices
    Q = np.full(
        fill_value=value,
        shape=(
            len(HandType),
            BLACKJACK_VALUE + 1,  # (0 - 21) should cover all possible hand values
            12,  # (0 - 11) should cover all dealer upcards
            2,  # Only two values for the bool
            len(Action),
        ),
        dtype=np.float32,
    )

    valid = np.zeros(Q.shape, dtype=bool)

    fill_valid_hard_actions(valid)
    fill_valid_soft_actions(valid)
    fill_valid_split_actions(valid)

    Q[~valid] = -np.inf  # Ensures invalid actions are never selected by policy
    return Q


def fill_valid_hard_actions(valid: np.ndarray):
    for hv in range(MIN_HARD_VALUE, MAX_HARD_VALUE + 1):
        # All hit or stand actions are always valid for hard hands:
        for action in (Action.HIT, Action.STAND):
            valid[HandType.HARD, hv, :, :, action.value] = True

        # Double down only when you have exactly 2 cards:
        valid[HandType.HARD, hv, :, 1, Action.DOUBLE_DOWN] = True


def fill_valid_soft_actions(valid: np.ndarray):
    for sv in range(MIN_SOFT_VALUE, MAX_SOFT_VALUE + 1):
        # All hit or stand actions are always valid for soft hands
        for action in (Action.HIT, Action.STAND):
            valid[HandType.SOFT, sv, :, :, action.value] = True

        # Double down only when you have exactly 2 cards
        valid[HandType.SOFT, sv, :, 1, Action.DOUBLE_DOWN] = True


def fill_valid_split_actions(valid: np.ndarray):
    for pv in range(MIN_SPLIT_VALUE, MAX_SPLIT_VALUE + 1):
        # All hit, stand or split actions are always valid for split hands with two cards:
        for action in (Action.HIT, Action.STAND, Action.SPLIT):
            valid[HandType.PAIR, pv, :, 1, action.value] = True

        # Double down only when you have exactly 2 cards:
        valid[HandType.PAIR, pv, :, 1, Action.DOUBLE_DOWN] = True
