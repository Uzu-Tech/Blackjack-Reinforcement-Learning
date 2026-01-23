from enum import IntEnum
from typing import NamedTuple

import numpy as np

MAX_VALUE = 21
MIN_VALUE = 4
NUM_HAND_VALUES = (MAX_VALUE - MIN_VALUE) + 1
NUM_UPCARDS = 10


class State(NamedTuple):
    hand_value: int
    upcard: int
    useable_ace: bool
    can_double: bool
    can_split: bool


class Action(IntEnum):
    HIT = 0
    STAND = 1
    DOUBLE = 2
    SPLIT = 3


def initialize_Q(value: float) -> np.ndarray:
    # Q(s, a) is the expected reward after taking action a in state s
    Q = np.full(
        fill_value=value,
        shape=(NUM_HAND_VALUES, NUM_UPCARDS, 2, 2, 2, len(Action)),
        dtype=np.float32,
    )

    legal = np.zeros(Q.shape, dtype=bool)
    fill_legal_actions(legal)

    Q[~legal] = -np.inf  # Ensures illegal actions are never selected by policy
    return Q


def fill_legal_actions(legal: np.ndarray):
    CAN_DOUBLE = 1
    CAN_SPLIT = 1

    # Can hit or stand in any state
    legal[:, :, :, :, :, Action.HIT] = True
    legal[:, :, :, :, :, Action.STAND] = True

    # Can only double with two cards
    legal[:, :, :, CAN_DOUBLE, :, Action.DOUBLE] = True

    # Can only split when can split
    legal[:, :, :, :, CAN_SPLIT, Action.SPLIT] = True


def flatten_Q(Q: np.ndarray) -> np.ndarray:
    return Q.reshape(-1, len(Action))