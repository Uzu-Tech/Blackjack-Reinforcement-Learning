from typing import Callable
import numpy as np
from state_space import Action

Policy = Callable[[int, np.ndarray], Action]

# Assuming Q is flattened for all these functions

def greedy(state: int, Q: np.ndarray) -> Action:
    return Action(np.argmax(Q[state]))

def random(state: int, Q: np.ndarray) -> Action:
    legal = Q[state] != -np.inf
    return Action(np.random.choice(np.flatnonzero(legal)))