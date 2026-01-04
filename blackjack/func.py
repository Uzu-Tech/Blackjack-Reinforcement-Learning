import numpy as np
from typing import Callable
from functools import partial

def constant_decay(n: int, c: float):
    return c

def polynomial_decay(n: int, p: float = 1.0, c: float = 1.0):
    return c / ((n + 1) ** p)

def log_decay(n: int, c: float = 1.0):
    return c / np.log1p(n + 1)

def rational_decay(n: int, c: float = 1.0):
    return c / (c + n)

def get_epsilon_schedule(decay: Callable[[int], float], per_state: bool):
    def epsilon_schedule(episode: int, num_state_visits: int) -> float:
        if per_state:
            return decay(num_state_visits)
        else:
            return decay(episode)
        
    return epsilon_schedule

DECAY_REGISTRY = {
    "constant": constant_decay,
    "polynomial": polynomial_decay,
    "log": log_decay,
    "rational": rational_decay 
}

def build_decay(
    decay_name: str,
    p: float | None,
    c: float,
):
    if decay_name not in DECAY_REGISTRY:
        raise ValueError(f"Unknown decay: {decay_name}")

    decay_fn = DECAY_REGISTRY[decay_name]

    if decay_name == "polynomial":
        if p is None:
            raise ValueError("polynomial decay requires p")
        return partial(decay_fn, p=p, c=c)

    if decay_name in {"log", "rational"}:
        return partial(decay_fn, c=c)

    return decay_fn