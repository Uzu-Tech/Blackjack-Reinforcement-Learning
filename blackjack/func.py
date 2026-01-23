import numpy as np

def constant_decay(n: int, c: float):
    return c

def polynomial_decay(n: int, p: float = 1.0, c: float = 1.0):
    return c / ((n + 1) ** p)

def log_decay(n: int, c: float = 1.0):
    return c / np.log1p(n + 1)

def rational_decay(n: int, c: float = 1.0):
    return c / (c + n)