# Type hints for C++ Blackjack Environment
from dataclasses import dataclass

# ---------- Result object ----------
@dataclass(frozen=True)
class Result:
    reward: float
    next_state: int # Equal to -1 if game terminated
    split_state: int
    terminated: bool

# ---------- BlackjackEnv API ----------
class BlackjackEnv:
    def __init__(self, seed: int) -> None: ...
    def new_game(self) -> None: ...
    def get_state(self) -> int: ...  # state index
    def play_hand(self, action: int) -> Result: ...