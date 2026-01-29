from blackjack.state_space import MIN_VALUE, NUM_UPCARDS

# Shorthand for readability
H = 0
S = 1
D = 2
P = 3

# Key: hand_value, Value: list of actions for upcards 2 through Ace
STRATEGY_HARD = {
    4: [H] * 10,
    5: [H] * 10,
    6: [H] * 10,
    7: [H] * 10,
    8: [H] * 10,
    9: [H, D, D, D, D, H, H, H, H, H],
    10: [D, D, D, D, D, D, D, D, H, H],
    11: [D, D, D, D, D, D, D, D, D, D],
    12: [H, H, S, S, S, H, H, H, H, H],
    13: [S, S, S, S, S, H, H, H, H, H],
    14: [S, S, S, S, S, H, H, H, H, H],
    15: [S, S, S, S, S, H, H, H, H, H],
    16: [S, S, S, S, S, H, H, H, H, H],
    17: [S] * 10,
    18: [S] * 10,
    19: [S] * 10,
    20: [S] * 10,
    21: [S] * 10,
}

STRATEGY_SOFT = {
    13: [H, H, H, D, D, H, H, H, H, H],  # A,2
    14: [H, H, H, D, D, H, H, H, H, H],  # A,3
    15: [H, H, D, D, D, H, H, H, H, H],  # A,4
    16: [H, H, D, D, D, H, H, H, H, H],  # A,5
    17: [H, D, D, D, D, H, H, H, H, H],  # A,6
    18: [S, "DS", "DS", "DS", "DS", S, S, H, H, H],  # A,7
    19: [S] * 10,
    20: [S] * 10,
    21: [S] * 10,
}

STRATEGY_PAIR = {
    4: [H, H, P, P, P, P, H, H, H, H],  # 2,2
    6: [H, H, P, P, P, P, H, H, H, H],  # 3,3
    8: [H, H, H, H, H, H, H, H, H, H],  # 4,4
    10: [D, D, D, D, D, D, D, D, H, H],  # 5,5 (Treat as Hard 10)
    12: [P, P, P, P, P, H, H, H, H, H],  # 6,6
    14: [P, P, P, P, P, P, H, H, H, H],  # 7,7
    16: [P] * 10,  # 8,8
    18: [P, P, P, P, P, S, P, P, S, S],  # 9,9
    20: [S] * 10,  # 10,10
    "A": [P] * 10,  # A,A
}


def basic_strategy(state: int) -> int:
    tables = [STRATEGY_HARD, STRATEGY_SOFT, STRATEGY_PAIR]
    can_split = state % 2
    state //= 2
    can_double = state % 2
    state //= 2
    useable_ace = state % 2
    state //= 2
    upcard = (state % NUM_UPCARDS) + 2  # dealer upcard rank (2..Ace)
    state //= NUM_UPCARDS
    hand_value = state + MIN_VALUE  # remaining value is player sum

    if not can_split:
        action = tables[useable_ace][hand_value][upcard - 2]
    else:
        if useable_ace:
            hand_value = "A"
        action = STRATEGY_PAIR[hand_value][upcard - 2]

    if action == D and not can_double:
        return H
    if action == "DS":
        return D if can_double else S
    return action