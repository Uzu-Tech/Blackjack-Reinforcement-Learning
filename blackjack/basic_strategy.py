from blackjack.env import Action, State

# Shorthand for readability
H = Action.HIT
S = Action.STAND
D = Action.DOUBLE_DOWN
P = Action.SPLIT

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
    2: [H, H, P, P, P, P, H, H, H, H],  # 2,2
    3: [H, H, P, P, P, P, H, H, H, H],  # 3,3
    4: [H, H, H, H, H, H, H, H, H, H],  # 4,4
    5: [D, D, D, D, D, D, D, D, H, H],  # 5,5 (Treat as Hard 10)
    6: [P, P, P, P, P, H, H, H, H, H],  # 6,6
    7: [P, P, P, P, P, P, H, H, H, H],  # 7,7
    8: [P] * 10,  # 8,8
    9: [P, P, P, P, P, S, P, P, S, S],  # 9,9
    10: [S] * 10,  # 10,10
    11: [P] * 10,  # A,A
}


def basic_strategy(state: State) -> Action:
    tables = [STRATEGY_HARD, STRATEGY_SOFT, STRATEGY_PAIR]
    hand_type, hand_value, upcard, can_double = state

    action = tables[hand_type][hand_value][upcard - 2]
    if action == D and not can_double:
        return H
    if action == "DS":
        return D if can_double else S
    return action
