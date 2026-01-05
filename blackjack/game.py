import random
from typing import Callable

import src.game as cpp
from blackjack.env import Action, Card, Hand, HandType, State, State_Action

random.seed(30)

CARDS = tuple(card for card in Card)

def run_episode(
    policy: Callable[[State], Action],
) -> tuple[float, dict[State_Action, float]]:
    hand = []
    dealer_hand = []
    draw_cards(hand, dealer_hand)
    play_dealer_hand(dealer_hand)

    state_action_returns = {}
    episode_return = play_hand(
        policy, hand, dealer_hand, state_action_returns, is_first_hand=True
    )
    return episode_return, state_action_returns


def play_hand(
    policy: Callable[[State], Action],
    hand: Hand,
    dealer_hand: Hand,
    state_action_returns: dict[State_Action, float],
    is_first_hand: bool,
) -> float:
    state = get_state(hand, get_upcard(dealer_hand))
    action = policy(state)

    # Base Case
    if action != Action.SPLIT:
        episode_return = play_non_pair_hand(
            policy, hand, dealer_hand, state_action_returns, is_first_hand
        )
        return episode_return

    # Split and decide actions for each hand
    hand1, hand2 = split_hand(hand)
    pair_hand_reward = sum(
        (
            calculate_reward(split_hand, dealer_hand, bet=1, is_first_hand=False)
            if is_ace_pair(hand)
            else play_hand(
                policy,
                split_hand,
                dealer_hand,
                state_action_returns,
                is_first_hand=False,
            )
        )
        for split_hand in (hand1, hand2)
    )
    # Checking if we've split the same hand already for first visit MC
    s_a = state + (Action.SPLIT,)
    if s_a not in state_action_returns:
        state_action_returns[s_a] = pair_hand_reward

    return pair_hand_reward


def play_non_pair_hand(
    policy: Callable[[State], Action],
    hand: Hand,
    dealer_hand: Hand,
    state_action_returns: dict[State_Action, float],
    is_first_hand: bool,
) -> float:
    visited_state_actions = []
    bet = 1

    while True:
        state = get_state(hand, get_upcard(dealer_hand))
        action = policy(state)
        visited_state_actions.append(state + (action,))

        if action == Action.STAND:
            break

        # Transition to next state
        deal_hand(hand)

        # Double Down only lets you draw one extra card
        if action == Action.DOUBLE_DOWN:
            bet = 2
            break

        if cpp.is_bust(hand):
            break

    episode_return = calculate_reward(hand, dealer_hand, bet, is_first_hand)
    for s_a in visited_state_actions:
        if s_a not in state_action_returns:
            state_action_returns[s_a] = episode_return

    return episode_return


def calculate_reward(
    hand: Hand, dealer_hand: Hand, bet: int, is_first_hand: bool
) -> float:
    if is_first_hand and cpp.has_blackjack(hand):
        if cpp.has_blackjack(dealer_hand):
            return 0
        return 1.5

    if cpp.is_bust(hand):
        return -bet

    if cpp.is_bust(dealer_hand) or cpp.get_hand_value(dealer_hand) < cpp.get_hand_value(
        hand
    ):
        return bet

    if cpp.get_hand_value(dealer_hand) > cpp.get_hand_value(hand):
        return -bet

    return 0


def split_hand(hand: Hand) -> tuple[Hand, Hand]:
    hand1 = [
        hand[0],
    ]
    hand2 = [
        hand[1],
    ]
    deal_hand(hand1)
    deal_hand(hand2)
    return hand1, hand2


def draw_cards(hand, dealer_hand):
    for _ in range(2):
        deal_hand(hand)
        deal_hand(dealer_hand)


def play_dealer_hand(dealer_hand: Hand):
    while cpp.get_hand_value(dealer_hand) < 17 or cpp.is_soft_17(dealer_hand):
        deal_hand(dealer_hand)
        if cpp.is_bust(dealer_hand):
            break


def get_state(hand: Hand, upcard: Card) -> State:
    hand_type = cpp.get_hand_type(hand)
    return (
        int(hand_type),
        (
            cpp.get_hand_value(hand)
            if hand_type != HandType.PAIR
            else cpp.get_card_value(hand[0])
        ),
        cpp.get_card_value(upcard),
        int(len(hand) == 2),
    )


def is_ace_pair(hand: Hand):
    return hand[0] == hand[1] == Card.ACE


def get_upcard(dealer_hand: Hand):
    return dealer_hand[0]

def deal_hand(hand: Hand):
    hand.append(CARDS[cpp.get_random_card()])
