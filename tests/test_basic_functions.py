from blackjack.env import HandType, Card
from blackjack.game import get_hand_type, get_hand_value

def test_get_hand_type_pair():
    hand = [Card.EIGHT, Card.EIGHT]
    assert get_hand_type(hand) == HandType.PAIR

def test_get_hand_type_soft():
    hand = [Card.ACE, Card.SIX]  # 17 soft
    assert get_hand_type(hand) == HandType.SOFT

def test_get_hand_type_hard():
    hand = [Card.TEN, Card.THREE]
    assert get_hand_type(hand) == HandType.HARD

def test_get_hand_value_soft_ace():
    hand = [Card.ACE, Card.SIX]
    assert get_hand_value(hand) == 17

def test_get_hand_value_bust_aces_reduced():
    hand = [Card.ACE, Card.ACE, Card.NINE]  # 11 + 11 + 9 = 31 → reduce to 21
    assert get_hand_value(hand) == 21