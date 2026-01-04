from blackjack.env import Card
from blackjack.game import calculate_reward

def test_reward_player_blackjack_vs_normal():
    hand = [Card.ACE, Card.JACK]
    dealer = [Card.NINE, Card.SEVEN]
    assert calculate_reward(hand, dealer, 1) == 1.5

def test_reward_blackjack_push():
    hand = [Card.ACE, Card.KING]
    dealer = [Card.ACE, Card.TEN]
    assert calculate_reward(hand, dealer, 1) == 0

def test_reward_player_bust():
    hand = [Card.TEN, Card.EIGHT, Card.FIVE]  # 23
    dealer = [Card.TEN, Card.EIGHT, Card.FIVE]
    assert calculate_reward(hand, dealer, 1) == -1

def test_reward_dealer_bust():
    hand = [Card.TEN, Card.FIVE]
    dealer = [Card.TEN, Card.SEVEN, Card.SIX]
    assert calculate_reward(hand, dealer, 1) == 1

def test_reward_player_higher_total():
    hand = [Card.TEN, Card.SEVEN]
    dealer = [Card.TEN, Card.SIX]
    assert calculate_reward(hand, dealer, 1) == 1