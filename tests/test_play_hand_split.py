import blackjack.game
from blackjack.env import Card, HandType, Action
from blackjack.game import play_hand

def test_play_hand_split(monkeypatch):
    monkeypatch.setattr(
        blackjack.game, "deal_hand", lambda hand: hand.append(Card.TEN)
    )

    hand = [Card.EIGHT, Card.EIGHT]
    dealer = [Card.TEN, Card.SEVEN]

    def policy(_): return Action.STAND
    returns = {}

    total = play_hand(Action.SPLIT, policy, hand, dealer, returns)

    assert total == 2
    assert returns == {
        (HandType.HARD, 18, 10, True, Action.STAND): 1,
        (HandType.PAIR, 8, 10, True, Action.SPLIT): 2
    }