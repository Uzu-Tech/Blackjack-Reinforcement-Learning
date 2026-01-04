import blackjack.game
from blackjack.env import Card
from blackjack.game import split_hand


def test_split_hand_calls_deal_twice(monkeypatch):
    # Force random.choice to always return Card.FIVE
    monkeypatch.setattr(
        blackjack.game, "deal_hand", lambda hand: hand.append(Card.FIVE)
    )

    hand = [Card.EIGHT, Card.EIGHT]

    h1, h2 = split_hand(hand)

    assert h1 == [Card.EIGHT, Card.FIVE]
    assert h2 == [Card.EIGHT, Card.FIVE]
