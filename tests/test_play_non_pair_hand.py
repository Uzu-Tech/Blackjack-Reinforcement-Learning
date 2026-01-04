import blackjack.game
from blackjack.env import Card, Action, HandType
from blackjack.game import play_non_pair_hand, get_hand_value

def constant_policy(action):
    return lambda _: action

def test_play_non_pair_stand(monkeypatch):
    hand = [Card.TEN, Card.SEVEN]
    dealer = [Card.TEN, Card.SIX]

    # Player stands immediately
    returns = {}
    policy = constant_policy(Action.STAND)
    episode_return = play_non_pair_hand(
        Action.STAND, policy, hand, dealer, returns
    )

    assert episode_return == 1   # 17 > 16
    assert returns == {(HandType.HARD, 17, 10, True, Action.STAND): 1}


def test_play_non_pair_hit_once(monkeypatch):
    monkeypatch.setattr(
        blackjack.game, "deal_hand", lambda hand: hand.append(Card.THREE)
    )

    hand = [Card.TEN, Card.SIX]
    dealer = [Card.TEN, Card.SIX]

    policy = constant_policy(Action.HIT)
    returns = {}

    episode_return = play_non_pair_hand(
        Action.HIT, policy, hand, dealer, returns
    )

    assert get_hand_value(hand) == 22
    assert episode_return == -1
    assert returns == {
        (HandType.HARD, 16, 10, True, Action.HIT): -1,
        (HandType.HARD, 19, 10, False, Action.HIT): -1
    }

def test_play_non_pair_double(monkeypatch):

    monkeypatch.setattr(
        blackjack.game, "deal_hand", lambda hand: hand.append(Card.FOUR)
    )

    hand = [Card.TEN, Card.SIX]
    dealer = [Card.SEVEN, Card.FIVE]

    policy = constant_policy(Action.DOUBLE_DOWN)
    returns = {}

    episode_return = play_non_pair_hand(
        Action.DOUBLE_DOWN, policy, hand, dealer, returns
    )

    assert get_hand_value(hand) == 20
    assert episode_return == 2
    assert returns == {
        (HandType.HARD, 16, 7, True, Action.DOUBLE_DOWN): 2,
    }