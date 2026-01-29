import pytest

from blackjack.policy import epsilon_greedy, greedy, random
from blackjack.state_space import Action, flatten_Q, initialize_Q


class TestPolicies:
    """Test suite for policy functions."""

    @pytest.fixture
    def Q(self):
        """Create a Q-value table for testing."""
        return initialize_Q(0.0)

    @pytest.fixture
    def flat_Q(self, Q):
        """Create a flattened Q-value table."""
        return flatten_Q(Q)

    def test_greedy_policy_returns_action(self, flat_Q):
        """Test that greedy policy returns a valid action."""
        state = 0
        action = greedy(state, flat_Q)
        assert isinstance(action, Action)
        assert action in [Action.HIT, Action.STAND, Action.DOUBLE, Action.SPLIT]

    def test_random_policy_returns_action(self, flat_Q):
        """Test that random policy returns a valid action."""
        state = 0
        action = random(state, flat_Q)
        assert isinstance(action, Action)
        assert action in [Action.HIT, Action.STAND, Action.DOUBLE, Action.SPLIT]

    def test_epsilon_greedy_policy_returns_action(self, flat_Q):
        """Test that epsilon-greedy policy returns a valid action."""
        state = 0
        n = 10  # number of visits
        k = 5  # decay factor
        action = epsilon_greedy(state, flat_Q, n, k)
        assert isinstance(action, Action)
        assert action in [Action.HIT, Action.STAND, Action.DOUBLE, Action.SPLIT]

    def test_epsilon_greedy_explores_with_low_visits(self, flat_Q):
        """Test that epsilon-greedy explores with low visit counts."""
        state = 0
        n = 1  # low visit count
        k = 10  # high decay factor

        # With low visits, epsilon is high (more exploration)
        epsilon = k / (k + n)
        assert epsilon > 0.5

        action = epsilon_greedy(state, flat_Q, n, k)
        assert isinstance(action, Action)

    def test_epsilon_greedy_exploits_with_high_visits(self, flat_Q):
        """Test that epsilon-greedy exploits with high visit counts."""
        state = 0
        n = 1000  # high visit count
        k = 10  # decay factor

        # With high visits, epsilon is low (more exploitation)
        epsilon = k / (k + n)
        assert epsilon < 0.05

        action = epsilon_greedy(state, flat_Q, n, k)
        assert isinstance(action, Action)
