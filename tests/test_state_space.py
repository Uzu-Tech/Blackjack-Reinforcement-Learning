import numpy as np

from blackjack.state_space import (
    NUM_HAND_VALUES,
    NUM_UPCARDS,
    Action,
    flatten_Q,
    initialize_Q,
)


class TestStateSpace:
    """Test suite for state space initialization."""

    def test_initialize_Q_shape(self):
        """Test that Q-value table has correct shape."""
        Q = initialize_Q(0.0)
        assert Q.shape == (NUM_HAND_VALUES, NUM_UPCARDS, 2, 2, 2, len(Action))

    def test_initialize_Q_with_value(self):
        """Test that Q-value table is initialized with correct value."""
        init_value = 1.5
        Q = initialize_Q(init_value)
        # Check some legal states are initialized correctly
        legal_value = Q[17, 5, 0, 0, 0, Action.HIT]
        assert legal_value == init_value

    def test_initialize_Q_illegal_actions_are_inf(self):
        """Test that illegal actions are marked with -inf."""
        Q = initialize_Q(0.0)
        # Can't split when can_split=0
        illegal_value = Q[10, 5, 0, 0, 0, Action.SPLIT]
        assert illegal_value == -np.inf
        illegal_value = Q[13, 1, 0, 1, 0, Action.SPLIT]
        assert illegal_value == -np.inf
        illegal_value = Q[17, 8, 0, 0, 0, Action.DOUBLE]
        assert illegal_value == -np.inf

    def test_flatten_Q(self):
        """Test that Q-value table is flattened correctly."""
        Q = initialize_Q(0.0)
        flat_Q = flatten_Q(Q)

        # Check shape
        expected_states = NUM_HAND_VALUES * NUM_UPCARDS * 2 * 2 * 2
        assert flat_Q.shape == (expected_states, len(Action))

        # Check values match
        state_idx = 0
        assert flat_Q[state_idx, Action.HIT] == Q[0, 0, 0, 0, 0, Action.HIT]
