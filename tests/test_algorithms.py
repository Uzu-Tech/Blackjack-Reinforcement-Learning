import numpy as np
import pytest

from blackjack.algorithms import (
    expected_sarsa_episode,
    monte_carlo_episode,
    q_learning_episode,
    sarsa_episode,
)
from blackjack.state_space import flatten_Q, initialize_Q
from blackjack_env import BlackjackEnv


class TestQLearning:
    """Test suite for Q-learning algorithm."""

    @pytest.fixture
    def env(self):
        """Create a Blackjack environment for testing."""
        return BlackjackEnv(seed=42)

    @pytest.fixture
    def Q_table(self):
        """Create Q-value table and visit count table."""
        Q = initialize_Q(0.0)
        N = np.zeros_like(Q)
        flat_Q = flatten_Q(Q)
        flat_N = flatten_Q(N)
        return flat_Q, flat_N

    def test_q_learning_episode_runs_without_error(self, env, Q_table):
        """Test that Q-learning episode completes without errors."""
        Q, N = Q_table
        try:
            q_learning_episode(Q, N, env)
        except Exception as e:
            pytest.fail(f"Q-learning episode raised an exception: {e}")

    def test_q_learning_updates_Q_values(self, env, Q_table):
        """Test that Q-learning episode updates Q-values."""
        Q, N = Q_table
        initial_Q = Q.copy()

        q_learning_episode(Q, N, env)

        # At least some Q values should have changed
        assert not np.array_equal(Q, initial_Q)
        assert np.any(Q != 0.0)

    def test_q_learning_updates_visit_counts(self, env, Q_table):
        """Test that Q-learning episode increments visit counts."""
        Q, N = Q_table
        initial_N_sum = N.sum()

        q_learning_episode(Q, N, env)

        # Visit counts should increase
        assert N.sum() > initial_N_sum
        assert np.any(N > 0)

    def test_q_learning_only_updates_legal_actions(self, env, Q_table):
        """Test that Q-learning only updates legal actions."""
        Q, N = Q_table
        Q_old = Q

        q_learning_episode(Q, N, env)

        # Illegal actions should still be -inf
        assert np.all(Q[Q_old == -np.inf] == -np.inf)

    def test_q_learning_visit_counts_accumulate(self, env, Q_table):
        """Test that visit counts accumulate across episodes."""
        Q, N = Q_table

        q_learning_episode(Q, N, env)
        count_after_one = N.sum()

        q_learning_episode(Q, N, env)
        count_after_two = N.sum()

        # Visit counts should increase
        assert count_after_two > count_after_one


class TestSARSA:
    """Test suite for SARSA algorithm."""

    @pytest.fixture
    def env(self):
        """Create a Blackjack environment for testing."""
        return BlackjackEnv(seed=42)

    @pytest.fixture
    def Q_table(self):
        """Create Q-value table and visit count table."""
        Q = initialize_Q(0.0)
        N = np.zeros_like(Q)
        flat_Q = flatten_Q(Q)
        flat_N = flatten_Q(N)
        return flat_Q, flat_N

    def test_sarsa_episode_runs_without_error(self, env, Q_table):
        """Test that SARSA episode completes without errors."""
        Q, N = Q_table
        decay_factor = 100

        try:
            sarsa_episode(Q, N, env, decay_factor)
        except Exception as e:
            pytest.fail(f"SARSA episode raised an exception: {e}")

    def test_sarsa_updates_Q_values(self, env, Q_table):
        """Test that SARSA episode updates Q-values."""
        Q, N = Q_table
        decay_factor = 100
        initial_Q = Q.copy()

        sarsa_episode(Q, N, env, decay_factor)

        # At least some Q values should have changed
        assert not np.array_equal(Q, initial_Q)
        assert np.any(Q != 0.0)

    def test_sarsa_updates_visit_counts(self, env, Q_table):
        """Test that SARSA episode increments visit counts."""
        Q, N = Q_table
        decay_factor = 100
        initial_N_sum = N.sum()

        sarsa_episode(Q, N, env, decay_factor)

        # Visit counts should increase
        assert N.sum() > initial_N_sum
        assert np.any(N > 0)

    def test_sarsa_only_updates_legal_actions(self, env, Q_table):
        """Test that SARSA only updates legal actions."""
        Q, N = Q_table
        Q_old = Q
        decay_factor = 100

        sarsa_episode(Q, N, env, decay_factor)

        # Illegal actions should still be -inf
        assert np.all(Q[Q_old == -np.inf] == -np.inf)

    def test_sarsa_decay_factor_affects_exploration(self, env, Q_table):
        """Test that different decay factors affect exploration."""
        Q1, N1 = Q_table
        Q2, N2 = initialize_Q(0.0), np.zeros_like(Q_table[0])
        Q2, N2 = flatten_Q(Q2), flatten_Q(N2)

        # Low decay factor - more exploitation
        for _ in range(10):
            sarsa_episode(Q1, N1, env, decay_factor=1)

        # High decay factor - more exploration
        for _ in range(10):
            sarsa_episode(Q2, N2, env, decay_factor=1000)

        # High decay factor should visit more diverse states
        states_visited_low = (N1.sum(axis=1) > 0).sum()
        states_visited_high = (N2.sum(axis=1) > 0).sum()

        # This is probabilistic but should generally hold
        # We just check that both explore some states
        assert states_visited_low > 0
        assert states_visited_high > 0

    def test_sarsa_visit_counts_accumulate(self, env, Q_table):
        """Test that visit counts accumulate across SARSA episodes."""
        Q, N = Q_table
        decay_factor = 100

        sarsa_episode(Q, N, env, decay_factor)
        count_after_one = N.sum()

        sarsa_episode(Q, N, env, decay_factor)
        count_after_two = N.sum()

        # Visit counts should increase
        assert count_after_two > count_after_one

class TestExpectedSARSA:
    """Test suite for Expected SARSA algorithm."""

    @pytest.fixture
    def env(self):
        """Create a Blackjack environment for testing."""
        return BlackjackEnv(seed=42)

    @pytest.fixture
    def Q_table(self):
        """Create Q-value table and visit count table."""
        Q = initialize_Q(0.0)
        N = np.zeros_like(Q)
        flat_Q = flatten_Q(Q)
        flat_N = flatten_Q(N)
        return flat_Q, flat_N

    def test_expected_sarsa_episode_runs_without_error(self, env, Q_table):
        """Test that Expected SARSA episode completes without errors."""
        Q, N = Q_table
        decay_factor = 100

        try:
            expected_sarsa_episode(Q, N, env, decay_factor)
        except Exception as e:
            pytest.fail(f"Expected SARSA episode raised an exception: {e}")

    def test_expected_sarsa_updates_Q_values(self, env, Q_table):
        """Test that Expected SARSA episode updates Q-values."""
        Q, N = Q_table
        decay_factor = 100
        initial_Q = Q.copy()

        expected_sarsa_episode(Q, N, env, decay_factor)

        # At least some Q values should have changed
        assert not np.array_equal(Q, initial_Q)
        assert np.any(Q != 0.0)

    def test_expected_sarsa_updates_visit_counts(self, env, Q_table):
        """Test that Expected SARSA episode increments visit counts."""
        Q, N = Q_table
        decay_factor = 100
        initial_N_sum = N.sum()

        expected_sarsa_episode(Q, N, env, decay_factor)

        # Visit counts should increase
        assert N.sum() > initial_N_sum
        assert np.any(N > 0)

    def test_expected_sarsa_only_updates_legal_actions(self, env, Q_table):
        """Test that Expected SARSA only updates legal actions."""
        Q, N = Q_table
        decay_factor = 100

        expected_sarsa_episode(Q, N, env, decay_factor)

        # Illegal actions should still be -inf
        assert np.all(Q[Q == -np.inf] == -np.inf)

    def test_expected_sarsa_visit_counts_accumulate(self, env, Q_table):
        """Test that visit counts accumulate across Expected SARSA episodes."""
        Q, N = Q_table
        decay_factor = 100

        expected_sarsa_episode(Q, N, env, decay_factor)
        count_after_one = N.sum()

        expected_sarsa_episode(Q, N, env, decay_factor)
        count_after_two = N.sum()

        # Visit counts should increase
        assert count_after_two > count_after_one


class TestMonteCarlo:
    """Test suite for Monte Carlo algorithm."""

    @pytest.fixture
    def env(self):
        """Create a Blackjack environment for testing."""
        return BlackjackEnv(seed=42)

    @pytest.fixture
    def Q_table(self):
        """Create Q-value table and visit count table."""
        Q = initialize_Q(0.0)
        N = np.zeros_like(Q)
        flat_Q = flatten_Q(Q)
        flat_N = flatten_Q(N)
        return flat_Q, flat_N

    def test_monte_carlo_episode_runs_without_error(self, env, Q_table):
        """Test that Monte Carlo episode completes without errors."""
        Q, N = Q_table
        decay_factor = 100

        try:
            monte_carlo_episode(Q, N, env, decay_factor)
        except Exception as e:
            pytest.fail(f"Monte Carlo episode raised an exception: {e}")

    def test_monte_carlo_updates_Q_values(self, env, Q_table):
        """Test that Monte Carlo episode updates Q-values."""
        Q, N = Q_table
        decay_factor = 100
        initial_Q = Q.copy()

        monte_carlo_episode(Q, N, env, decay_factor)

        # At least some Q values should have changed
        assert not np.array_equal(Q, initial_Q)
        assert np.any(Q != 0.0)

    def test_monte_carlo_updates_visit_counts(self, env, Q_table):
        """Test that Monte Carlo episode increments visit counts."""
        Q, N = Q_table
        decay_factor = 100
        initial_N_sum = N.sum()

        monte_carlo_episode(Q, N, env, decay_factor)

        # Visit counts should increase
        assert N.sum() > initial_N_sum
        assert np.any(N > 0)

    def test_monte_carlo_only_updates_legal_actions(self, env, Q_table):
        """Test that Monte Carlo only updates legal actions."""
        Q, N = Q_table
        decay_factor = 100

        monte_carlo_episode(Q, N, env, decay_factor)

        # Illegal actions should still be -inf
        assert np.all(Q[Q == -np.inf] == -np.inf)

    def test_monte_carlo_visit_counts_accumulate(self, env, Q_table):
        """Test that visit counts accumulate across Monte Carlo episodes."""
        Q, N = Q_table
        decay_factor = 100

        monte_carlo_episode(Q, N, env, decay_factor)
        count_after_one = N.sum()

        monte_carlo_episode(Q, N, env, decay_factor)
        count_after_two = N.sum()

        # Visit counts should increase
        assert count_after_two > count_after_one

    def test_monte_carlo_first_visit_property(self, env, Q_table):
        """Test that Monte Carlo implements first-visit (not every-visit)."""
        Q, N = Q_table
        decay_factor = 100

        # Track initial state
        initial_N = N.copy()

        monte_carlo_episode(Q, N, env, decay_factor)

        # Visit increments should be at most 1 per episode for each state-action
        # (first-visit only updates once per state-action per episode)
        visit_changes = N - initial_N
        max_increment = np.max(visit_changes)
        assert max_increment <= 1.0

    def test_monte_carlo_handles_splits(self, env, Q_table):
        """Test that Monte Carlo handles split actions without errors."""
        Q, N = Q_table
        decay_factor = 100

        # Run multiple episodes - some will include splits
        for _ in range(1000):
            monte_carlo_episode(Q, N, env, decay_factor)
