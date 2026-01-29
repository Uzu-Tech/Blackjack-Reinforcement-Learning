import numpy as np
import pytest

from blackjack.agent import Agent, evaluate_policy


class TestAgent:
    """Test the Agent class."""

    def test_agent_init(self):
        """Test Agent initialization with Q-learning."""
        agent = Agent("Q Learning", Q_init=0.0, decay_factor=None, seed=42)
        assert agent.Q is not None
        assert agent.N is not None
        assert agent.seed == 42
        assert agent.train_returns is None
        assert agent.test_returns is None

    def test_agent_init_sarsa_requires_decay(self):
        """Test that epsilon algorithms require decay_factor."""
        with pytest.raises(ValueError, match="decay factor must be specified"):
            Agent("SARSA", Q_init=0.0, decay_factor=None, seed=42)

    def test_agent_train(self):
        """Test that training runs and updates Q/N values."""
        agent = Agent("Q Learning", Q_init=0.0, decay_factor=None, seed=42)
        returns = agent.train(num_episodes=5)

        assert returns is not None
        assert len(returns) == 5
        assert agent.train_returns is not None
        # Q and N should have been updated (at least some entries > 0)
        assert np.any(agent.N > 0)

    def test_agent_evaluate(self):
        """Test that evaluate runs and returns test rewards."""
        agent = Agent("Q Learning", Q_init=0.0, decay_factor=None, seed=42)
        agent.train(num_episodes=10)
        test_returns = agent.evaluate(num_episodes=5)

        assert test_returns is not None
        assert len(test_returns) == 5
        assert agent.test_returns is not None

    def test_agent_with_sarsa(self):
        """Test Agent with SARSA algorithm and decay factor."""
        agent = Agent("SARSA", Q_init=0.0, decay_factor=100, seed=42)
        returns = agent.train(num_episodes=5)

        assert returns is not None
        assert len(returns) == 5
        assert np.any(agent.N > 0)

    def test_agent_with_expected_sarsa(self):
        """Test Agent with Expected SARSA algorithm."""
        agent = Agent("Expected SARSA", Q_init=0.0, decay_factor=100, seed=42)
        returns = agent.train(num_episodes=5)

        assert returns is not None
        assert len(returns) == 5

    def test_agent_with_monte_carlo(self):
        """Test Agent with Monte Carlo algorithm."""
        agent = Agent("Monte Carlo", Q_init=0.0, decay_factor=100, seed=42)
        returns = agent.train(num_episodes=5)

        assert returns is not None
        assert len(returns) == 5


class TestPolicy:
    """Test the test_policy function."""

    def test_policy_runs(self):
        """Test that evaluate_policy runs without error."""

        def dummy_policy(state):
            return 0

        returns = evaluate_policy(dummy_policy, num_episodes=3, seed=42)

        assert returns is not None
        assert len(returns) == 3

    def test_policy_reproducibility(self):
        """Test that evaluate_policy with same seed produces same returns."""

        def dummy_policy(state):
            return 0

        returns1 = evaluate_policy(dummy_policy, num_episodes=5, seed=42)
        returns2 = evaluate_policy(dummy_policy, num_episodes=5, seed=42)

        assert np.allclose(returns1, returns2)

    def test_greedy_policy_in_agent(self):
        """Test that greedy policy works within agent evaluation."""
        agent = Agent("Q Learning", Q_init=0.0, decay_factor=None, seed=42)
        agent.train(num_episodes=10)
        test_returns = agent.evaluate(num_episodes=3)

        assert len(test_returns) == 3
