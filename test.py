from blackjack.agent import test_policy
from blackjack.basic_strategy import basic_strategy
import os

if __name__ == "__main__":
   # Q = np.load("Q_opt.npy")

   # plot_strategy_hard(Q)
  #  plot_strategy_soft(Q)
   # plot_strategy_pair(Q)

    print(f"Reward: {round(100 * test_policy(basic_strategy, 10_000_000), 5)}%")