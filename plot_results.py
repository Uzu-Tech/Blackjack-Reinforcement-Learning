import sqlite3
from typing import Optional

import plotly.graph_objects as go
import polars as pl
from pathlib import Path
from blackjack.visualizer import plot_strategy_hard, plot_strategy_pair, plot_strategy_soft
import numpy as np

import compare_algos

save_path = Path("plots")

def plot_compare_algos(experiment_id: int, save_path: Optional[Path] = None) -> None:
    conn = sqlite3.connect(compare_algos.DATABASE_PATH)
    table_name = f"{compare_algos.DATABASE_NAME}_exp_{experiment_id}"
    results_df = pl.read_database(f"SELECT * FROM {table_name}", conn)
    conn.close()

    if results_df.is_empty():
        print(f"No data found in table '{table_name}'.")
        return

    fig = go.Figure()
    for algo in compare_algos.ALGORITHMS_MAP.keys():
        if algo != "Q Learning":
            algo_df = (
                results_df
                .filter(pl.col("algorithm") == algo)
                .sort("decay_factor")
            )

            fig.add_trace(
                go.Scatter(
                    x=algo_df["decay_factor"].to_list(),
                    y=algo_df["mean_return"].to_list(),
                    mode="lines",
                    name=algo
                )
            )

    q_learning_return = results_df.filter(
        pl.col("algorithm") == "Q Learning"
    )["mean_return"][0]

    fig.add_trace(
        go.Scatter(
            x=[results_df["decay_factor"].min(), results_df["decay_factor"].max()],
            y=[q_learning_return, q_learning_return],
            mode="lines",
            name="Q Learning"
        )
    )

    fig.update_layout(
        title="Performance of Different Reinforcement Learning Algorithms in Blackjack",
        xaxis_title="Exploration Rate",
        yaxis_title="Expected Return",
        legend_title="Algorithm"
    )

    if save_path:
        fig.write_image(save_path / f"{table_name}.png", scale=4)


def plot_Q_table(Q_path: Path, save_path: Path):
    Q = np.load(Q_path)
    plot_strategy_hard(Q, save_path)
    plot_strategy_soft(Q, save_path)
    plot_strategy_pair(Q, save_path)


if __name__ == "__main__":
    agent_path = Path("trained_agents")
    agent_name = "Q_Learning__200000000"

    save_path = Path("plots") / agent_name
    save_path.mkdir(parents=True, exist_ok=True)
    plot_Q_table(agent_path / (agent_name + ".npy"), save_path)
