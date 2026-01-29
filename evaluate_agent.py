import sqlite3
from functools import partial
from pathlib import Path

import numpy as np

from blackjack.agent import evaluate_policy, evaluate_Q
from blackjack.basic_strategy import basic_strategy
from blackjack.policy import random
from blackjack.state_space import flatten_Q, initialize_Q
from train_agent import NUM_TRAIN_EPISODES

SEED = 42
NUM_TEST_EPISODES = 200_000_000  # Confidence interval of around +/- 0.02%

SAVED_AGENT_PATH = Path("trained_agents")
SAVED_AGENTS = (
    f"Q_Learning__{NUM_TRAIN_EPISODES}.npy",
    f"Expected_SARSA__{NUM_TRAIN_EPISODES}.npy",
)

DATABASE_NAME = "evaluate_agents"
DATABASE_PATH = f"databases/{DATABASE_NAME}.sqlite3"
REGISTRY_NAME = "hyperparameter_registry"


def save_hyperparameters(
    cursor: sqlite3.Cursor, conn: sqlite3.Connection, agents_evaluated: list[str]
) -> int | None:
    # Create table if doesn't exist
    cursor.execute(
        f"""--sql
        CREATE TABLE IF NOT EXISTS {REGISTRY_NAME} (
            experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            num_episodes INTEGER,
            agents_evaluated TEXT
        )"""
    )
    conn.commit()

    # Save hyperparameters
    cursor.execute(
        f"""--sql
        INSERT INTO {REGISTRY_NAME}(
            num_episodes, agents_evaluated
        ) VALUES (?, ?)""",
        (NUM_TEST_EPISODES, ", ".join(agents_evaluated)),
    )
    conn.commit()
    return cursor.lastrowid


def create_evaluation(
    cursor: sqlite3.Cursor, conn: sqlite3.Connection, experiment_id: int | None
) -> str:
    table_name = f"{DATABASE_NAME}_test_{experiment_id}"
    cursor.execute(
        f"""--sql
        CREATE TABLE IF NOT EXISTS {table_name} (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent TEXT NOT NULL,
            mean_return REAL
        )"""
    )
    conn.commit()
    return table_name


def evaluate_agent(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    agent: str,
    evaluate_func,
    table_name: str,
):
    mean_return = float(np.mean(evaluate_func(num_episodes=NUM_TEST_EPISODES)))
    cursor.execute(
        f"""--sql
        INSERT INTO {table_name} (agent, mean_return)
        VALUES (?, ?)
        """,
        (agent, mean_return),
    )
    conn.commit()


def download_saved_agents(files: tuple[str, ...]):
    return [np.load(SAVED_AGENT_PATH / file) for file in files]


if __name__ == "__main__":
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    agent_names = [
        filename.replace(".npy", "").replace("_", " ") for filename in SAVED_AGENTS
    ]

    agent_Qs = download_saved_agents(SAVED_AGENTS)

    experiment_id = save_hyperparameters(
        cursor,
        conn,
        agent_names + ["Basic Strategy", "Random"],
    )
    table_name = create_evaluation(cursor, conn, experiment_id)

    for Q, agent_name in zip(agent_Qs, agent_names):
        evaluate_func = partial(evaluate_Q, Q=Q, seed=SEED)
        evaluate_agent(cursor, conn, agent_name, evaluate_func, table_name)
        print(f"{agent_name} evaluated")

    evaluate_func = partial(evaluate_policy, policy=basic_strategy, seed=SEED)
    evaluate_agent(cursor, conn, "Basic Strategy", evaluate_func, table_name)
    print("Basic Stat evaluated")

    random_policy = partial(random, Q=flatten_Q(initialize_Q(0)))
    evaluate_func = partial(evaluate_policy, policy=random_policy, seed=SEED)
    evaluate_agent(cursor, conn, "Random Strategy", evaluate_func, table_name)
    print("Random evaluated")

    conn.close()

    print(f"\nResults saved to {DATABASE_PATH} (experiment_id: {experiment_id})")
