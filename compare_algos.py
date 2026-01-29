import datetime
import sqlite3

import numpy as np

from blackjack.agent import Agent
from blackjack.algorithms import ALGORITHMS_MAP

DATABASE_NAME = "compare_algos"
DATABASE_PATH = f"databases/{DATABASE_NAME}.sqlite3"
REGISTRY_NAME = "hyperparameter_registry"

SEED = 42
TRAIN_EPISODES = 1_000_000
TEST_EPISODES = 1_000_000
DECAY_FACTOR_STEP_SIZE = 10
DECAY_FACTOR_MAX = 1000


def save_hyperparameters(cursor: sqlite3.Cursor, conn: sqlite3.Connection) -> None:
    # Create table if doesn't exist
    cursor.execute(
        f"""--sql
        CREATE TABLE IF NOT EXISTS {REGISTRY_NAME} (
            experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
            train_episodes INTEGER,
            test_episodes INTEGER,
            algorithms TEXT,
            decay_factor_step_size INTEGER,
            decay_factor_max INTEGER,
            seed INTEGER,
            created DATETIME
        )"""
    )
    conn.commit()

    # Save hyperparameters
    cursor.execute(
        f"""--sql
        INSERT INTO {REGISTRY_NAME}(
            train_episodes, test_episodes, algorithms, decay_factor_step_size,
            decay_factor_max, seed, created
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            TRAIN_EPISODES,
            TEST_EPISODES,
            ",".join(list(ALGORITHMS_MAP.keys())),
            DECAY_FACTOR_STEP_SIZE,
            DECAY_FACTOR_MAX,
            SEED,
            datetime.datetime.now(),
        ),
    )
    conn.commit()


def create_experiment(cursor: sqlite3.Cursor, conn: sqlite3.Connection) -> str:
    """Create experiment results table and return table name."""
    # Get the most recent experiment_id from registry
    cursor.execute(f"""--sql SELECT MAX(experiment_id) FROM {REGISTRY_NAME}""")
    result = cursor.fetchone()
    experiment_id = result[0] if result and result[0] else 1

    # Create table with experiment_id in name
    table_name = f"{DATABASE_NAME}_exp_{experiment_id}"
    cursor.execute(
        f"""--sql
        CREATE TABLE IF NOT EXISTS {table_name} (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            algorithm TEXT NOT NULL,
            decay_factor INTEGER,
            mean_return REAL
        )"""
    )
    conn.commit()
    return table_name


def run_trial(
    cursor: sqlite3.Cursor,
    conn: sqlite3.Connection,
    table_name: str,
    trial_num: int,
    algo: str,
    decay_factor: int | None = None,
):
    """Run single trial, save to DB, print result, return next trial_num."""
    agent = Agent(algo_name=algo, Q_init=0, decay_factor=decay_factor, seed=SEED)
    agent.train(num_episodes=TRAIN_EPISODES)
    mean_return = float(np.mean(agent.evaluate(num_episodes=TEST_EPISODES)))
    cursor.execute(
        f"""--sql
        INSERT INTO {table_name}(algorithm, decay_factor, mean_return)
        VALUES (?, ?, ?)""",
        (algo, decay_factor, mean_return),
    )
    conn.commit()

    decay_str = f" (decay={decay_factor})" if decay_factor else ""
    print(f"Trial {trial_num}: {algo}{decay_str} = {mean_return:.6f}")


def run_experiment(
    cursor: sqlite3.Cursor, conn: sqlite3.Connection, table_name: str
) -> None:
    """Run experiments and save results to database."""
    trial_num = 1
    for algo in ALGORITHMS_MAP.keys():
        if algo == "Q Learning":
            run_trial(cursor, conn, table_name, trial_num, algo)
            trial_num += 1
        else:
            for decay_factor in np.arange(
                DECAY_FACTOR_STEP_SIZE, DECAY_FACTOR_MAX + 1, DECAY_FACTOR_STEP_SIZE
            ):
                run_trial(cursor, conn, table_name, trial_num, algo, int(decay_factor))
                trial_num += 1


def main():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    save_hyperparameters(cursor, conn)
    table_name = create_experiment(cursor, conn)
    run_experiment(cursor, conn, table_name)
    conn.close()
    print(f"\nResults saved to {DATABASE_PATH} (table_name: {table_name})")


if __name__ == "__main__":
    main()
