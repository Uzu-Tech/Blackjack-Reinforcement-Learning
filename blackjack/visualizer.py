import numpy as np
import plotly.express as px
from pathlib import Path

HAND_HARD = 0
HAND_SOFT = 1
HAND_PAIR = 2

ACTIONS = ["HIT", "STAND", "DOUBLE", "SPLIT"]

def plot_strategy_hard(Q: np.ndarray, save_path: Path):
    best = np.argmax(Q, axis=-1)

    useable_ace = 0
    can_double = 1
    can_split = 0
    hard = best[:17, :, useable_ace, can_double, can_split]

    # valid player values 4–21, dealer 1–11
    fig = px.imshow(
        hard,
        labels=dict(x="Dealer Upcard", y="Player Total", color="Action"),
        x=list(range(2, 12)),
        y=list(range(4, 21)),
        color_continuous_scale="Viridis"
    )
    fig.update_coloraxes(
        colorbar=dict(
            tickvals=[0,1,2,3], 
            ticktext=ACTIONS
        )
    )
    fig.show()
    fig.write_image(save_path / "hard_strategy.png", format="png")

def plot_strategy_soft(Q: np.ndarray, save_path: Path):
    best = np.argmax(Q, axis=-1)

    # only take hard hands with 2 cards
    useable_ace = 1
    can_double = 1
    can_split = 0
    soft = best[9:, :, useable_ace, can_double, can_split]

    # valid player values 4–21, dealer 1–11
    fig = px.imshow(
        soft,
        labels=dict(x="Dealer Upcard", y="Player Total", color="Action"),
        x=list(range(2, 12)),
        y=list(range(13, 22)),
        color_continuous_scale="Viridis"
    )
    fig.update_coloraxes(
        colorbar=dict(
            tickvals=[0,1,2,3], 
            ticktext=ACTIONS
        )
    )
    fig.show()
    fig.write_image(save_path / "soft_strategy.png", format="png")

def plot_strategy_pair(Q: np.ndarray, save_path: Path):
    best: np.ndarray = np.argmax(Q, axis=-1)

    # only take hard hands with 2 cards
    useable_ace = 0
    can_double = 1
    can_split = 1
    pair = best[0:19:2, :, useable_ace, can_double, can_split]

    useable_ace = 1
    soft12 = best[8, :, useable_ace, can_double, can_split]
    pair = np.vstack([pair, soft12])

    # valid player values 4–21, dealer 1–11
    fig = px.imshow(
        pair,
        labels=dict(x="Dealer Upcard", y="Player Total", color="Action"),
        x=list(range(2, 12)),
        y=list(range(2, 12)),
        color_continuous_scale="Viridis"
    )
    fig.update_coloraxes(
        colorbar=dict(
            tickvals=[0,1,2,3], 
            ticktext=ACTIONS
        )
    )
    fig.show()
    fig.write_image(save_path / "pair_strategy.png", format="png")
