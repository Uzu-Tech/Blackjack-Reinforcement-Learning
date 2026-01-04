import numpy as np
import plotly.express as px

HAND_HARD = 0
HAND_SOFT = 1
HAND_PAIR = 2

ACTIONS = ["HIT", "STAND", "DOUBLE", "SPLIT"]

def plot_sliding_window(arr: np.ndarray, window_size: float):
    window = int(window_size * arr.size)
    smoothed = np.convolve(arr, np.ones(window)/window, mode="valid")

    fig = px.line(y=smoothed)
    fig.show()

def plot_strategy_hard(Q: np.ndarray):
    best = np.argmax(Q, axis=-1)

    # only take hard hands with 2 cards
    hard = best[HAND_HARD, :, :, 1]

    # valid player values 4–21, dealer 1–11
    fig = px.imshow(
        hard[5:21, 2:12],
        labels=dict(x="Dealer Upcard", y="Player Total", color="Action"),
        x=list(range(2, 12)),
        y=list(range(5, 21)),
        color_continuous_scale="Viridis"
    )
    fig.update_coloraxes(
        colorbar=dict(
            tickvals=[0,1,2,3], 
            ticktext=ACTIONS
        )
    )
    fig.show()

def plot_strategy_soft(Q: np.ndarray):
    best = np.argmax(Q, axis=-1)

    # only take hard hands with 2 cards
    soft = best[HAND_SOFT, :, :, 1]

    # valid player values 4–21, dealer 1–11
    fig = px.imshow(
        soft[13:21, 2:12],
        labels=dict(x="Dealer Upcard", y="Player Total", color="Action"),
        x=list(range(2, 12)),
        y=list(range(13, 21)),
        color_continuous_scale="Viridis"
    )
    fig.update_coloraxes(
        colorbar=dict(
            tickvals=[0,1,2,3], 
            ticktext=ACTIONS
        )
    )
    fig.show()

def plot_strategy_pair(Q: np.ndarray):
    best = np.argmax(Q, axis=-1)

    # only take hard hands with 2 cards
    pair = best[HAND_PAIR, :, :, 1]

    # valid player values 4–21, dealer 1–11
    fig = px.imshow(
        pair[2:12, 2:12],
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
