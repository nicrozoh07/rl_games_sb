"""The investor agent's market: the price data, its time split, the actions, and the env.

Contents:
  - POSITIONS       : action -> fraction of the capital invested in the asset
  - SPLITS          : first and last day of the train, valid and test periods
  - INITIAL_CAPITAL : the portfolio value at the start of every episode
  - COST_BPS        : default transaction cost, in basis points of the value traded
  - load()          : the common dataset, daily GOOGL candles
  - split()         : the rows of one period
  - InvestorEnv     : Gymnasium env, one episode = one pass over a period

Indicators are computed on the whole series and then split, so the first days
of a period can look back into the previous one: those prices were already
known then (see features.py and its lookahead tests).

Timing of one step, from day t to day t+1 (P = closing price):

    observe   indicators of day t (closes up to P[t]) and the current position
    decide    a target position, at the close of day t
    trade     at P[t], paying COST_BPS on the value bought or sold
    market    the asset returns P[t+1] / P[t] - 1, which the agent has not seen
    reward    log(value after the move / value before the trade)

An episode starts in cash and ends at the last day of its period, whose next
close belongs to the following period and is never used.
"""
from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
import pandas as pd
from gym_anytrading.datasets import STOCKS_GOOGL
from gymnasium import spaces

from rl_games_sb import features

# Action i keeps POSITIONS[i] of the portfolio in the asset and the rest in
# cash. No short selling, no leverage.
POSITIONS = (0.0, 0.5, 1.0)

# Consecutive, non-overlapping periods. Train fits the agent and the scaler,
# valid picks hyperparameters, and test is used once, for the final evaluation.
SPLITS = {
    "train": ("2009-05-22", "2014-12-31"),
    "valid": ("2015-01-01", "2015-12-31"),
    "test": ("2016-01-01", "2018-08-29"),
}

INITIAL_CAPITAL = 100_000.0
COST_BPS = 10.0  # 10 bps = 0.10%


def load() -> pd.DataFrame:
    """Daily GOOGL candles: timestamp, open, high, low, close, volume.

    The 2014 stock split is already adjusted for, so "Adj Close" equals
    "Close" on every row and is dropped.
    """
    return (
        STOCKS_GOOGL.drop(columns="Adj Close")
        .rename(columns=str.lower)
        .rename_axis("timestamp")
    )


def split(data: pd.DataFrame | pd.Series, period: str) -> pd.DataFrame | pd.Series:
    """The rows of `data` (indexed by date) inside one of SPLITS."""
    start, end = SPLITS[period]
    return data.loc[start:end]


class InvestorEnv(gym.Env):
    """Invest a share of the portfolio in GOOGL, one day at a time.

    Observation: the indicators of features.FEATURES for today, scaled with
    the train period's mean and std, followed by the fraction of the
    portfolio currently in the asset.

    The position is a target that is kept every day: between closes the
    asset's share drifts (at 50%, a rise makes it a bit more than 50%), and
    choosing 50% again rebalances it, paying the cost on that small trade.
    At 0% and 100% there is no drift, so holding costs nothing.

    The episode ends with truncated=True, not terminated: the market does not
    end there, the data does, so SB3 still bootstraps the last value.
    """

    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(self, period: str = "train", cost_bps: float = COST_BPS, render_mode: str | None = None):
        close = load()["close"]
        feats = features.compute(close)
        scaler = features.Scaler.fit(split(feats, "train"))

        self.dates = split(feats, period).index
        self.observations = scaler.transform(split(feats, period)).to_numpy(np.float32)
        self.prices = close.loc[self.dates].to_numpy()
        self.cost = cost_bps / 10_000

        self.action_space = spaces.Discrete(len(POSITIONS))
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(len(features.FEATURES) + 1,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.t = 0
        self.value = INITIAL_CAPITAL
        self.position = 0.0
        return self._obs(), self._info(cost=0.0)

    def step(self, action):
        target = POSITIONS[action]

        # At the close of day t: trade from the current position to the target.
        cost = self.cost * abs(target - self.position) * self.value
        invested = self.value - cost

        # From the close of t to the close of t+1: only the asset share moves.
        asset_return = self.prices[self.t + 1] / self.prices[self.t] - 1
        new_value = invested * (1 + target * asset_return)

        reward = float(np.log(new_value / self.value))
        self.position = target * (1 + asset_return) / (1 + target * asset_return)
        self.value = new_value
        self.t += 1

        truncated = self.t == len(self.prices) - 1
        return self._obs(), reward, False, truncated, self._info(cost=cost)

    def _obs(self) -> np.ndarray:
        return np.append(self.observations[self.t], np.float32(self.position))

    def _info(self, cost: float) -> dict:
        return {
            "date": self.dates[self.t],
            "value": self.value,
            "position": self.position,
            "cost": cost,
        }
