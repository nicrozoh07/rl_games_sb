"""State features for the investor agent: indicators of a daily closing-price series.

Contents:
  - FEATURES : the market feature names, in observation order
  - WARMUP   : days of history the longest indicator needs before its first value
  - compute(): one row of indicators per day, from closes up to that day only
  - Scaler   : z-score fitted on the training period, applied to any period

Timing: the row for day t uses closes up to and including t. The agent decides
at the close of t, and the return it earns is close[t+1] / close[t] - 1, which
these features never see. The current position is part of the state too, but
it comes from the env, not from prices.

Every indicator is scale-free (returns, ratios, RSI), so a price of $200 in
2009 and $1,200 in 2018 give comparable values. The scaler only brings them to
a similar range for the network, and its mean and std come from training data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ("ret_1", "ret_20", "gap_sma50", "vol_20", "rsi_14")

# The 50-day moving average needs 50 closes, so the first row with every
# feature defined is the 50th day.
WARMUP = 49


def compute(close: pd.Series) -> pd.DataFrame:
    """Indicators per day, without the first WARMUP days (not all defined yet).

    Every operation looks back only: diff/shift by positive lags, trailing
    rolling windows, and an exponential average that runs forward in time.
    """
    log_ret = np.log(close / close.shift(1))

    delta = close.diff()
    # Wilder's RSI: exponential averages of up moves and down moves.
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    features = pd.DataFrame(
        {
            "ret_1": log_ret,
            "ret_20": np.log(close / close.shift(20)),
            "gap_sma50": close / close.rolling(50).mean() - 1,
            "vol_20": log_ret.rolling(20).std(),
            "rsi_14": 1 - 1 / (1 + gain / loss),  # RSI / 100, in [0, 1]
        },
        index=close.index,
    )
    return features.iloc[WARMUP:]


@dataclass
class Scaler:
    """(x - mean) / std per feature, with mean and std from the training rows only."""

    mean: pd.Series
    std: pd.Series

    @classmethod
    def fit(cls, train: pd.DataFrame) -> Scaler:
        return cls(mean=train.mean(), std=train.std())

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return (features - self.mean) / self.std
