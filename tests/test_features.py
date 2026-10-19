"""Lookahead audit of the state features: the row for day t must not depend on later prices."""
import numpy as np
import pandas as pd
import pytest
from gym_anytrading.datasets import STOCKS_GOOGL

from rl_games_sb import features

CLOSE = STOCKS_GOOGL["Close"]


@pytest.mark.parametrize("t", [features.WARMUP, 500, 1500, len(CLOSE) - 2])
def test_future_prices_do_not_change_past_features(t):
    """Replace every price after day t with noise: rows up to t stay the same."""
    rng = np.random.default_rng(0)
    altered = CLOSE.copy()
    altered.iloc[t + 1 :] = rng.uniform(1, 5000, len(CLOSE) - t - 1)

    original = features.compute(CLOSE).loc[: CLOSE.index[t]]
    changed = features.compute(altered).loc[: CLOSE.index[t]]

    pd.testing.assert_frame_equal(original, changed)


def test_scaler_ignores_the_evaluation_period():
    """The scaler fitted on training rows is the same whatever the later prices are."""
    split = CLOSE.index[1500]
    altered = CLOSE.copy()
    altered.loc[split:] *= 3

    scaler = features.Scaler.fit(features.compute(CLOSE).loc[:split].iloc[:-1])
    scaler_altered = features.Scaler.fit(features.compute(altered).loc[:split].iloc[:-1])

    pd.testing.assert_series_equal(scaler.mean, scaler_altered.mean)
    pd.testing.assert_series_equal(scaler.std, scaler_altered.std)


def test_known_values():
    """A steady 1% daily rise: known returns, price above its average, RSI at its max."""
    close = pd.Series(100 * 1.01 ** np.arange(100))
    f = features.compute(close)

    assert list(f.columns) == list(features.FEATURES)
    assert not f.isna().any().any()
    assert np.allclose(f["ret_1"], np.log(1.01))
    assert np.allclose(f["ret_20"], 20 * np.log(1.01))
    assert (f["gap_sma50"] > 0).all()
    assert np.allclose(f["vol_20"], 0)
    assert np.allclose(f["rsi_14"], 1)
