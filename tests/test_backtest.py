"""Backtest rows and metrics on policies whose result is known in advance."""
import numpy as np
import pandas as pd
import pytest

from rl_games_sb import backtest, investor

COST = investor.COST_BPS / 10_000


def test_rows_cover_every_day_of_the_period():
    days = backtest.run(lambda obs: 0, "valid")
    assert len(days) == len(investor.InvestorEnv(period="valid").dates)
    assert days["action"].isna().sum() == 1  # the last day: a value, no decision


def test_buy_and_hold_metrics():
    days = backtest.run(backtest.buy_and_hold, "test")
    m = backtest.metrics(days)
    prices = investor.InvestorEnv(period="test").prices

    assert m["total_return"] == pytest.approx((1 - COST) * prices[-1] / prices[0] - 1)
    assert m["trades"] == 1
    assert m["exposure"] == 1
    assert m["costs"] == pytest.approx(COST * investor.INITIAL_CAPITAL)
    assert days["reward"].sum() == pytest.approx(np.log(1 + m["total_return"]))


def test_cash_metrics():
    m = backtest.metrics(backtest.run(lambda obs: 0, "test"))
    assert m["total_return"] == 0
    assert m["max_drawdown"] == 0
    assert m["trades"] == 0
    assert m["exposure"] == 0


def test_random_policy_is_reproducible_and_uses_every_action():
    first = backtest.run(backtest.random_policy(0), "valid")
    again = backtest.run(backtest.random_policy(0), "valid")
    other = backtest.run(backtest.random_policy(1), "valid")

    assert first["action"].equals(again["action"])
    assert not first["action"].equals(other["action"])
    shares = first["action"].value_counts(normalize=True)
    assert set(shares.index) == {0, 1, 2}
    assert shares.between(0.25, 0.42).all()  # ~1/3 each over 251 days


def test_metrics_match_the_activity_examples():
    """$100,000 -> $150,000 -> $120,000: +20% accumulated, -20% drawdown (part 2 of the activity)."""
    days = pd.DataFrame({
        "value": [100_000.0, 150_000.0, 120_000.0],
        "position": [1.0, 1.0, np.nan],
        "cost": [100.0, 0.0, np.nan],
    })
    m = backtest.metrics(days)

    assert m["total_return"] == pytest.approx(0.20)
    assert m["max_drawdown"] == pytest.approx(-0.20)
    assert m["costs"] == 100
    assert m["trades"] == 1
