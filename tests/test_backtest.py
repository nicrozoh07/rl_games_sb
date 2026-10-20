"""Backtest rows and metrics on policies whose result is known in advance."""
import numpy as np
import pytest

from rl_games_sb import backtest, investor

COST = investor.COST_BPS / 10_000


def test_rows_cover_every_day_of_the_period():
    days = backtest.run(lambda obs: 0, "valid")
    assert len(days) == len(investor.InvestorEnv(period="valid").dates)
    assert days["action"].isna().sum() == 1  # the last day: a value, no decision


def test_buy_and_hold_metrics():
    days = backtest.run(lambda obs: 2, "test")
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
