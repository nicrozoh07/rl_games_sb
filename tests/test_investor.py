"""The dataset, the data split, and the env's accounting and lookahead safety."""
import hashlib
from pathlib import Path

import gym_anytrading.datasets
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_games_sb import investor

COST = investor.COST_BPS / 10_000


def test_dataset_is_the_documented_one():
    """The SHA-256 in data/README.md: any change to the data fails here."""
    csv = Path(gym_anytrading.datasets.__file__).parent / "data" / "STOCKS_GOOGL.csv"
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    assert digest == "1a3d459fdbcddb9362a157b07438b0fd83e6eb04406a136b5b0a10c5b0df107a"


def test_load_columns():
    df = investor.load()
    assert df.index.name == "timestamp"
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_periods_are_disjoint_ordered_and_complete():
    df = investor.load()
    train, valid, test = (investor.split(df, p) for p in ("train", "valid", "test"))

    assert train.index[-1] < valid.index[0]
    assert valid.index[-1] < test.index[0]
    assert len(train) + len(valid) + len(test) == len(df)


def run(env, actions):
    """Play a fixed list of actions: observations (reset included), rewards, infos."""
    obs, info = env.reset()
    seen, rewards, infos = [obs], [], [info]
    for a in actions:
        obs, reward, _, _, info = env.step(a)
        seen.append(obs)
        rewards.append(reward)
        infos.append(info)
    return np.array(seen), np.array(rewards), infos


# Scaled indicators have no natural bound, so the observation Box is unbounded.
@pytest.mark.filterwarnings("ignore:.*Box observation space m")
def test_env_passes_gymnasium_checks():
    check_env(investor.InvestorEnv(), skip_render_check=True)


@pytest.mark.parametrize("period", ["train", "valid", "test"])
def test_episode_covers_the_period_and_truncates(period):
    env = investor.InvestorEnv(period=period)
    env.reset()
    steps, done = 0, False
    while not done:
        _, _, terminated, truncated, info = env.step(0)
        steps += 1
        done = terminated or truncated

    assert truncated and not terminated
    assert steps == len(env.dates) - 1
    assert info["date"] == investor.split(investor.load(), period).index[-1]


def test_cash_never_changes():
    env = investor.InvestorEnv(period="test")
    _, rewards, infos = run(env, [0] * (len(env.dates) - 1))

    assert np.all(rewards == 0)
    assert infos[-1]["value"] == investor.INITIAL_CAPITAL


def test_all_in_is_buy_and_hold():
    """100% every day: one entry cost, then the asset's return, nothing else."""
    env = investor.InvestorEnv(period="test")
    _, rewards, infos = run(env, [2] * (len(env.dates) - 1))

    growth = env.prices[-1] / env.prices[0]
    assert infos[-1]["value"] == pytest.approx(investor.INITIAL_CAPITAL * (1 - COST) * growth)
    assert rewards.sum() == pytest.approx(np.log(infos[-1]["value"] / investor.INITIAL_CAPITAL))
    assert [i["cost"] for i in infos[1:3]] == [pytest.approx(COST * investor.INITIAL_CAPITAL), 0]


def test_costs_follow_the_value_traded():
    """Cash -> 100% -> 50%: pay on the whole value, then on the half that is sold."""
    env = investor.InvestorEnv(period="test")
    _, _, infos = run(env, [2, 1])

    assert infos[1]["cost"] == pytest.approx(COST * investor.INITIAL_CAPITAL)
    assert infos[2]["cost"] == pytest.approx(COST * 0.5 * infos[1]["value"])
    # At 50%, a price move makes the share drift away from exactly one half.
    assert infos[2]["position"] != 0.5


def test_env_does_not_see_future_prices(monkeypatch):
    """Replace every close after test day k with noise: everything up to day k is identical."""
    original = investor.load()
    k = 300
    cut = investor.split(original, "test").index[k]
    altered = original.copy()
    after = altered.index > cut
    altered.loc[after, "close"] = np.random.default_rng(0).uniform(1, 5000, after.sum())

    actions = list(np.random.default_rng(1).integers(0, 3, size=k)) + [2]
    results = []
    for data in (original, altered):
        monkeypatch.setattr(investor, "load", lambda data=data: data)
        results.append(run(investor.InvestorEnv(period="test"), actions))
    (obs_a, rew_a, _), (obs_b, rew_b, _) = results

    # Observations of days 0..k and rewards of the k steps between them.
    np.testing.assert_array_equal(obs_a[: k + 1], obs_b[: k + 1])
    np.testing.assert_array_equal(rew_a[:k], rew_b[:k])
    # The step from day k to k+1 uses the first altered close, so it must differ:
    # proof that the comparison above would catch a leak.
    assert rew_a[k] != rew_b[k]
