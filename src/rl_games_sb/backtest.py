"""Play a policy over one period of the investor env and measure the result.

Contents:
  - run()     : one episode, one row per trading day
  - metrics() : return, risk and trading statistics of an episode

A policy is any function from an observation to an action, e.g.
`lambda obs: model.predict(obs, deterministic=True)[0]` for an SB3 model, or
`lambda obs: 2` for "always 100%".
"""
from __future__ import annotations

from collections.abc import Callable

import gymnasium as gym
import numpy as np
import pandas as pd

from rl_games_sb import envs, investor


def run(
    policy: Callable[[np.ndarray], int], period: str, env_id: str = envs.INVESTOR_ENV_ID
) -> pd.DataFrame:
    """One row per day of the period, indexed by date.

    `value` is the portfolio at that day's close, before trading. `action`,
    `position` (the target) and `cost` are the decision taken at that close,
    and `reward` is what the env paid for it one day later. The last day has
    a value but no decision.
    """
    env = gym.make(env_id, period=period)
    obs, info = env.reset()
    rows, done = [], False
    while not done:
        action = int(policy(obs))
        obs, reward, terminated, truncated, next_info = env.step(action)
        rows.append({
            "date": info["date"],
            "value": info["value"],
            "action": action,
            "position": investor.POSITIONS[action],
            "cost": next_info["cost"],
            "reward": reward,
        })
        info, done = next_info, terminated or truncated
    rows.append({"date": info["date"], "value": info["value"]})
    return pd.DataFrame(rows).set_index("date")


def metrics(days: pd.DataFrame) -> dict[str, float]:
    """Statistics of an episode from run(). Rates are annualized over 252 days."""
    value = days["value"]
    daily = value.pct_change().dropna()
    held = days["position"].dropna()
    growth = value.iloc[-1] / value.iloc[0]
    volatility = daily.std() * np.sqrt(252)
    return {
        "final_value": value.iloc[-1],
        "total_return": growth - 1,
        "annual_return": growth ** (252 / len(daily)) - 1,
        "volatility": volatility,
        # Undefined for a policy that never invests (no volatility).
        "sharpe": daily.mean() * 252 / volatility if volatility > 0 else np.nan,
        "max_drawdown": (value / value.cummax() - 1).min(),
        # A trade is a change of target; rebalancing a 50% drift is not counted.
        "trades": int((held.diff().fillna(held.iloc[0]) != 0).sum()),
        "costs": days["cost"].sum(),
        "exposure": held.mean(),
    }
