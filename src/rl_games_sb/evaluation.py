"""The report of part 13: the final agent against the baselines on one period.

Contents:
  - FINAL_PATH        : the frozen final agent (part 12)
  - AGENT, BUY_AND_HOLD, RANDOM : strategy names, as they appear in the report
  - load_final_agent(): the frozen models, after checking their hashes
  - compare()         : daily rows of every strategy on one period
  - per_run()         : one row of metrics per run
  - summary()         : one row per strategy
  - plot_equity()     : equity and drawdown curves

The same code runs the dry run on the valid period and, once, the final
evaluation on the test period, so the procedure cannot change in between.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter
from stable_baselines3 import DQN

from rl_games_sb import backtest

FINAL_PATH = Path("configs/investor_final.json")
RANDOM_SEEDS = range(200)

AGENT = "Agente RL"
BUY_AND_HOLD = "Comprar y mantener"
RANDOM = "Aleatorio"

# Several runs of a strategy become one row: the agent's few seeds all count
# (mean), the random agent's 200 runs are summarized by their median.
AGGREGATE = {AGENT: "mean", BUY_AND_HOLD: "mean", RANDOM: "median"}


def load_final_agent() -> tuple[dict[int, DQN], str]:
    """The frozen models by seed and their env id. Raises if a model file changed."""
    final = json.loads(FINAL_PATH.read_text(encoding="utf-8"))
    models = {}
    for seed, entry in final["models"].items():
        if hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"{entry['path']} is not the model frozen in {FINAL_PATH}")
        models[int(seed)] = DQN.load(entry["path"], device="cpu")
    return models, final["env"]


def compare(period: str) -> dict[str, list[pd.DataFrame]]:
    """backtest.run() rows of the agent's seeds, buy-and-hold, and 200 random runs."""
    models, env_id = load_final_agent()
    return {
        AGENT: [
            backtest.run(lambda obs, m=m: m.predict(obs, deterministic=True)[0], period, env_id)
            for m in models.values()
        ],
        BUY_AND_HOLD: [backtest.run(backtest.buy_and_hold, period)],
        RANDOM: [backtest.run(backtest.random_policy(seed), period) for seed in RANDOM_SEEDS],
    }


def per_run(runs: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """backtest.metrics() of every run, indexed by (strategy, run number)."""
    rows = {
        (name, i): backtest.metrics(days)
        for name, all_days in runs.items()
        for i, days in enumerate(all_days)
    }
    return pd.DataFrame(rows).T.rename_axis(["strategy", "run"])


def summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy from per_run(), aggregated as AGGREGATE says."""
    return pd.DataFrame({
        name: metrics.loc[name].agg(how) for name, how in AGGREGATE.items() if name in metrics.index
    }).T


def plot_equity(runs: dict[str, list[pd.DataFrame]], title: str) -> None:
    """Portfolio value (top) and drawdown (bottom): each agent seed, buy-and-hold,
    and the random runs' median with their 5-95 percentile band."""
    _, (top, bottom) = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, height_ratios=[2, 1])

    random_values = pd.concat([d["value"] for d in runs[RANDOM]], axis=1)
    top.fill_between(random_values.index, random_values.quantile(0.05, axis=1),
                     random_values.quantile(0.95, axis=1), color="tab:gray", alpha=0.25,
                     label=f"{RANDOM}: percentiles 5–95")
    top.plot(random_values.median(axis=1), color="tab:gray", label=f"{RANDOM}: mediana")

    for i, days in enumerate(runs[AGENT]):
        top.plot(days["value"], color="tab:blue", label=AGENT if i == 0 else None)
        bottom.plot(days["value"] / days["value"].cummax() - 1, color="tab:blue")
    bh = runs[BUY_AND_HOLD][0]["value"]
    top.plot(bh, color="black", ls="--", label=BUY_AND_HOLD)
    bottom.plot(bh / bh.cummax() - 1, color="black", ls="--")

    top.set_ylabel("valor del portafolio ($)")
    top.set_title(title)
    top.legend(loc="upper left")
    bottom.set_ylabel("drawdown")
    bottom.yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.show()
