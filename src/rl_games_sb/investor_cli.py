"""Command line for the investor agent: `uv run investor <command>`.

Commands:
  train     : train the DQN agent with fixed seeds (default: the final agent, reward v1)
  check     : compare trained models with the frozen final agent, weight by weight
  evaluate  : the report of part 13 on one period (agent, buy-and-hold, random)
  baselines : the baselines alone on one period

Every command runs from the repository root, like `rlgames-sb`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN

from rl_games_sb import backtest, config, evaluation, investor, registry

# Reward version -> its config and the name its saves use (the notebooks expect these names).
REWARDS = {
    "v0": ("configs/investor_dqn.yaml", "investor-v0"),
    "v1": ("configs/investor_dqn_penalized.yaml", "investor-pen400"),
}
PERIODS = tuple(investor.SPLITS)


def _save_path(folder: Path, name: str, seed: int) -> Path:
    return folder / f"dqn_{name}_seed{seed}.zip"


def _show(table: pd.DataFrame) -> None:
    shown = table.drop(columns="final_value", errors="ignore").copy()
    for col in ("total_return", "annual_return", "volatility", "max_drawdown"):
        shown[col] = shown[col].map("{:+.1%}".format)
    shown["sharpe"] = shown["sharpe"].map("{:.2f}".format)
    shown["trades"] = shown["trades"].map("{:.0f}".format)
    shown["costs"] = shown["costs"].map("${:,.0f}".format)
    shown["exposure"] = shown["exposure"].map("{:.2f}".format)
    print(shown.to_string())


def cmd_train(args: argparse.Namespace) -> None:
    config_path, name = REWARDS[args.reward]
    cfg = config.load(config_path)
    for seed in args.seeds:
        path = _save_path(args.out, name, seed)
        if path.exists() and not args.force:
            print(f"{path} exists, skipping (--force to overwrite)")
            continue
        start = time.time()
        model = registry.create(cfg.algo, cfg.env, seed=seed, **cfg.hyperparams)
        model.learn(cfg.timesteps)
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        print(f"{path}: {cfg.timesteps:,} steps on {cfg.env}, seed {seed}, {time.time() - start:.0f} s")


def cmd_check(args: argparse.Namespace) -> None:
    final = json.loads(evaluation.FINAL_PATH.read_text(encoding="utf-8"))
    for seed, entry in final["models"].items():
        path = args.dir / Path(entry["path"]).name
        if not path.exists():
            print(f"seed {seed}: {path} not found")
            continue
        same = evaluation.weights_sha256(DQN.load(path, device="cpu")) == entry["weights_sha256"]
        print(f"seed {seed}: {path} {'has the frozen weights' if same else 'DIFFERS from the frozen weights'}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    metrics = evaluation.per_run(evaluation.compare(args.period))
    print(f"\n{args.period}: agent mean of seeds, random median of {len(evaluation.RANDOM_SEEDS)} runs\n")
    _show(evaluation.summary(metrics))
    print(f"\n{evaluation.AGENT}, by seed\n")
    _show(metrics.loc[evaluation.AGENT])


def cmd_baselines(args: argparse.Namespace) -> None:
    fixed = {"buy and hold": backtest.buy_and_hold, "always cash": lambda obs: 0, "always 50%": lambda obs: 1}
    rows = {name: backtest.metrics(backtest.run(policy, args.period)) for name, policy in fixed.items()}
    random_runs = pd.DataFrame([
        backtest.metrics(backtest.run(backtest.random_policy(seed), args.period)) for seed in evaluation.RANDOM_SEEDS
    ])
    for q in (0.05, 0.5, 0.95):
        rows[f"random, percentile {q:.0%}"] = random_runs.quantile(q)
    print(f"\n{args.period}: random seeds 0-{len(evaluation.RANDOM_SEEDS) - 1}, percentiles column by column\n")
    _show(pd.DataFrame(rows).T)


def main() -> None:
    parser = argparse.ArgumentParser(prog="investor", description="Investor agent: train, check, evaluate.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Train the agent with fixed seeds")
    p.add_argument("--reward", choices=REWARDS, default="v1", help="v1 is the final agent (default)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--out", type=Path, default=registry.SAVE_DIR, help="Folder for the models (default: saves)")
    p.add_argument("--force", action="store_true", help="Overwrite existing models")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("check", help="Compare trained models with the frozen final agent")
    p.add_argument("--dir", type=Path, default=registry.SAVE_DIR, help="Folder with the models (default: saves)")
    p.set_defaults(func=cmd_check)

    for name, func, text in [
        ("evaluate", cmd_evaluate, "Agent against the baselines"),
        ("baselines", cmd_baselines, "Baselines only"),
    ]:
        p = sub.add_parser(name, help=f"{text} on one period")
        p.add_argument("--period", choices=PERIODS, default="valid")
        p.set_defaults(func=func)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
