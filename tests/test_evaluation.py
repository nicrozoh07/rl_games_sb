"""The report: frozen models are hash-checked, strategies aggregate as documented."""
import json

import pandas as pd
import pytest
from stable_baselines3 import DQN

from rl_games_sb import evaluation, registry


def save_small_model(path, seed=0):
    model = registry.create("dqn", "investor-v0", seed=seed, device="cpu", policy_kwargs={"net_arch": [8]})
    model.save(path)
    return model


def test_weights_hash_survives_a_new_save(tmp_path):
    """The .zip changes on every save; the weights hash must not."""
    model = save_small_model(tmp_path / "a.zip")
    model.save(tmp_path / "b.zip")
    loaded = [DQN.load(tmp_path / name, device="cpu") for name in ("a.zip", "b.zip")]

    assert evaluation.weights_sha256(loaded[0]) == evaluation.weights_sha256(loaded[1])
    assert evaluation.weights_sha256(loaded[0]) != evaluation.weights_sha256(save_small_model(tmp_path / "c.zip", seed=1))


def test_a_changed_model_is_refused(tmp_path, monkeypatch):
    path = tmp_path / "model.zip"
    frozen = evaluation.weights_sha256(save_small_model(path))
    manifest = tmp_path / "final.json"
    entry = {"path": str(path), "weights_sha256": frozen}
    manifest.write_text(json.dumps({"env": "investor-v0", "models": {"0": entry}}))
    monkeypatch.setattr(evaluation, "FINAL_PATH", manifest)
    assert list(evaluation.load_final_agent()[0]) == [0]

    save_small_model(path, seed=1)  # "retrained" after the freeze
    with pytest.raises(ValueError, match="not the model frozen"):
        evaluation.load_final_agent()


def test_summary_averages_the_agent_and_takes_the_random_median():
    metrics = pd.DataFrame(
        {"total_return": [0.1, 0.2, 0.6, 0.5, -0.3, 0.0, 0.1]},
        index=pd.MultiIndex.from_tuples(
            [(evaluation.AGENT, 0), (evaluation.AGENT, 1), (evaluation.AGENT, 2),
             (evaluation.BUY_AND_HOLD, 0),
             (evaluation.RANDOM, 0), (evaluation.RANDOM, 1), (evaluation.RANDOM, 2)],
            names=["strategy", "run"],
        ),
    )
    table = evaluation.summary(metrics)

    assert table.loc[evaluation.AGENT, "total_return"] == pytest.approx(0.3)
    assert table.loc[evaluation.BUY_AND_HOLD, "total_return"] == pytest.approx(0.5)
    assert table.loc[evaluation.RANDOM, "total_return"] == pytest.approx(0.0)
