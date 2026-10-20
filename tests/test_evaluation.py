"""The report: frozen models are hash-checked, strategies aggregate as documented."""
import json

import pandas as pd
import pytest

from rl_games_sb import evaluation


def test_a_changed_model_is_refused(tmp_path, monkeypatch):
    model = tmp_path / "model.zip"
    model.write_bytes(b"retrained after seeing the test period")
    manifest = tmp_path / "final.json"
    manifest.write_text(json.dumps({"env": "investor-v0", "models": {"0": {"path": str(model), "sha256": "0" * 64}}}))
    monkeypatch.setattr(evaluation, "FINAL_PATH", manifest)

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
