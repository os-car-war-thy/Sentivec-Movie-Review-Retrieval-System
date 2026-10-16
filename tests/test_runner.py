"""Exercise artifact creation with explicit synthetic stand-ins, never research scores."""

import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from sentivec import experiment
from sentivec.config import profile
from sentivec.metrics import normalize


class SyntheticEncoder:
    repo = "synthetic-test-only"
    revision = "synthetic-test-only"
    maximum = 32

    def __init__(self, *args):
        pass

    def encode(self, texts):
        return normalize(np.array([np.frombuffer(hashlib.sha256(t.encode()).digest(), dtype=np.uint8)
                                   for t in texts], dtype=float) - 127)

    def cached(self, texts, cache_dir):
        return self.encode(texts)


class SyntheticClassifier:
    def __init__(self, *args):
        pass

    def scores(self, texts):
        raw = [hashlib.sha256(t.encode()).digest()[0] / 255 for t in texts]
        return np.array([[1 - value, -100, value] for value in raw])

    def cached(self, texts, cache_dir):
        return self.scores(texts)


def test_full_scenario_writes_complete_artifacts_with_synthetic_models(tmp_path, monkeypatch):
    dataset = tmp_path / "synthetic.csv"
    pd.DataFrame({"review": [f"Synthetic review {i}" for i in range(1000)],
                  "sentiment": ["positive" if i % 2 else "negative" for i in range(1000)]}).to_csv(
                      dataset, index=False)
    config = profile("smoke", imdb_path=str(dataset), cache_dir=str(tmp_path / "cache"))
    monkeypatch.setattr(experiment, "Encoder", SyntheticEncoder)
    monkeypatch.setattr(experiment, "SentimentModel", SyntheticClassifier)
    monkeypatch.setattr(experiment, "seed_everything", lambda seed: None)
    monkeypatch.setattr(experiment, "online_timing", lambda *args: (
        pd.DataFrame({"total_ms": [0.0]}), pd.DataFrame({"synthetic_test_only": [True]})))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)))
    directory = tmp_path / "synthetic-output"
    experiment.run_scenario(config, "imdb_en", 42, directory)
    expected = ["frozen_parameters.json", "dataset_provenance.json", "split_manifest.csv",
                "classifier_metrics.csv", "per_query_metrics.csv", "ranked_runs.csv",
                "paired_differences.csv", "table_effectiveness.tex", "table_latency.tex",
                "scalability.csv", "judgments_template.csv", "mixed_tone_diagnostics.csv"]
    assert all((directory / name).exists() for name in expected)
    frozen = json.loads((directory / "frozen_parameters.json").read_text())
    assert frozen["document_metadata"] == "predicted polarity, not dataset labels"
    rows = pd.read_csv(directory / "per_query_metrics.csv")
    assert rows.query_id.nunique() == config.test_size
    assert set(rows["mode"]) == {*experiment.MODES, "lexical"}
    assert rows.groupby(["index", "mode"]).size().eq(config.test_size).all()
