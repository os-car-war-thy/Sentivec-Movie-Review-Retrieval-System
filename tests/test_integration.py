from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sentivec.annotations import evaluate_judgments, export_pool
from sentivec.config import profile
from sentivec.experiment import evaluate_vectors, summarize
from sentivec.metrics import normalize
from sentivec.models import aggregate_chunks, cache_array
from sentivec.retrieval import build_indexes


def test_synthetic_vector_evaluation_annotation_roundtrip(tmp_path):
    rng = np.random.default_rng(8)
    vectors = normalize(rng.normal(size=(100, 16)))
    query_vectors = normalize(rng.normal(size=(10, 16)))
    corpus = pd.DataFrame({"doc_id": [f"d-{i}" for i in range(100)],
                           "text": [f"Document {i}" for i in range(100)], "label": [0, 1] * 50})
    queries = pd.DataFrame({"doc_id": [f"q-{i}" for i in range(10)],
                            "text": [f"Query {i}" for i in range(10)], "label": [0, 1] * 5})
    indexes, _ = build_indexes(vectors, names=("flat", "hnsw"))
    config = profile("smoke")
    rows, runs = evaluate_vectors(corpus, queries, vectors, query_vectors,
                                  np.linspace(.1, .9, 100), np.linspace(.1, .9, 10),
                                  np.full(10, .9), indexes, config, .7, .25)
    assert len(rows) == 10 * 2 * 5
    assert rows.groupby(["index", "mode"]).size().eq(10).all()
    summary, differences = summarize(rows, config, 42)
    assert not summary.empty and not differences.empty
    rows.to_csv(tmp_path / "per_query_metrics.csv", index=False)
    runs.to_csv(tmp_path / "ranked_runs.csv", index=False)
    export_pool(corpus, queries, runs, tmp_path)
    judgments = pd.read_csv(tmp_path / "judgments_template.csv")
    assert "mode" not in judgments and "ranking_score" not in judgments
    # Synthetic grades test evaluator mechanics; these are not research judgments.
    judgments["semantic_grade"] = 1
    judgments["sentiment_match"] = 1
    judgments["adjudicator"] = "synthetic-test"
    path = tmp_path / "completed.csv"
    judgments.to_csv(path, index=False)
    human = evaluate_judgments(tmp_path, path)
    assert len(human) == 100
    assert human["human_joint_nDCG@10"].between(0, 1).all()
    judgments.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match="unjudged"):
        evaluate_judgments(tmp_path, path)


def test_chunk_aggregation_uses_owners_and_token_weights():
    values = np.array([[1, 0], [0, 1], [2, 2]], dtype=float)
    result = aggregate_chunks(values, np.array([0, 0, 1]), np.array([3, 1, 2]), 2)
    np.testing.assert_allclose(result, [[.75, .25], [2, 2]])


def test_cache_identity_changes_with_text_revision_and_protocol(tmp_path):
    calls = []

    def compute(texts):
        calls.append(texts)
        return np.ones((len(texts), 2))

    cache_array(tmp_path, {"revision": "a"}, ["first"], compute)
    cache_array(tmp_path, {"revision": "a"}, ["first"], compute)
    cache_array(tmp_path, {"revision": "b"}, ["first"], compute)
    cache_array(tmp_path, {"revision": "b"}, ["changed"], compute)
    assert len(calls) == 3


def test_notebook_is_valid_clean_python():
    import ast
    import nbformat

    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "notebooks" / "SentiVec_Reproducible_Study.ipynb", as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
            assert cell.outputs == []
            assert cell.execution_count is None
