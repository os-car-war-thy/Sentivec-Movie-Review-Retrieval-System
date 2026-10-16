import numpy as np
import pytest

from sentivec.metrics import normalize, retrieval_metrics
from sentivec.retrieval import build_indexes, exact_neighbors, proxy_reference, retrieve


@pytest.fixture
def example():
    vectors = normalize([[1, 0], [.99, .1], [.9, .2], [.5, .9], [0, 1], [-1, 0]])
    indexes, _ = build_indexes(vectors, names=("flat",))
    return vectors, indexes["flat"], np.array([0, 0, 1, 1, 1, 0], dtype=float)


def test_gate_changes_actual_retrieval_and_falls_back(example):
    vectors, index, doc_p = example
    semantic = retrieve(index, vectors, [1, 0], doc_p, mode="semantic", k=2)
    uncertain = retrieve(index, vectors, [1, 0], doc_p, .99, .6, "gated_filter", 2,
                         threshold=.7)
    certain = retrieve(index, vectors, [1, 0], doc_p, .99, .9, "gated_filter", 2,
                       threshold=.7)
    assert not uncertain["filter_active"]
    np.testing.assert_array_equal(uncertain["indices"], semantic["indices"])
    assert certain["filter_active"]
    assert np.all(doc_p[certain["indices"]] == 1)
    assert not np.array_equal(certain["indices"], semantic["indices"])


def test_reference_searches_entire_sentiment_subset():
    vectors = normalize([[1, float(i) / 100] for i in range(100)])
    labels = np.array([0] * 99 + [1])
    reference = proxy_reference(vectors, [1, 0], labels, 1, top_n=1)
    assert reference.tolist() == [99]


def test_missing_faiss_neighbors_never_become_negative_iloc(example):
    vectors, index, doc_p = example
    result = retrieve(index, vectors, [1, 0], doc_p, k=20)
    assert len(result["indices"]) == len(vectors)
    assert np.all(result["indices"] >= 0)
    assert retrieval_metrics(result["indices"].tolist(), [0], (20,))["fill@20"] == .3


def test_cosine_is_recomputed_from_original_vectors(example):
    vectors, index, doc_p = example
    result = retrieve(index, vectors, [1, 0], doc_p, k=3)
    np.testing.assert_allclose(result["cosine"], vectors[result["indices"]] @ [1, 0], atol=1e-6)
    np.testing.assert_array_equal(result["indices"], exact_neighbors(vectors, [1, 0], 3))


def test_pq_is_explicitly_skipped_on_small_corpora():
    vectors = normalize(np.random.default_rng(42).normal(size=(100, 16)))
    indexes, skipped = build_indexes(vectors, names=("flat", "ivfpq"))
    assert list(indexes) == ["flat"]
    assert skipped[0]["index"] == "ivfpq"


def test_soft_reranking_can_change_order(example):
    vectors, index, doc_p = example
    result = retrieve(index, vectors, [1, 0], doc_p, .99, .99, "soft_rerank", k=2, alpha=2)
    assert doc_p[result["indices"][0]] == 1


def test_ivf_and_hnsw_return_valid_unique_candidates():
    vectors = normalize(np.random.default_rng(1).normal(size=(400, 16)))
    indexes, _ = build_indexes(vectors, seed=1)
    for index in indexes.values():
        result = retrieve(index, vectors, vectors[0], np.full(400, .5), k=20)
        assert len(set(result["indices"])) == 20
        assert result["indices"][0] == 0
