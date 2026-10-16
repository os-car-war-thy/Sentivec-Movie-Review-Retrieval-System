import numpy as np
import pytest

from sentivec.metrics import (classification_metrics, fit_temperature, judged_ndcg,
                              normalize, paired_difference, polarity, retrieval_metrics)


def test_precision_penalizes_short_result_lists():
    result = retrieval_metrics([1], [1, 2, 3, 4, 5], (5,))
    assert result["P@5"] == pytest.approx(0.2)
    assert result["fill@5"] == pytest.approx(0.2)
    assert result["Recall@5"] == pytest.approx(0.2)


def test_metric_cutoffs_and_order():
    result = retrieval_metrics([9, 1, 2], [1, 2, 3, 4, 5], (3,))
    assert result["AP@3"] == pytest.approx((1 / 2 + 2 / 3) / 3)
    assert result["RR@3"] == pytest.approx(0.5)
    assert retrieval_metrics([1, 2, 3], [1, 2, 3], (3,))["nDCG@3"] == 1


def test_invalid_reference_and_duplicate_results_fail():
    with pytest.raises(ValueError):
        retrieval_metrics([1, 1], [1], (2,))
    with pytest.raises(ValueError):
        retrieval_metrics([1], [], (2,))


def test_neutral_mass_suppresses_gate_confidence():
    p, confidence, neutral = polarity([[0, 10, 1]])
    assert p[0] > 0.5
    assert confidence[0] < 0.01
    assert neutral[0] > 0.99


def test_temperature_reduces_overconfident_wrong_predictions():
    logits = np.array([[0, -100, 5], [5, -100, 0], [0, -100, 5], [5, -100, 0]])
    labels = [1, 0, 0, 1]
    temperature = fit_temperature(logits, labels)
    assert temperature > 1
    before = classification_metrics(labels, polarity(logits)[0])["binary_nll"]
    after = classification_metrics(labels, polarity(logits, temperature)[0])["binary_nll"]
    assert after < before


def test_paired_bootstrap_is_reproducible():
    a = paired_difference([1, 2, 3], [0, 1, 2], samples=200, seed=3)
    assert a == paired_difference([1, 2, 3], [0, 1, 2], samples=200, seed=3)
    assert a["mean"] == a["ci_low"] == a["ci_high"] == 1


def test_judged_ndcg_requires_complete_grades():
    assert judged_ndcg(["a", "b"], {"a": 2, "b": 1}, 2) == 1
    with pytest.raises(ValueError):
        judged_ndcg(["missing"], {"a": 2}, 1)


def test_normalization_rejects_zero_and_nonfinite_vectors():
    with pytest.raises(ValueError):
        normalize([[0, 0]])
    with pytest.raises(ValueError):
        normalize([[np.nan, 0]])
    np.testing.assert_allclose(np.linalg.norm(normalize([[3, 4]]), axis=1), 1)
