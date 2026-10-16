"""Fixed-cutoff IR metrics, calibration diagnostics, and paired uncertainty."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, f1_score


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Expected a finite matrix.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero-length embedding.")
    return np.ascontiguousarray(vectors / norms, dtype=np.float32)


def fit_temperature(logits, labels):
    """Fit only on calibration data; binary polarity conditioned on non-neutral."""
    binary = np.asarray(logits, dtype=float)[:, [0, 2]]
    labels = np.asarray(labels, dtype=int)
    if len(binary) != len(labels) or set(labels) != {0, 1}:
        raise ValueError("Calibration requires aligned examples of both classes.")

    def loss(log_temperature):
        scaled = binary / np.exp(log_temperature)
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(labels)), labels]))

    result = minimize_scalar(loss, bounds=(np.log(0.05), np.log(10)), method="bounded")
    if not result.success:
        raise RuntimeError("Temperature fitting failed.")
    return float(np.exp(result.x))


def polarity(logits, temperature=1.0):
    """Columns are negative/neutral/positive. Neutral mass reduces filter confidence."""
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] != 3 or not np.isfinite(logits).all():
        raise ValueError("Expected finite [negative, neutral, positive] logits.")
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    full = softmax(logits / temperature, axis=1)
    p = softmax(logits[:, [0, 2]] / temperature, axis=1)[:, 1]
    confidence = np.maximum(p, 1 - p) * (1 - full[:, 1])
    return p, confidence, full[:, 1]


def classification_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    pred = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1 - p)
    correct = (pred == y).astype(float)
    ece = 0.0
    edges = np.linspace(0.5, 1.0, 11)
    for i in range(len(edges) - 1):
        mask = (confidence >= edges[i]) & (confidence <= edges[i + 1] if i == 9
                                          else confidence < edges[i + 1])
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "brier": float(np.mean((p - y) ** 2)),
            "binary_nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "ece_10_bins": float(ece)}


def retrieval_metrics(retrieved, relevant, ks=(5, 10, 20)):
    """AP@k denominator min(k, |R|); missing result ranks count as nonrelevant."""
    ids = list(retrieved)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate retrieved document IDs.")
    relevant = set(relevant)
    if not relevant:
        raise ValueError("Reference has no relevant documents; handle this explicitly.")
    out = {}
    for k in ks:
        if k < 1:
            raise ValueError("k must be positive.")
        hits = np.array([int(doc in relevant) for doc in ids[:k]], dtype=float)
        hits = np.pad(hits, (0, k - len(hits)))
        ranks = np.arange(1, k + 1)
        out[f"P@{k}"] = float(hits.sum() / k)
        out[f"Recall@{k}"] = float(hits.sum() / len(relevant))
        out[f"AP@{k}"] = float(np.sum(np.cumsum(hits) / ranks * hits) / min(k, len(relevant)))
        dcg = float(np.sum(hits / np.log2(ranks + 1)))
        idcg = float(np.sum(1 / np.log2(np.arange(1, min(k, len(relevant)) + 1) + 1)))
        out[f"nDCG@{k}"] = dcg / idcg
        positions = np.flatnonzero(hits)
        out[f"RR@{k}"] = 1 / (int(positions[0]) + 1) if len(positions) else 0.0
        out[f"fill@{k}"] = min(k, len(ids)) / k
    return out


def judged_ndcg(retrieved, grades, k=10):
    """Graded nDCG over a completely judged candidate pool; no unjudged-as-zero."""
    if not grades or any(doc not in grades for doc in retrieved[:k]):
        raise ValueError("Every evaluated result must be judged.")
    gains = np.array([2 ** grades[doc] - 1 for doc in retrieved[:k]], dtype=float)
    ideal = np.sort([2 ** value - 1 for value in grades.values()])[::-1][:k]
    if not np.any(ideal):
        return 0.0
    return float(np.sum(gains / np.log2(np.arange(len(gains)) + 2)) /
                 np.sum(ideal / np.log2(np.arange(len(ideal)) + 2)))


def bootstrap_mean(values, samples=1000, seed=42):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires nonempty finite observations.")
    rng = np.random.default_rng(seed)
    # Draw in batches to bound memory with larger query sets.
    means = np.concatenate([
        values[rng.integers(0, len(values), size=(min(128, samples - start), len(values)))].mean(1)
        for start in range(0, samples, 128)
    ])
    low, high = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci_low": float(low), "ci_high": float(high),
            "n_queries": len(values)}


def paired_difference(left, right, samples=1000, seed=42):
    """Inputs must already have the same query order; estimates left minus right."""
    if len(left) != len(right):
        raise ValueError("Paired observations must have equal length.")
    return bootstrap_mean(np.asarray(left) - np.asarray(right), samples, seed)
