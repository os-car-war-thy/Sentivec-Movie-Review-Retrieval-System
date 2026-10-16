"""Normalized vector search, exact score recomputation, and sentiment policies."""

from dataclasses import dataclass
from time import perf_counter

import faiss
import numpy as np

from .metrics import normalize


@dataclass
class SearchIndex:
    name: str
    index: object
    build_seconds: float
    serialized_bytes: int
    parameters: dict


def build_indexes(vectors, seed=42, names=("flat", "ivf", "hnsw", "ivfpq")):
    vectors = normalize(vectors)
    n, dimension = vectors.shape
    faiss.omp_set_num_threads(1)
    result, skipped = {}, []
    nlist = min(100, max(1, n // 40))
    for name in names:
        start = perf_counter()
        if name == "flat":
            index = faiss.IndexFlatIP(dimension)
            parameters = {"metric": "inner_product", "normalized": True}
        elif name == "ivf":
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension,
                                      nlist, faiss.METRIC_INNER_PRODUCT)
            index.cp.seed = seed
            index.train(vectors)
            index.nprobe = min(10, nlist)
            parameters = {"nlist": nlist, "nprobe": index.nprobe}
        elif name == "hnsw":
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 80
            index.hnsw.efSearch = 64
            parameters = {"M": 32, "efConstruction": 80, "efSearch": 64}
        elif name == "ivfpq":
            # 39 examples per 256 PQ centroids is FAISS's recommended minimum.
            if n < 39 * 256 or dimension % 8:
                skipped.append({"index": name, "reason":
                                "8-bit PQ requires >=9984 unique training vectors and d divisible by 8"})
                continue
            index = faiss.IndexIVFPQ(faiss.IndexFlatIP(dimension), dimension,
                                     nlist, 8, 8, faiss.METRIC_INNER_PRODUCT)
            index.cp.seed = seed
            index.pq.cp.seed = seed
            index.train(vectors)
            index.nprobe = min(10, nlist)
            parameters = {"nlist": nlist, "nprobe": index.nprobe, "m": 8, "bits": 8}
        else:
            raise ValueError(f"Unknown index {name}.")
        index.add(vectors)
        build_seconds = perf_counter() - start
        result[name] = SearchIndex(name, index, build_seconds,
                                   len(faiss.serialize_index(index)), parameters)
    return result, skipped


def retrieve(bundle, vectors, query, document_probabilities, query_probability=0.5,
             query_confidence=0.0, mode="semantic", k=20, multiplier=10,
             threshold=0.7, alpha=0.25):
    if mode not in {"semantic", "hard_filter", "gated_filter", "soft_rerank", "oracle_metadata"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    query = normalize(np.asarray(query).reshape(1, -1))
    if len(document_probabilities) != len(vectors) or k <= 0 or multiplier < 1:
        raise ValueError("Invalid corpus metadata, k, or multiplier.")
    search_k = min(len(vectors), max(k, k * multiplier))
    _, raw_ids = bundle.index.search(query, search_k)
    ids = raw_ids[0]
    ids = ids[ids >= 0]  # FAISS pads unavailable neighbors with -1.
    ids = np.unique(ids)
    cosine = vectors[ids] @ query[0]  # Original vectors, including for compressed indexes.
    agreement = (query_probability * document_probabilities[ids] +
                 (1 - query_probability) * (1 - document_probabilities[ids]))
    active = mode in {"hard_filter", "oracle_metadata"} or (
        mode == "gated_filter" and query_confidence >= threshold
    )
    if active:
        keep = (document_probabilities[ids] >= 0.5) == (query_probability >= 0.5)
        ids, cosine, agreement = ids[keep], cosine[keep], agreement[keep]
    scores = cosine + alpha * (agreement - 0.5) if mode == "soft_rerank" else cosine
    order = np.lexsort((ids, -scores))[:k]
    return {"indices": ids[order], "cosine": cosine[order], "ranking_scores": scores[order],
            "filter_active": bool(active), "candidate_count": len(raw_ids[0][raw_ids[0] >= 0])}


def proxy_reference(vectors, query, gold_labels, query_label, top_n=50):
    """Exact top-N *within* the gold sentiment subset, without a bounded pool."""
    eligible = np.flatnonzero(np.asarray(gold_labels) == query_label)
    if not len(eligible):
        raise ValueError("No reference documents of requested sentiment.")
    scores = vectors[eligible] @ normalize(np.asarray(query).reshape(1, -1))[0]
    return eligible[np.lexsort((eligible, -scores))[:top_n]]


def exact_neighbors(vectors, query, k):
    scores = vectors @ normalize(np.asarray(query).reshape(1, -1))[0]
    ids = np.arange(len(vectors))
    return ids[np.lexsort((ids, -scores))[:k]]


def ann_recall(retrieved, exact, k):
    return len(set(retrieved[:k]) & set(exact[:k])) / min(k, len(exact))
