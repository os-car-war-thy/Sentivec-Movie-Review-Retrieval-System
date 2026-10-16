"""Blinded candidate pools and strictly complete, independent human evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .metrics import judged_ndcg, retrieval_metrics

KEYS = ["query_id", "doc_id"]


def export_pool(corpus, queries, runs, directory, depth=10, seed=42):
    selected = runs[(runs.query_id.isin(queries.doc_id)) & (runs["rank"] <= depth)]
    candidates = selected[KEYS].drop_duplicates()
    rng = np.random.default_rng(seed)
    random_candidates = [{"query_id": query, "doc_id": doc}
                         for query in queries.doc_id
                         for doc in rng.choice(corpus.doc_id, min(5, len(corpus)), replace=False)]
    candidates = pd.concat([candidates, pd.DataFrame(random_candidates)]).drop_duplicates(KEYS)
    pool = candidates.merge(queries[["doc_id", "text", "label"]].rename(columns={
        "doc_id": "query_id", "text": "query_text", "label": "requested_sentiment"
    }), on="query_id", validate="many_to_one").merge(
        corpus[["doc_id", "text"]].rename(columns={"text": "document_text"}),
        on="doc_id", validate="many_to_one"
    )
    pool = pool.iloc[rng.permutation(len(pool))].reset_index(drop=True)
    pool["semantic_grade"] = ""
    pool["sentiment_match"] = ""
    pool["adjudicator"] = ""
    pool.to_csv(Path(directory) / "judgments_template.csv", index=False)
    # Provenance stays separate: do not show this file to annotators.
    selected.to_csv(Path(directory) / "annotation_pool_provenance.csv", index=False)


def validate_judgments(frame):
    required = set(KEYS + ["semantic_grade", "sentiment_match", "adjudicator"])
    if not required <= set(frame):
        raise ValueError(f"Judgments need these columns: {sorted(required)}")
    if frame.empty or frame.duplicated(KEYS).any():
        raise ValueError("Judgments must be nonempty and unique per query/document pair.")
    if not frame.semantic_grade.isin([0, 1, 2]).all():
        raise ValueError("Complete semantic grades 0, 1, 2 are required.")
    if not frame.sentiment_match.isin([0, 1]).all():
        raise ValueError("Complete sentiment matches 0 or 1 are required.")
    if frame.adjudicator.fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Record the adjudicator identifier for every final judgment.")


def evaluate_judgments(directory, judgments_path, k=10):
    directory = Path(directory)
    judgments = pd.read_csv(judgments_path)
    validate_judgments(judgments)
    template = pd.read_csv(directory / "judgments_template.csv")
    expected = set(map(tuple, template[KEYS].to_numpy()))
    actual = set(map(tuple, judgments[KEYS].to_numpy()))
    if not expected <= actual:
        raise ValueError(f"{len(expected - actual)} pool pairs are unjudged. Finish adjudication first.")
    # Extra judgments are excluded so the ideal ranking uses the registered pool.
    judgments = template[KEYS].merge(judgments, on=KEYS, validate="one_to_one")
    runs = pd.read_csv(directory / "ranked_runs.csv")
    settings = pd.read_csv(directory / "per_query_metrics.csv")[["index", "mode"]].drop_duplicates()
    rows = []
    for query_id, query_grades in judgments.groupby("query_id"):
        semantic = query_grades.set_index("doc_id").semantic_grade.to_dict()
        joint = (query_grades.set_index("doc_id").semantic_grade *
                 query_grades.set_index("doc_id").sentiment_match).to_dict()
        relevant = {doc for doc, grade in joint.items() if grade > 0}
        for setting in settings.itertuples(index=False):
            ranked = runs[(runs.query_id == query_id) & (runs["index"] == setting.index)
                          & (runs["mode"] == setting.mode)].sort_values("rank")
            ids = ranked.doc_id.tolist()[:k]
            row = {"query_id": query_id, "index": setting.index, "mode": setting.mode,
                   "human_semantic_nDCG@10": judged_ndcg(ids, semantic, k),
                   "human_joint_nDCG@10": judged_ndcg(ids, joint, k),
                   "pool_relevant_count": len(relevant)}
            if relevant:
                metrics = retrieval_metrics(ids, relevant, (k,))
                row.update({f"human_pool_{name}": value for name, value in metrics.items()})
            else:
                # Include no-positive queries with zero nDCG, but pooled AP/recall undefined.
                row.update({f"human_pool_{name}@{k}": None for name in ["AP", "Recall"]})
                row.update({f"human_pool_P@{k}": 0.0, f"human_pool_RR@{k}": 0.0,
                            f"human_pool_fill@{k}": len(ids) / k})
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(directory / "human_per_query_metrics.csv", index=False)
    result.groupby(["index", "mode"]).mean(numeric_only=True).to_csv(
        directory / "human_metrics_summary.csv")
    return result


def annotator_agreement(first, second):
    """Before adjudication: quadratic semantic kappa and binary tone kappa."""
    for frame in (first, second):
        if frame.duplicated(KEYS).any():
            raise ValueError("Duplicate annotation pairs.")
    paired = first.merge(second, on=KEYS, suffixes=("_a", "_b"), validate="one_to_one")
    if len(paired) != len(first) or len(first) != len(second):
        raise ValueError("Both annotators must judge the same complete pool.")
    return {"n_pairs": len(paired),
            "semantic_quadratic_kappa": cohen_kappa_score(
                paired.semantic_grade_a, paired.semantic_grade_b, weights="quadratic"),
            "sentiment_kappa": cohen_kappa_score(paired.sentiment_match_a, paired.sentiment_match_b)}
