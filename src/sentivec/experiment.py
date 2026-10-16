"""Run the registered protocol; test labels never select parameters."""

import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np
import pandas as pd

from .config import MODELS, PROTOCOL_VERSION
from .data import (load_amazon, load_imdb, sha256_file, split_calibration_validation,
                   stratified_cap)
from .metrics import (bootstrap_mean, classification_metrics, fit_temperature,
                      paired_difference, polarity, retrieval_metrics)
from .models import Encoder, SentimentModel, synchronize
from .retrieval import ann_recall, build_indexes, exact_neighbors, proxy_reference, retrieve

MODES = ("semantic", "hard_filter", "gated_filter", "soft_rerank", "oracle_metadata")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
                          encoding="utf-8")


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(1)
    faiss.omp_set_num_threads(1)


def evaluate_vectors(corpus, queries, vectors, query_vectors, document_probabilities,
                     query_probabilities, confidences, indexes, config, threshold, alpha):
    """Pure vector evaluation; useful for synthetic regression tests as well."""
    rows, runs = [], []
    gold = corpus["label"].to_numpy()
    for i, query in queries.iterrows():
        reference = proxy_reference(vectors, query_vectors[i], gold, query.label,
                                     config.reference_size)
        exact = exact_neighbors(vectors, query_vectors[i], max(config.ks))
        for name, bundle in indexes.items():
            for mode in MODES:
                metadata = gold.astype(float) if mode == "oracle_metadata" else document_probabilities
                result = retrieve(bundle, vectors, query_vectors[i], metadata,
                                  query_probabilities[i], confidences[i], mode, max(config.ks),
                                  config.candidate_multiplier, threshold, alpha)
                ids = result["indices"]
                row = {"query_id": query.doc_id, "index": name, "mode": mode,
                       "query_label": int(query.label),
                       "predicted_query_label": int(query_probabilities[i] >= 0.5),
                       "query_confidence": float(confidences[i]),
                       "query_characters": len(query.text),
                       "reference_count": len(reference), "filter_active": result["filter_active"]}
                row.update({f"proxy_{key}": value for key, value in
                            retrieval_metrics(ids.tolist(), reference.tolist(), config.ks).items()})
                for k in config.ks:
                    row[f"sentiment_P@{k}"] = float(np.sum(gold[ids[:k]] == query.label) / k)
                if mode == "semantic":
                    row["ANN_recall@20"] = ann_recall(ids, exact, max(config.ks))
                rows.append(row)
                for rank, (doc, cosine, score) in enumerate(zip(
                    ids, result["cosine"], result["ranking_scores"]
                ), start=1):
                    runs.append({"query_id": query.doc_id, "doc_id": corpus.iloc[doc].doc_id,
                                 "index": name, "mode": mode, "rank": rank,
                                 "cosine": float(cosine), "ranking_score": float(score)})
    return pd.DataFrame(rows), pd.DataFrame(runs)


def tune_policy(corpus, tuning, vectors, query_vectors, doc_p, query_p, confidence,
                flat, config):
    records = []
    gold = corpus["label"].to_numpy()
    for mode, candidates in [("gated_filter", config.thresholds), ("soft_rerank", config.alphas)]:
        for value in candidates:
            metrics, activated, correct = [], [], []
            for i, query in tuning.iterrows():
                reference = proxy_reference(vectors, query_vectors[i], gold, query.label,
                                             config.reference_size)
                result = retrieve(flat, vectors, query_vectors[i], doc_p, query_p[i], confidence[i],
                                  mode=mode, k=max(config.ks), multiplier=config.candidate_multiplier,
                                  threshold=value if mode == "gated_filter" else 0.7,
                                  alpha=value if mode == "soft_rerank" else 0)
                metrics.append(retrieval_metrics(result["indices"].tolist(), reference.tolist(), (10,)))
                activated.append(result["filter_active"])
                correct.append(int(query_p[i] >= 0.5) == query.label)
            accepted = confidence >= value if mode == "gated_filter" else np.ones(len(tuning), bool)
            records.append({"mode": mode, "value": float(value),
                            "proxy_nDCG@10": float(np.mean([m["nDCG@10"] for m in metrics])),
                            "fill@10": float(np.mean([m["fill@10"] for m in metrics])),
                            "filter_activation": float(np.mean(activated)),
                            "confidence_coverage": float(accepted.mean()),
                            "selective_accuracy": (float(np.asarray(correct)[accepted].mean())
                                                   if accepted.any() else None)})
    table = pd.DataFrame(records)
    gated = table[table["mode"] == "gated_filter"].sort_values(
        ["proxy_nDCG@10", "value"], ascending=[False, False], kind="stable"
    ).iloc[0]
    soft = table[table["mode"] == "soft_rerank"].sort_values(
        ["proxy_nDCG@10", "value"], ascending=[False, True], kind="stable"
    ).iloc[0]
    return float(gated.value), float(soft.value), table


def add_bm25(corpus, queries, vectors, query_vectors, config):
    import re
    from rank_bm25 import BM25Okapi
    tokenize = lambda text: re.findall(r"\w+", text.casefold())  # noqa: E731
    bm25 = BM25Okapi([tokenize(text) for text in corpus.text], k1=1.5, b=0.75)
    rows, runs = [], []
    gold = corpus.label.to_numpy()
    for i, query in queries.iterrows():
        scores = bm25.get_scores(tokenize(query.text))
        ids = np.lexsort((np.arange(len(corpus)), -scores))[:max(config.ks)]
        reference = proxy_reference(vectors, query_vectors[i], gold, query.label, config.reference_size)
        row = {"query_id": query.doc_id, "index": "bm25", "mode": "lexical",
               "query_label": int(query.label), "query_characters": len(query.text)}
        row.update({f"proxy_{key}": value for key, value in
                    retrieval_metrics(ids.tolist(), reference.tolist(), config.ks).items()})
        for k in config.ks:
            row[f"sentiment_P@{k}"] = float(np.sum(gold[ids[:k]] == query.label) / k)
        rows.append(row)
        for rank, doc in enumerate(ids, 1):
            runs.append({"query_id": query.doc_id, "doc_id": corpus.iloc[doc].doc_id,
                         "index": "bm25", "mode": "lexical", "rank": rank,
                         "ranking_score": float(scores[doc])})
    return pd.DataFrame(rows), pd.DataFrame(runs)


def summarize(rows, config, seed):
    summaries, differences = [], []
    columns = [c for c in rows if c.startswith(("proxy_", "sentiment_P@", "ANN_recall"))]
    for (index, mode), frame in rows.groupby(["index", "mode"]):
        for metric in columns:
            values = frame[metric].dropna().to_numpy()
            if len(values):
                summaries.append({"index": index, "mode": mode, "metric": metric,
                                  **bootstrap_mean(values, config.bootstrap_samples, seed)})
        if index == "bm25" or mode == "semantic":
            continue
        baseline = rows[(rows["index"] == index) & (rows["mode"] == "semantic")]
        paired = frame.set_index("query_id").join(baseline.set_index("query_id"),
                                                   lsuffix="_new", rsuffix="_base", how="inner")
        if len(paired) != len(frame) or len(frame) != len(baseline):
            raise ValueError("Missing paired queries; comparisons would be biased.")
        for metric in ["proxy_nDCG@10", "sentiment_P@10"]:
            differences.append({"index": index, "mode": mode, "metric": metric,
                                "comparison": "mode minus semantic, same queries and index",
                                **paired_difference(paired[metric + "_new"],
                                                    paired[metric + "_base"],
                                                    config.bootstrap_samples, seed)})
    return pd.DataFrame(summaries), pd.DataFrame(differences)


def online_timing(encoder, classifier, temperature, queries, vectors, doc_p, indexes,
                  config, threshold, alpha, seed):
    """Uncached, warmed, single-request stage timings; no batching throughput claims."""
    records = []
    examples = queries.iloc[:config.timing_queries]
    encoder.encode([examples.iloc[0].text])
    classifier.scores([examples.iloc[0].text])
    for bundle in indexes.values():
        bundle.index.search(vectors[:1], min(len(vectors), max(config.ks) * config.candidate_multiplier))
    jobs = [(i, repeat, name, mode) for i in range(len(examples))
            for repeat in range(config.timing_repeats) for name in indexes
            for mode in ("semantic", "gated_filter")]
    random.Random(seed).shuffle(jobs)
    for i, repeat, name, mode in jobs:
        text = examples.iloc[i].text
        synchronize()
        start = perf_counter()
        if mode == "gated_filter":
            p, confidence, _ = polarity(classifier.scores([text]), temperature)
            q_p, q_c = p[0], confidence[0]
        else:
            q_p, q_c = 0.5, 0.0
        synchronize()
        classified = perf_counter()
        q_vector = encoder.encode([text])[0]
        synchronize()
        encoded = perf_counter()
        retrieve(indexes[name], vectors, q_vector, doc_p, q_p, q_c, mode,
                 max(config.ks), config.candidate_multiplier, threshold, alpha)
        finished = perf_counter()
        records.append({"query_id": examples.iloc[i].doc_id, "repeat": repeat, "index": name,
                        "mode": mode, "classification_ms": 1000 * (classified - start),
                        "encoding_ms": 1000 * (encoded - classified),
                        "search_and_policy_ms": 1000 * (finished - encoded),
                        "total_ms": 1000 * (finished - start)})
    raw = pd.DataFrame(records)
    summary = raw.groupby(["index", "mode"])["total_ms"].agg(
        mean_ms="mean", p50_ms="median", p95_ms=lambda x: x.quantile(0.95), requests="count"
    ).reset_index()
    return raw, summary


def scalability(vectors, query_vectors, seed):
    """Genuine nested corpus subsets, no vector replication or invented scales."""
    records = []
    order = np.random.default_rng(seed).permutation(len(vectors))
    sizes = sorted(set([n for n in [200, 500, 1000, 5000, 10000, 20000, len(vectors)]
                        if n <= len(vectors)]))
    for size in sizes:
        subset = vectors[order[:size]]
        indexes, skipped = build_indexes(subset, seed)
        for row in skipped:
            records.append({"n_documents": size, **row, "status": "skipped"})
        for name, bundle in indexes.items():
            latencies, recalls = [], []
            bundle.index.search(query_vectors[:1], min(200, size))
            for q in query_vectors[:20]:
                exact = exact_neighbors(subset, q, 20)
                for _ in range(3):
                    start = perf_counter()
                    result = retrieve(bundle, subset, q, np.full(size, 0.5), k=20)
                    latencies.append(1000 * (perf_counter() - start))
                    recalls.append(ann_recall(result["indices"], exact, 20))
            records.append({"n_documents": size, "index": name, "status": "complete",
                            "build_seconds": bundle.build_seconds,
                            "serialized_index_bytes": bundle.serialized_bytes,
                            "retained_vectors_bytes": subset.nbytes,
                            "search_policy_p50_ms": float(np.median(latencies)),
                            "search_policy_p95_ms": float(np.quantile(latencies, 0.95)),
                            "ANN_recall@20": float(np.mean(recalls))})
    return pd.DataFrame(records)


def validation_sensitivity(indexes, vectors, query_vectors, corpus, tuning,
                           doc_p, query_p, confidence, threshold, config):
    """Exploratory validation-only sweeps; restores predeclared final settings."""
    records = []
    for name, bundle in indexes.items():
        if name in {"ivf", "ivfpq"}:
            values, attribute = sorted(set([1, 4, min(16, bundle.index.nlist)])), "nprobe"
            values = [v for v in values if v <= bundle.index.nlist]
            owner = bundle.index
        elif name == "hnsw":
            values, attribute, owner = [16, 64, 128], "efSearch", bundle.index.hnsw
        else:
            values, attribute, owner = [0], None, None
        original = getattr(owner, attribute) if attribute else None
        try:
            for value in values:
                if attribute:
                    setattr(owner, attribute, value)
                for multiplier in [2, 5, 10, 20]:
                    scores, fills, recalls = [], [], []
                    for i, query in tuning.iloc[:50].iterrows():
                        result = retrieve(bundle, vectors, query_vectors[i], doc_p, query_p[i],
                                          confidence[i], "gated_filter", max(config.ks), multiplier,
                                          threshold)
                        ref = proxy_reference(vectors, query_vectors[i], corpus.label.to_numpy(),
                                              query.label, config.reference_size)
                        score = retrieval_metrics(result["indices"].tolist(), ref.tolist(), (10,))
                        scores.append(score["nDCG@10"])
                        fills.append(score["fill@10"])
                        semantic = retrieve(bundle, vectors, query_vectors[i], doc_p,
                                            mode="semantic", k=20, multiplier=multiplier)
                        recalls.append(ann_recall(semantic["indices"],
                                                  exact_neighbors(vectors, query_vectors[i], 20), 20))
                    records.append({"index": name, "parameter": attribute or "none", "value": value,
                                    "candidate_multiplier": multiplier,
                                    "proxy_nDCG@10": float(np.mean(scores)),
                                    "fill@10": float(np.mean(fills)),
                                    "ANN_recall@20": float(np.mean(recalls))})
        finally:
            if attribute:
                setattr(owner, attribute, original)
    return pd.DataFrame(records)


def run_scenario(config, scenario, seed, directory):
    directory.mkdir(parents=True)
    seed_everything(seed)
    if scenario == "imdb_en":
        splits, provenance = load_imdb(config.imdb_path, config, seed)
    else:
        splits, provenance = load_amazon(scenario.split("_")[1], config, seed)
    partitions = split_calibration_validation(splits.pop("validation"), seed)
    splits.update(partitions)
    corpus, calibration, tuning, test = [splits[k] for k in ("corpus", "calibration", "tuning", "test")]
    pd.concat([frame[["doc_id", "text_hash", "label"]].assign(split=name)
               for name, frame in splits.items()]).to_csv(directory / "split_manifest.csv", index=False)
    write_json(directory / "dataset_provenance.json", provenance)
    multilingual = not scenario.endswith("_en")
    primary = "xlm_roberta" if multilingual else "distilbert"
    classifier_keys = ["xlm_roberta"] if multilingual else ["vader", "roberta", "distilbert"]
    encoder = Encoder("multilingual_minilm" if multilingual else "minilm",
                      config.device, config.batch_size)
    vectors = encoder.cached(corpus.text.tolist(), config.cache_dir)
    tuning_vectors = encoder.cached(tuning.text.tolist(), config.cache_dir)
    indexes, skipped = build_indexes(vectors, seed)
    write_json(directory / "index_manifest.json", {
        "completed": {name: {"parameters": b.parameters, "build_seconds": b.build_seconds,
                              "serialized_bytes": b.serialized_bytes} for name, b in indexes.items()},
        "skipped": skipped, "retained_vectors_bytes": vectors.nbytes,
        "note": "Serialized bytes exclude retained original vectors, model weights and process RAM."
    })
    models, temperatures, classifier_rows = {}, {}, []
    for key in classifier_keys:
        print(f"  Calibrating {scenario}/{seed}/{key}", flush=True)
        model = SentimentModel(key, config.device, config.batch_size)
        models[key] = model
        cal_scores = model.cached(calibration.text.tolist(), config.cache_dir)
        temperature = fit_temperature(cal_scores, calibration.label.to_numpy())
        temperatures[key] = temperature
        val_scores = model.cached(tuning.text.tolist(), config.cache_dir)
        for calibrated in [False, True]:
            p, _, _ = polarity(val_scores, temperature if calibrated else 1.0)
            classifier_rows.append({"classifier": key, "split": "tuning", "calibrated": calibrated,
                                    **classification_metrics(tuning.label.to_numpy(), p)})
    classifier = models[primary]
    start = perf_counter()
    doc_logits = classifier.cached(corpus.text.tolist(), config.cache_dir)
    doc_inference_seconds = perf_counter() - start
    doc_p, _, _ = polarity(doc_logits, temperatures[primary])
    val_p, val_conf, _ = polarity(classifier.cached(tuning.text.tolist(), config.cache_dir),
                                 temperatures[primary])
    threshold, alpha, sensitivity = tune_policy(corpus, tuning, vectors, tuning_vectors,
                                                doc_p, val_p, val_conf, indexes["flat"], config)
    sensitivity.to_csv(directory / "threshold_alpha_validation.csv", index=False)
    # Freeze parameters on disk BEFORE encoding/scoring any test queries.
    write_json(directory / "frozen_parameters.json", {
        "primary_classifier": primary, "temperatures": temperatures,
        "threshold": threshold, "alpha": alpha, "candidate_multiplier": config.candidate_multiplier,
        "objective": "tuning proxy_nDCG@10 on FlatIP; conservative threshold / smallest alpha ties",
        "selection_data": "calibration and tuning only; target-domain validation used",
        "document_metadata": "predicted polarity, not dataset labels",
        "corpus_metadata_load_or_inference_seconds": doc_inference_seconds,
        "encoder_repo": encoder.repo, "encoder_revision": encoder.revision,
        "encoder_token_window": encoder.maximum,
    })
    validation_sensitivity(indexes, vectors, tuning_vectors, corpus, tuning, doc_p, val_p,
                           val_conf, threshold, config).to_csv(
                               directory / "index_candidate_validation.csv", index=False)
    test_vectors = encoder.cached(test.text.tolist(), config.cache_dir)
    error_rows = []
    for key, model in models.items():
        logits = model.cached(test.text.tolist(), config.cache_dir)
        for calibrated in [False, True]:
            p, _, _ = polarity(logits, temperatures[key] if calibrated else 1.0)
            classifier_rows.append({"classifier": key, "split": "test", "calibrated": calibrated,
                                    **classification_metrics(test.label.to_numpy(), p)})
        p, conf, neutral = polarity(logits, temperatures[key])
        for i, row in test.iterrows():
            error_rows.append({"query_id": row.doc_id, "classifier": key, "text": row.text,
                               "gold_label": int(row.label), "prediction": int(p[i] >= 0.5),
                               "positive_probability": float(p[i]), "gate_confidence": float(conf[i]),
                               "neutral_probability": float(neutral[i]),
                               "correct": bool((p[i] >= 0.5) == row.label),
                               "characters": len(row.text)})
    pd.DataFrame(classifier_rows).to_csv(directory / "classifier_metrics.csv", index=False)
    errors = pd.DataFrame(error_rows)
    errors.to_csv(directory / "classifier_predictions.csv", index=False)
    errors[~errors.correct].sort_values("gate_confidence", ascending=False).head(50).to_csv(
        directory / "high_confidence_errors.csv", index=False)
    errors.assign(length_bin=pd.cut(errors.characters, [0, 500, 1500, 3000, np.inf])).groupby(
        ["classifier", "gold_label", "length_bin"], observed=True
    ).agg(accuracy=("correct", "mean"), count=("correct", "size")).to_csv(
        directory / "error_slices.csv")
    test_p, test_conf, _ = polarity(classifier.cached(test.text.tolist(), config.cache_dir),
                                   temperatures[primary])
    rows, runs = evaluate_vectors(corpus, test, vectors, test_vectors, doc_p, test_p, test_conf,
                                  indexes, config, threshold, alpha)
    bm25_rows, bm25_runs = add_bm25(corpus, test, vectors, test_vectors, config)
    rows, runs = pd.concat([rows, bm25_rows], ignore_index=True), pd.concat(
        [runs, bm25_runs], ignore_index=True)
    rows.to_csv(directory / "per_query_metrics.csv", index=False)
    runs.to_csv(directory / "ranked_runs.csv", index=False)
    summary, differences = summarize(rows, config, seed)
    summary.to_csv(directory / "metrics_summary.csv", index=False)
    differences.to_csv(directory / "paired_differences.csv", index=False)
    # Preserve a compact main table suitable for a single paper column.
    compact = rows.groupby(["index", "mode"])[["proxy_nDCG@10", "sentiment_P@10"]].mean()
    compact.to_csv(directory / "table_effectiveness.csv")
    compact.to_latex(directory / "table_effectiveness.tex", float_format="%.3f")
    raw_timing, timing = online_timing(encoder, classifier, temperatures[primary], test,
                                      vectors, doc_p, indexes, config, threshold, alpha, seed)
    raw_timing.to_csv(directory / "online_timing_requests.csv", index=False)
    timing.to_csv(directory / "table_latency.csv", index=False)
    timing.to_latex(directory / "table_latency.tex", index=False, float_format="%.2f")
    scalability(vectors, test_vectors, seed).to_csv(directory / "scalability.csv", index=False)
    from .annotations import export_pool
    annotation_queries = stratified_cap(test, min(config.annotation_queries, len(test)), seed)
    export_pool(corpus, annotation_queries, runs, directory, config.annotation_depth, seed)
    from .robustness import run_robustness
    run_robustness(classifier, temperatures[primary], test, directory,
                   language=scenario.split("_")[1])
    del models, classifier, encoder
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return directory


def run_study(config):
    # Must be set before CUDA initialization, including before the first seed call.
    import os
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()[:10]
    root = Path(config.output_dir) / f"{config.profile}-{stamp}-{digest}"
    root.mkdir(parents=True)
    write_json(root / "config.json", config.to_dict())
    write_json(root / "status.json", {"status": "running", "protocol": PROTOCOL_VERSION})
    distributions = ["numpy", "pandas", "faiss-cpu", "torch", "transformers",
                     "sentence-transformers", "scikit-learn", "scipy"]
    environment = {"python": sys.version, "platform": platform.platform(),
                   "packages": {name: importlib.metadata.version(name) for name in distributions},
                   "model_revisions": MODELS, "threads": 1,
                   "determinism": "strict torch algorithms, FAISS single thread; hardware-dependent floats"}
    import torch
    environment["cuda"] = torch.version.cuda
    environment["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    environment["git_commit"] = git.stdout.strip() if git.returncode == 0 else None
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
    environment["git_dirty"] = bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
    write_json(root / "environment.json", environment)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                             text=True, check=True)
    (root / "environment.freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    try:
        for scenario in config.scenarios:
            for seed in config.seeds:
                print(f"Running {scenario}, seed={seed}, profile={config.profile}", flush=True)
                run_scenario(config, scenario, seed, root / scenario / f"seed-{seed}")
        write_json(root / "status.json", {"status": "automated_complete_human_judgments_pending",
                                         "protocol": PROTOCOL_VERSION,
                                         "note": "Proxy results alone do not establish human relevance."})
    except Exception as exc:
        write_json(root / "status.json", {"status": "failed", "error_type": type(exc).__name__,
                                         "error": str(exc)})
        raise
    finally:
        hashes = {str(path.relative_to(root)): sha256_file(path)
                  for path in sorted(root.rglob("*")) if path.is_file()
                  and path.name != "artifact_sha256.json"}
        write_json(root / "artifact_sha256.json", hashes)
    return root
