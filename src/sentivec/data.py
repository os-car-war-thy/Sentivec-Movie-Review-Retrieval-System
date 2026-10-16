"""Dataset acquisition, canonical text identity, and disjoint stratified splits."""

import hashlib
import html
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import AMAZON_REVISION, KAGGLE_DATASET


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_text(value):
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def canonicalize(frame, namespace):
    frame = frame.copy()
    frame["text"] = frame["text"].map(clean_text)
    frame = frame[frame["text"].str.len() > 0].copy()
    if not frame["label"].isin([0, 1]).all():
        raise ValueError("Unexpected labels; expected binary 0/1.")
    frame["text_hash"] = frame["text"].map(lambda s: hashlib.sha256(s.encode()).hexdigest())
    conflicts = frame.groupby("text_hash")["label"].nunique().gt(1)
    if conflicts.any():
        raise ValueError(f"{conflicts.sum()} duplicate texts have conflicting labels.")
    frame = frame.drop_duplicates("text_hash").sort_values("text_hash").reset_index(drop=True)
    frame["doc_id"] = namespace + ":" + frame["text_hash"]
    return frame


def stratified_cap(frame, limit, seed):
    if limit >= len(frame):
        return frame.sort_values("doc_id").reset_index(drop=True)
    sample, _ = train_test_split(frame, train_size=limit, stratify=frame["label"],
                                 random_state=seed)
    return sample.sort_values("doc_id").reset_index(drop=True)


def assert_disjoint(splits):
    seen = set()
    for name, frame in splits.items():
        identities = set(frame["text_hash"])
        if seen.intersection(identities):
            raise ValueError(f"Text leakage into split {name}.")
        if len(identities) != len(frame):
            raise ValueError(f"Duplicate text in split {name}.")
        seen.update(identities)
        if set(frame["label"]) != {0, 1}:
            raise ValueError(f"Both sentiment classes are required in {name}.")


def download_imdb(destination="data/raw"):
    """Explicit opt-in download; Kaggle may require its own account authentication."""
    import shutil
    import kagglehub

    downloaded = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    files = list(downloaded.rglob("IMDB Dataset.csv"))
    if len(files) != 1:
        raise FileNotFoundError("Expected exactly one IMDB Dataset.csv in Kaggle download.")
    target = Path(destination) / "IMDB Dataset.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(target) != sha256_file(files[0]):
            raise FileExistsError(f"Different dataset already exists at {target}.")
    else:
        shutil.copyfile(files[0], target)
    return target


def load_imdb(path, config, seed):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Place the Kaggle IMDB Dataset.csv at {path}, or run `sentivec download`."
        )
    raw = pd.read_csv(path)
    if not {"review", "sentiment"} <= set(raw):
        raise ValueError("CSV must contain review and sentiment columns.")
    mapped = raw["sentiment"].map({"negative": 0, "positive": 1})
    if mapped.isna().any():
        raise ValueError("Unknown/missing sentiment labels in IMDB CSV.")
    frame = canonicalize(pd.DataFrame({"text": raw["review"], "label": mapped}), "imdb")
    corpus, holdout = train_test_split(frame, test_size=0.2, stratify=frame["label"],
                                       random_state=seed)
    validation, test = train_test_split(holdout, test_size=0.5,
                                       stratify=holdout["label"], random_state=seed)
    if len(validation) < config.validation_size or len(test) < config.test_size:
        raise ValueError("Dataset is too small for the configured holdouts.")
    splits = {"corpus": stratified_cap(corpus, config.corpus_size, seed),
              "validation": stratified_cap(validation, config.validation_size, seed),
              "test": stratified_cap(test, config.test_size, seed)}
    assert_disjoint(splits)
    return splits, {"source": KAGGLE_DATASET, "sha256": sha256_file(path),
                    "raw_rows": len(raw), "unique_clean_rows": len(frame),
                    "split_protocol": "stratified 80/10/10 by canonical text; then capped"}


def load_amazon(language, config, seed):
    """MTEB mirror, pinned JSONL files. 0/1 stars -> negative; 3/4 -> positive."""
    from huggingface_hub import hf_hub_download

    splits, hashes, seen = {}, {}, set()
    for source_split, name, limit in [
        ("train", "corpus", config.corpus_size),
        ("validation", "validation", config.validation_size),
        ("test", "test", config.test_size),
    ]:
        path = hf_hub_download("mteb/amazon_reviews_multi", f"{language}/{source_split}.jsonl",
                               repo_type="dataset", revision=AMAZON_REVISION)
        raw = pd.read_json(path, lines=True)
        if not {"text", "label"} <= set(raw) or not raw["label"].isin(range(5)).all():
            raise ValueError("Unexpected Amazon mirror schema or star labels.")
        binary = raw[raw["label"] != 2].copy()
        binary["label"] = (binary["label"] >= 3).astype(int)
        frame = canonicalize(binary[["text", "label"]], f"amazon_{language}")
        # Earlier official partitions take precedence over duplicate later texts.
        frame = frame[~frame["text_hash"].isin(seen)].copy()
        seen.update(frame["text_hash"])
        if len(frame) < limit:
            raise ValueError(f"Amazon {language}/{source_split} has only {len(frame)} usable rows.")
        splits[name] = stratified_cap(frame, limit, seed)
        hashes[source_split] = sha256_file(path)
    assert_disjoint(splits)
    return splits, {"source": "mteb/amazon_reviews_multi", "revision": AMAZON_REVISION,
                    "language": language, "file_sha256": hashes,
                    "label_mapping": "0,1 negative; 2 excluded; 3,4 positive (zero-based stars)"}


def split_calibration_validation(validation, seed):
    calibration, tuning = train_test_split(validation, test_size=0.5,
                                           stratify=validation["label"], random_state=seed)
    result = {"calibration": calibration.reset_index(drop=True),
              "tuning": tuning.reset_index(drop=True)}
    assert_disjoint(result)
    return result
