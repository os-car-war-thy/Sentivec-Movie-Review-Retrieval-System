"""Lazy model loading, token-aware long-text handling, and content-addressed caching."""

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import MODELS, PROTOCOL_VERSION
from .metrics import normalize


def synchronize():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def chunk_texts(tokenizer, texts, maximum):
    """Non-overlapping token windows; all windows retained, token-count weighted."""
    window = maximum - tokenizer.num_special_tokens_to_add(pair=False)
    if window < 8:
        raise ValueError("Model context is too small.")
    chunks, owners, weights = [], [], []
    for owner, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not tokens:
            raise ValueError("Empty tokenized document.")
        for start in range(0, len(tokens), window):
            ids = tokens[start:start + window]
            chunks.append(tokenizer.decode(ids, skip_special_tokens=True))
            owners.append(owner)
            weights.append(len(ids))
    return chunks, np.asarray(owners), np.asarray(weights, dtype=float)


def aggregate_chunks(values, owners, weights, count):
    sums = np.zeros((count, values.shape[1]), dtype=np.float64)
    totals = np.zeros(count, dtype=np.float64)
    np.add.at(sums, owners, values * weights[:, None])
    np.add.at(totals, owners, weights)
    if np.any(totals == 0):
        raise ValueError("Document has no chunks.")
    return (sums / totals[:, None]).astype(np.float32)


def cache_array(cache_dir, descriptor, texts, compute):
    descriptor = {**descriptor, "protocol": PROTOCOL_VERSION, "texts": list(texts)}
    digest = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.npy"
    if path.exists():
        result = np.load(path, allow_pickle=False)
        if len(result) != len(texts) or not np.isfinite(result).all():
            raise ValueError(f"Invalid cache: {path}")
        return result
    result = compute(list(texts))
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, result, allow_pickle=False)
    temporary.replace(path)
    return result


def choose_device(requested):
    import torch
    return ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested


class Encoder:
    def __init__(self, key="minilm", device="auto", batch_size=32):
        from sentence_transformers import SentenceTransformer
        self.key, self.batch_size = key, batch_size
        self.repo, self.revision = MODELS[key]
        self.device = choose_device(device)
        self.model = SentenceTransformer(self.repo, revision=self.revision, device=self.device,
                                         trust_remote_code=False)
        self.maximum = int(self.model.max_seq_length)

    def encode(self, texts):
        chunks, owners, weights = chunk_texts(self.model.tokenizer, list(texts), self.maximum)
        vectors = self.model.encode(chunks, batch_size=self.batch_size, convert_to_numpy=True,
                                     normalize_embeddings=False, show_progress_bar=False)
        return normalize(aggregate_chunks(vectors, owners, weights, len(texts)))

    def cached(self, texts, cache_dir):
        return cache_array(cache_dir, {"kind": "embeddings", "repo": self.repo,
                                       "revision": self.revision, "maximum": self.maximum},
                           texts, self.encode)


class SentimentModel:
    def __init__(self, key="distilbert", device="auto", batch_size=32):
        self.key, self.batch_size = key, batch_size
        self.device = choose_device(device)
        if key == "vader":
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.model = SentimentIntensityAnalyzer()
            self.repo, self.revision = "vaderSentiment", "3.3.2"
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.repo, self.revision = MODELS[key]
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo, revision=self.revision,
                                                      trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.repo, revision=self.revision, trust_remote_code=False
        ).to(self.device).eval()
        labels = {int(i): str(label).lower() for i, label in self.model.config.id2label.items()}
        expected = {0: "negative", 1: "positive"} if key == "distilbert" else {
            0: "negative", 1: "neutral", 2: "positive"
        }
        if labels != expected:
            raise ValueError(f"Unexpected label mapping for pinned {key}: {labels}")

    def scores(self, texts):
        if self.key == "vader":
            p = np.clip([(self.model.polarity_scores(text)["compound"] + 1) / 2
                         for text in texts], 1e-6, 1 - 1e-6)
            return np.column_stack((np.log(1 - p), np.full(len(p), -100.0), np.log(p)))
        import torch
        maximum = min(int(self.tokenizer.model_max_length), 512)
        chunks, owners, weights = chunk_texts(self.tokenizer, list(texts), maximum)
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(chunks), self.batch_size):
                inputs = self.tokenizer(chunks[start:start + self.batch_size], padding=True,
                                         truncation=True, max_length=maximum, return_tensors="pt")
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                outputs.append(self.model(**inputs).logits.float().cpu().numpy())
        logits = aggregate_chunks(np.concatenate(outputs), owners, weights, len(texts))
        if self.key == "distilbert":
            logits = np.column_stack((logits[:, 0], np.full(len(logits), -100.0), logits[:, 1]))
        return logits

    def cached(self, texts, cache_dir):
        return cache_array(cache_dir, {"kind": "sentiment_logits", "repo": self.repo,
                                       "revision": self.revision, "pooling": "token_weighted_logits"},
                           texts, self.scores)
