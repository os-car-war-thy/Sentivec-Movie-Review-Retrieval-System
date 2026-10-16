"""Explicit, serializable experimental protocol and immutable Hub revisions."""

from dataclasses import asdict, dataclass

MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2",
               "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    "multilingual_minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                            "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"),
    "distilbert": ("distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                   "714eb0fa89d2f80546fda750413ed43d93601a13"),
    "roberta": ("cardiffnlp/twitter-roberta-base-sentiment-latest",
                "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"),
    "xlm_roberta": ("cardiffnlp/twitter-xlm-roberta-base-sentiment",
                    "f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8"),
}
AMAZON_REVISION = "c379a6705fec24a2493fa68e011692605f44e119"
KAGGLE_DATASET = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews"
PROTOCOL_VERSION = "sentivec-2026-10-v1"


@dataclass(frozen=True)
class Config:
    profile: str = "pilot"
    seeds: tuple = (42,)
    scenarios: tuple = ("imdb_en", "amazon_en", "amazon_de")
    corpus_size: int = 2000
    validation_size: int = 200
    test_size: int = 100
    batch_size: int = 32
    ks: tuple = (5, 10, 20)
    reference_size: int = 50
    candidate_multiplier: int = 10
    thresholds: tuple = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    alphas: tuple = (0.0, 0.1, 0.25, 0.5)
    bootstrap_samples: int = 1000
    timing_queries: int = 20
    timing_repeats: int = 3
    annotation_queries: int = 25
    annotation_depth: int = 10
    device: str = "auto"
    imdb_path: str = "data/raw/IMDB Dataset.csv"
    output_dir: str = "outputs"
    cache_dir: str = "cache"

    def __post_init__(self):
        if self.corpus_size < 40 or self.validation_size < 20 or self.test_size < 10:
            raise ValueError("Insufficient data for stratified splits/calibration.")
        if not self.seeds or not self.scenarios or not set(self.scenarios) <= {
            "imdb_en", "amazon_en", "amazon_de", "amazon_es", "amazon_fr"
        }:
            raise ValueError("Choose at least one seed and a supported scenario.")
        if self.ks != (5, 10, 20) or self.annotation_depth != 10:
            raise ValueError("This registered protocol fixes k=(5,10,20) and annotation depth=10.")

    def to_dict(self):
        return asdict(self)


def profile(name="pilot", **overrides):
    """Smoke validates wiring; pilot explores; paper specifies repeated experiments."""
    presets = {
        "smoke": dict(corpus_size=400, validation_size=40, test_size=20,
                      scenarios=("imdb_en",), timing_queries=5, timing_repeats=2,
                      annotation_queries=5, bootstrap_samples=200),
        "pilot": {},
        "paper": dict(corpus_size=20000, validation_size=1000, test_size=500,
                      seeds=(42, 43, 44), bootstrap_samples=5000,
                      annotation_queries=100, timing_queries=50, timing_repeats=5),
    }
    if name not in presets:
        raise ValueError(f"Unknown profile: {name}")
    return Config(**{**presets[name], **overrides, "profile": name})
