"""Generate the output-free research notebook from reviewable plain Python cells."""

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": textwrap.dedent(source).strip() + "\n"})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": textwrap.dedent(source).strip() + "\n"})


md("""
# SentiVec: reproducible sentiment-aware retrieval study

**Research implementation; results must be generated.** This notebook replaces the
experimental workflow, not the manuscript. It imports the reusable `sentivec` package
in this repository. Run in order with Python 3.11/3.12. Start with `smoke` or `pilot`;
use `paper` for the prespecified repeated study after inspecting the protocol.

Primary data: [Kaggle IMDB 50K](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
The project does not redistribute reviews or weights. The default pilot includes IMDB
and English/German Amazon reviews. Full runs download models and secondary data and
can take substantial time on CPU. No credentials or public tunnels are embedded.

See `docs/PROTOCOL.md`, `docs/ANNOTATION.md`, and `docs/PUBLICATION_CHECKLIST.md`.
""")
md("""
## 1. Research questions and evidence

1. Do sentiment policies improve sentiment alignment and agreement with a declared
   embedding/sentiment reference?
2. Do independent, blinded human judgments support improved semantic and joint relevance?
3. What are the effects of uncertainty, classifier errors, domain/language shifts,
   candidate budgets, ANN approximation, and collection size?

The automated reference is **a proxy**, derived from the same embedding family and a
sentiment constraint. Report proxy metrics as such. Human relevance requires completed
independent annotations; no automatically produced score substitutes for them.

Primary automated contrast: IMDB, seed 42, FlatIP gated filtering minus semantic retrieval
on proxy nDCG@10. Other comparisons are exploratory. Report paired query intervals
separately per seed. The paper profile uses seeds 42/43/44; do not pool repeated seeds as
independent query samples. Secondary scenarios use target-domain validation labels and
within-language retrieval, so they are neither zero-shot nor cross-language evaluations.
""")
code("""
import os
import sys
from pathlib import Path

# Set before CUDA initialization.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "pyproject.toml").exists() and (p / "src" / "sentivec").exists()), None)
if ROOT is None:
    raise RuntimeError("Open this notebook inside the cloned SentiVec repository.")
os.chdir(ROOT)
if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise RuntimeError("Use the supported Python 3.11 or 3.12 environment.")
print("Repository:", ROOT)
print("Python:", sys.version)
""")
md("""
## 2. Environment

Install from the repository root with `python -m pip install -r requirements.txt`, then
select that environment as the kernel. Alternatively set the switch below to `True`
once; restart the kernel after installing packages. Exact direct pins are in
`pyproject.toml`; each experiment also writes a transitive dependency freeze.
""")
code("""
INSTALL_DEPENDENCIES = False
if INSTALL_DEPENDENCIES:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    raise RuntimeError("Dependencies installed. Restart the kernel, then run from the top.")

import importlib.metadata
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sentivec.config import MODELS, PROTOCOL_VERSION, profile
from sentivec.data import download_imdb, load_imdb
from sentivec.experiment import run_study
from sentivec.metrics import retrieval_metrics, paired_difference

print("Protocol:", PROTOCOL_VERSION)
display(pd.DataFrame([
    {"package": p, "version": importlib.metadata.version(p)} for p in
    ["numpy", "pandas", "torch", "transformers", "sentence-transformers", "faiss-cpu"]
]))
""")
md("""
## 3. Configuration and data

Change this cell **before** running experiments. `smoke` uses one small IMDB scenario;
`pilot` is exploratory. `paper` uses 20,000 corpus reviews, 1,000 validation reviews,
500 test queries and three seeds per scenario. Validation is split equally into separate
calibration and policy-tuning sets. The Kaggle file has no official split IDs: this
protocol uses declared, stratified 80/10/10 partitions after cleaning/deduplication.
""")
code("""
PROFILE = "pilot"  # "smoke", "pilot", or "paper"
IMDB_PATH = ROOT / "data" / "raw" / "IMDB Dataset.csv"
DOWNLOAD_IMDB_IF_MISSING = True  # Explicit Kaggle download; may require Kaggle authentication.
DEVICE = "auto"  # "cpu", "cuda", or "auto"

CONFIG = profile(PROFILE, imdb_path=str(IMDB_PATH), device=DEVICE,
                 output_dir=str(ROOT / "outputs"), cache_dir=str(ROOT / "cache"))
# To restrict to the primary dataset, add scenarios=("imdb_en",) to profile(...).
display(pd.Series(CONFIG.to_dict(), name="configuration"))
display(pd.DataFrame([{"component": key, "repository": value[0], "revision": value[1]}
                      for key, value in MODELS.items()]))

if "imdb_en" in CONFIG.scenarios and not IMDB_PATH.exists():
    if not DOWNLOAD_IMDB_IF_MISSING:
        raise FileNotFoundError("Download and extract the Kaggle CSV to data/raw/ first.")
    download_imdb(IMDB_PATH.parent)
""")
code("""
# Preflight records and checks the primary source before any models are loaded.
if "imdb_en" in CONFIG.scenarios:
    partitions, provenance = load_imdb(IMDB_PATH, CONFIG, CONFIG.seeds[0])
    display(pd.Series(provenance, name="IMDB provenance"))
    display(pd.DataFrame([
        {"split": name, "reviews": len(frame), "negative": int((frame.label == 0).sum()),
         "positive": int((frame.label == 1).sum())} for name, frame in partitions.items()
    ]))
    del partitions
""")
md("""
## 4. Metric definitions and implementation checks

Precision@k divides by **k**, including absent ranks. Recall@k divides by reference
size. AP@k divides by `min(k, |R|)`. nDCG uses binary proxy relevance, and RR@k is
truncated at k. Mean AP@20 and mean RR@20 should be named with their cutoff.

The proxy consists of the exact top 50 neighbors within the gold sentiment subset,
searching the entire corpus. Main retrieval uses predicted document metadata. A separately
labelled `oracle_metadata` control uses dataset document labels and predicted query
sentiment. Gold labels never enter the main candidate-filtering policy.
""")
code("""
short_list = retrieval_metrics(["a"], ["a", "b", "c", "d", "e"], ks=(5,))
assert short_list["P@5"] == 0.2
assert short_list["fill@5"] == 0.2
assert retrieval_metrics(["a", "b"], ["a", "b"], ks=(2,))["AP@2"] == 1.0
assert paired_difference([1, 1], [0, 0], samples=100)["mean"] == 1.0
print("Fixed-cutoff metric checks passed. Run pytest for the full synthetic suite.")
""")
md("""
## 5. Execute the registered protocol

The runner chunks long documents using each model's tokenizer, calibrates logits on
calibration reviews, chooses threshold/soft weight on tuning reviews, and writes frozen
parameters before test inference. It does not silently skip failed queries. Temperatures
calibrate binary polarity; gating confidence additionally discounts neutral probability.

All dense policies search the same candidate budget. Retrieved candidates are rescored
from normalized original vectors, including IVFPQ candidates. BM25 supplies a lexical
control. Bounded filtering can return fewer than k results; report fill rates.

Each run gets a fresh output directory. Content-addressed model caches can be reused;
online timing performs fresh inference. A failed run retains its failure status and
partial artifacts. IVFPQ is recorded as skipped when there are insufficient training
vectors; smoke/pilot scores do not represent all four indexes.
""")
code("""
RUN_DIR = run_study(CONFIG)
print("Artifacts:", RUN_DIR)
print((RUN_DIR / "status.json").read_text())
""")
md("""
## 6. Inspect the main scenario and frozen choices

These displays read generated artifacts; they contain no prewritten experimental scores.
Choose a different scenario/seed below to inspect it. Every scenario/seed has separate
source hashes, split IDs, parameters, predictions, and per-query measurements.
""")
code("""
SCENARIO = CONFIG.scenarios[0]
SEED = CONFIG.seeds[0]
EXP = RUN_DIR / SCENARIO / f"seed-{SEED}"
read_table = lambda name: pd.read_csv(EXP / name)
display(pd.Series(json.loads((EXP / "frozen_parameters.json").read_text())))
display(pd.Series(json.loads((EXP / "index_manifest.json").read_text())))
display(read_table("classifier_metrics.csv"))
""")
md("""
## 7. Validation threshold and reranking sensitivity

Threshold selection maximizes tuning proxy nDCG@10; ties favor a higher threshold.
Below threshold, gated filtering falls back to semantic retrieval. Soft reranking uses
the smallest alpha in a tie. Confidence coverage and selective accuracy are separate
quantities; their harmonic mean is not reported as classification F1.
""")
code("""
validation = read_table("threshold_alpha_validation.csv")
display(validation)
gated = validation[validation["mode"] == "gated_filter"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(gated.value, gated["proxy_nDCG@10"], marker="o")
axes[0].set(xlabel="Gate threshold", ylabel="Validation proxy nDCG@10")
axes[1].plot(gated.value, gated.confidence_coverage, marker="o", label="coverage")
axes[1].plot(gated.value, gated.selective_accuracy, marker="s", label="selective accuracy")
axes[1].set(xlabel="Gate threshold", ylabel="Proportion", ylim=(0, 1.05))
axes[1].legend()
fig.tight_layout()
fig.savefig(EXP / "validation_thresholds.png", dpi=200)
plt.show()
display(read_table("index_candidate_validation.csv"))
""")
md("""
## 8. Test effectiveness, ablation, and uncertainty

Keep proxy relevance and sentiment alignment separate. The oracle metadata control
isolates document-label noise. Recomputed cosine scores are descriptive vector scores,
not independent human relevance. Bootstrap intervals below resample paired queries
within this scenario/seed, not repeated requests or pooled seeds.
""")
code("""
display(read_table("table_effectiveness.csv"))
summary = read_table("metrics_summary.csv")
display(summary[summary.metric.isin(["proxy_nDCG@10", "proxy_AP@20", "proxy_RR@20",
                                    "sentiment_P@10", "proxy_fill@10", "ANN_recall@20"])])
display(read_table("paired_differences.csv"))

view = summary[(summary.metric == "proxy_nDCG@10") & (summary["index"] == "flat")]
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(view["mode"], view["mean"], yerr=np.vstack([
    view["mean"] - view.ci_low, view.ci_high - view["mean"]]), capsize=3)
ax.set(ylabel="Test proxy nDCG@10", title=f"{SCENARIO}, seed {SEED}: query bootstrap intervals")
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
fig.savefig(EXP / "proxy_policy_comparison.png", dpi=200)
plt.show()
""")
md("""
## 9. Latency and real collection-size scaling

Online timing includes fresh sentiment inference (for gated retrieval), fresh embedding,
search and policy application. It excludes startup, network transport, rendering, and
offline corpus processing. Search-stage scaling uses precomputed query vectors and
must not be described as full request latency. Serialized index bytes exclude retained
original vectors, model weights, Python objects, and peak process/GPU memory.
""")
code("""
display(read_table("table_latency.csv"))
timings = read_table("online_timing_requests.csv")
display(timings.groupby(["index", "mode"])[
    ["classification_ms", "encoding_ms", "search_and_policy_ms", "total_ms"]].mean())
scale = read_table("scalability.csv")
display(scale)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for name, group in scale[scale.status == "complete"].groupby("index"):
    axes[0].plot(group.n_documents, group.search_policy_p50_ms, marker="o", label=name)
    axes[1].plot(group.n_documents, group.serialized_index_bytes / 1024**2, marker="o", label=name)
axes[0].set(xlabel="Unique corpus documents", ylabel="Search/policy p50 (ms)")
axes[1].set(xlabel="Unique corpus documents", ylabel="Serialized index (MiB)")
axes[0].legend()
fig.tight_layout()
fig.savefig(EXP / "collection_scaling.png", dpi=200)
plt.show()
""")
md("""
## 10. Error analysis and mixed/neutral inputs

Inspect actual errors, including confident errors, by class and length. Handwritten
contrast/negation/sarcasm probes are diagnostics with subjective expected polarities;
they are not a representative held-out benchmark. Neutral/ambiguous probes have no
binary expected label. Uppercase and line-break consistency are stress tests, not
additional independent test queries. A neutral/mixed-aware encoder or classifier is
not claimed merely because a binary policy falls back below a confidence threshold.
""")
code("""
display(read_table("error_slices.csv"))
display(read_table("high_confidence_errors.csv").head(10))
mixed_path = EXP / "mixed_tone_diagnostics.csv"
if mixed_path.exists():
    display(pd.read_csv(mixed_path))
perturbations = read_table("format_robustness.csv")
display(perturbations.groupby("variant")[["correct", "consistent_with_original"]].mean())
""")
md("""
## 11. Independent human annotation (required for human-relevance claims)

Give `judgments_template.csv` to independent annotators without the pool-provenance
file. The template pools top-10 candidates from all systems plus random controls and
separates semantic relevance from requested sentiment. Follow `docs/ANNOTATION.md`:
two independent annotations, agreement analysis, then final adjudication.

The evaluator requires all registered pairs to be judged. Human nDCG is normalized
against the judged pool. AP/recall are explicitly pool-relative. Queries without positive
joint judgments remain in nDCG and have undefined pooled recall/AP. Do not replace
missing judgments with zeros or remove difficult queries after seeing results.
""")
code("""
from sentivec.annotations import evaluate_judgments

JUDGMENTS_PATH = None  # e.g. ROOT / "data" / "annotations" / "judgments.csv"
print("Blinded template:", EXP / "judgments_template.csv")
if JUDGMENTS_PATH is None:
    print("Independent relevance evaluation pending. Automated proxy scores are insufficient.")
else:
    human = evaluate_judgments(EXP, JUDGMENTS_PATH)
    display(human.groupby(["index", "mode"]).mean(numeric_only=True))
    paired = human[human["index"] == "flat"].pivot(
        index="query_id", columns="mode", values="human_joint_nDCG@10")
    display(pd.Series(paired_difference(paired.gated_filter, paired.semantic,
                                       CONFIG.bootstrap_samples, SEED),
                      name="Human joint nDCG@10: gated minus semantic"))
""")
md("""
## 12. Cross-domain/language and repeated-seed reporting

Report each scenario separately. English IMDB and English products share model families;
German products use multilingual models, so language and model changes are confounded.
Target-domain calibration/tuning is allowed and recorded. Repeated-seed means/standard
deviations are descriptive, and are not confidence intervals over independent studies.
""")
code("""
all_summaries = []
for path in sorted(RUN_DIR.glob("*/seed-*/metrics_summary.csv")):
    frame = pd.read_csv(path)
    frame["scenario"] = path.parent.parent.name
    frame["seed"] = int(path.parent.name.split("-")[1])
    all_summaries.append(frame)
all_summaries = pd.concat(all_summaries, ignore_index=True)
display(all_summaries[(all_summaries.metric == "proxy_nDCG@10") &
                      (all_summaries["index"] == "flat")])
seed_summary = all_summaries.groupby(["scenario", "index", "mode", "metric"])["mean"].agg(
    seed_mean="mean", seed_std="std", seeds="count")
seed_summary.to_csv(RUN_DIR / "descriptive_seed_summary.csv")
display(seed_summary.loc[(slice(None), "flat", slice(None), "proxy_nDCG@10"), :])
""")
md("""
## 13. Related-work comparison and manuscript outputs

Use a qualitative comparison where evaluation protocols differ; do not copy unrelated
classification scores into a retrieval leaderboard. Verify complete bibliography entries
and add directly related sentiment-aware/filtered retrieval studies before submission.
""")
code("""
related = pd.DataFrame([
    {"work": "Sentence-BERT (2019)", "role": "sentence embeddings",
     "comparison": "representation foundation; no sentiment-filtered retrieval claim",
     "source": "https://aclanthology.org/D19-1410/"},
    {"work": "FAISS", "role": "exact/approximate vector search",
     "comparison": "index foundation; ANN recall is distinct from sentiment relevance",
     "source": "https://github.com/facebookresearch/faiss"},
    {"work": "TimeLMs / CardiffNLP RoBERTa", "role": "English sentiment classification",
     "comparison": "classifier baseline with Twitter-to-review domain mismatch",
     "source": "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest"},
    {"work": "XLM-T", "role": "multilingual sentiment classification",
     "comparison": "classifier component; no claim of equivalent retrieval evaluation",
     "source": "https://github.com/cardiffnlp/xlm-t"},
    {"work": "SentiVec (this implementation)", "role": "sentiment-aware retrieval protocol",
     "comparison": "predicted metadata, calibrated gating, proxy and independent evaluation",
     "source": "See this repository's exact commit and run manifests"},
])
display(related)
related.to_csv(RUN_DIR / "qualitative_related_work.csv", index=False)
print("Paper tables:", EXP / "table_effectiveness.tex", EXP / "table_latency.tex")
""")
md("""
## 14. Conclusions, limitations, and release

Write the conclusion only after the runs and annotation are complete. Quantify the
primary paired comparison, explain confident classifier failures and filtering shortfall,
and distinguish proxy agreement from independent relevance. State the measured corpus
size and latency boundary. Discuss offline metadata cost, actual index+vector storage,
neutral/aspect limitations, review-as-query inputs, and target-domain validation.

No concurrent-load, cross-language retrieval, fine-tuning, sentiment-concatenation,
cross-validation, or deployed database experiment is implemented here. Future work can
study explicit user sentiment constraints, aspect-sensitive ranking, natural search
queries, model drift, incremental indexing, and concurrent deployment.

Before publication: complete `docs/PUBLICATION_CHECKLIST.md`, regenerate manuscript
tables from identified runs, verify references, and adjudicate human labels. Before a
GitHub commit, clear notebook outputs with `scripts/clear_notebook_outputs.py` so raw
reviews and environment details are not accidentally included. Generated artifacts are
ignored by Git. The original private submission/review files remain excluded.
""")


def main():
    for i, cell in enumerate(cells):
        cell["id"] = f"sentivec-{i:03d}"
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python (SentiVec)", "language": "python", "name": "sentivec"},
        "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}
    target = ROOT / "notebooks" / "SentiVec_Reproducible_Study.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created {target.relative_to(ROOT)} ({len(cells)} cells).")


if __name__ == "__main__":
    main()
