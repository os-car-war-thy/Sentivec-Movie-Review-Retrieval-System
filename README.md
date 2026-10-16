# SentiVec

Reproducible research on **sentiment-aware review retrieval**: sentence embeddings,
FAISS search, calibrated sentiment classification, and confidence-based retrieval policies.

**Status:** a new research implementation. The original submission's results have not
been copied into this codebase. Full neural experiments and independent human evaluation
must be completed before reporting new findings. The notebook is intentionally saved
without outputs.

The original notebook **was already run**. Its saved numerical outputs are preserved in
[results/original_notebook](results/original_notebook/), with an
[assessment of those existing results](docs/EXISTING_RESULTS.md). They belong to the
original implementation; the new implementation has not been run on the research data.

## Start here

Use **Python 3.11 or 3.12**. A GPU is recommended for the full study; CPU execution is
supported and slower. Model and secondary-dataset downloads require an internet connection.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name sentivec --display-name "Python (SentiVec)"
```

Install Python 3.11 first if the Windows launcher cannot find it. On Linux/macOS,
create the environment with `python3.11 -m venv .venv` and activate with
`source .venv/bin/activate`. Open
[the research notebook](notebooks/SentiVec_Reproducible_Study.ipynb) in VS Code or Jupyter
and select the SentiVec kernel. Run cells in order. The notebook uses this repository's
`src/sentivec` package, so retain the repository layout.

## Dataset

Primary dataset: [IMDB Dataset of 50K Movie Reviews, by lakshmi25npathi on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

Download and extract **IMDB Dataset.csv** into `data/raw/`, or use:

```powershell
sentivec download
```

Kaggle may request authentication or acceptance of its dataset terms. Use Kaggle's own
authentication flow; keep credentials outside the repository. The loader expects `review`
and `sentiment` columns and records the CSV's SHA-256 checksum.

Cross-domain/multilingual experiments use pinned JSONL snapshots from
[MTEB's Amazon Reviews Multi mirror](https://huggingface.co/datasets/mteb/amazon_reviews_multi).
English and German scenarios run by default. Three-star reviews are excluded; one/two
stars map to negative and four/five to positive. Dataset and model rights are separate
from this repository's MIT code license. Raw data and weights are not redistributed.

## Run experiments

Run these commands from the repository root after installation:

```powershell
# Small IMDB integration run; still downloads actual pretrained models.
sentivec run --profile smoke

# Exploratory IMDB + English/German product-review experiments.
sentivec run --profile pilot

# Three seeds, up to 20,000 corpus reviews and 500 test queries per scenario.
sentivec run --profile paper

# Restrict a run or supply an existing CSV.
sentivec run --profile pilot --scenarios imdb_en --imdb-path "D:\datasets\IMDB Dataset.csv"
```

| Profile | Corpus/scenario | Validation queries | Test queries | Seeds | Intended use |
|---|---:|---:|---:|---|---|
| smoke | 400 | 40 | 20 | 42 | Check neural pipeline wiring |
| pilot | 2,000 | 200 | 100 | 42 | Explore protocol and costs |
| paper | 20,000 | 1,000 | 500 | 42, 43, 44 | Prespecified full experiments |

Validation is divided equally into calibration and policy-tuning partitions. A profile
name does not establish publication readiness. Full runs can take substantial time and
disk space, particularly on CPU. IVFPQ is explicitly skipped below 9,984 training vectors;
its skip is recorded rather than replaced with invented results.

## What the study measures

- **Classifiers:** calibrated/uncalibrated DistilBERT, RoBERTa, and VADER on English;
  XLM-RoBERTa on German. The primary classifier is fixed in advance for each language.
- **Policies:** semantic retrieval, hard filtering, confidence-gated filtering with
  semantic fallback, soft sentiment reranking, and a separately labelled gold-document-
  metadata experiment. Main document metadata is predicted, not supplied by test labels.
- **Indexes:** Flat inner product, IVF, HNSW, and IVFPQ where training size permits.
  Embeddings are normalized and candidates rescored using original vectors.
- **Controls:** BM25, validation-only threshold/candidate/index sensitivity sweeps,
  mixed-tone diagnostic probes, formatting perturbations, and error slices.
- **Reporting:** fixed-cutoff IR metrics, paired query bootstrap intervals, single-request
  latency with separate stages, real corpus-size scaling, and CSV/LaTeX tables.

**Automated relevance is a proxy:** exact nearest neighbors constrained by gold query
and document sentiment. It measures agreement with that reference and favors its
construction. It cannot by itself establish improved human relevance. Use the blinded
annotation workflow for independent evidence. Secondary domains use their own validation
labels; these are domain-adapted evaluations, not zero-shot transfer. Multilingual
experiments retrieve within each language, not across languages.

See [the protocol](docs/PROTOCOL.md), [annotation instructions](docs/ANNOTATION.md),
[reviewer-response map](docs/REVIEWER_MAP.md), and
[publication checklist](docs/PUBLICATION_CHECKLIST.md).

## Reproducibility and artifacts

Each run creates a new directory under `outputs/` containing configuration, exact model
revisions, dataset/split hashes, hardware/package metadata, a dependency freeze, frozen
validation-selected parameters, per-query metrics, ranked outputs, latency observations,
tables, and artifact checksums. Runs fail visibly on errors; failed queries are not
silently omitted. Caches are keyed by input text, model revision, and protocol version.
Cached computations are excluded from online timing. Floating-point results and timings
can vary across hardware; preserve the environment manifest alongside reported results.

Direct dependencies are pinned in `pyproject.toml`; transitives are resolved at install
time and recorded in each run's `environment.freeze.txt`. For exact replication, preserve
that freeze and use the same OS/Python/hardware and repository commit. An editable-install
path in a freeze refers to the original checkout and must be replaced with the new local
checkout when rebuilding the environment.

Completed human labels can be evaluated with:

```powershell
sentivec evaluate-judgments outputs/<run>/imdb_en/seed-42 data/annotations/judgments.csv
```

## Verify the code

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check src tests scripts
python scripts/check_notebook.py
```

Tests use synthetic inputs and real CPU FAISS. They require no private data, pretrained
models, Kaggle credentials, or GPU. GitHub Actions runs the same checks on Python 3.11/3.12.
These checks validate implementation behavior; they do not establish research findings.
See [the local verification record](docs/VERIFICATION.md) for what was actually checked.

## Push this folder to GitHub

Create an **empty repository** on GitHub, without an initial README or license. Then:

```powershell
git status --short
git add .
git diff --cached --stat
git commit -m "Add reproducible SentiVec research project"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace both uppercase placeholders. If Git requests your identity, configure your own
name/email before committing. Review staged files before pushing. Original submission
materials are preserved locally and ignored by Git. The old notebook contains a private
ngrok token: revoke/rotate it before sharing that original file elsewhere. New notebooks
and source contain no tunnel credentials or automatic hosting steps.

Before committing an executed notebook, clear outputs:

```powershell
python scripts/clear_notebook_outputs.py notebooks/SentiVec_Reproducible_Study.ipynb
```

## Layout

```text
notebooks/             Guided research notebook
src/sentivec/          Data, models, calibration, retrieval, evaluation, and CLI
tests/                 Synthetic regression and integration tests
scripts/               Notebook generation, validation, and output clearing
docs/                  Protocol, annotation, reviewer map, and release checklist
data/README.md         Dataset placement instructions
.github/workflows/     Credential-free continuous integration
```

## Citation and license

New code/documentation use the [MIT license](LICENSE). See [CITATION.cff](CITATION.cff)
and [source references](docs/REFERENCES.md). Add the actual author names and repository
URL to the citation metadata before making a release. The original manuscript is not
represented as a published paper or bundled under the code license.
