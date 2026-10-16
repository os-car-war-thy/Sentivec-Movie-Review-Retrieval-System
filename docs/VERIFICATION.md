# Local verification record

Checked on 2026-10-17 using an isolated Windows Python 3.11.17 environment.

- Editable package installation with the `dev` extra succeeded.
- Pinned core versions: NumPy 2.2.6, pandas 2.3.3, SciPy 1.16.2,
  scikit-learn 1.7.2, FAISS CPU 1.12.0.
- `python -m pytest -q`: **25 passed**.
- `ruff check src tests scripts`: passed.
- `python scripts/check_notebook.py`: passed; 29 cells, valid schema and Python syntax,
  no saved execution counts or outputs.
- The full `research,dev` dependency set resolved in a dry-run installation.
- Git initialized on `main`; original supplied files excluded by `.gitignore`.

The tests include real CPU FAISS searches, fixed-k metrics, full-subset proxy references,
calibration, gating/fallback behavior, disjoint/reproducible splits, cache identity,
blinded pool export, incomplete-judgment rejection, and a scenario-level artifact
roundtrip using explicitly synthetic model stand-ins in temporary test directories.

**Not executed:** real pretrained-model inference, Kaggle/Amazon dataset experiments,
the neural smoke/pilot/paper profiles, independent human annotation, or GitHub-hosted CI.
Dependency resolution is not a substitute for running the neural stack. No new research
results or publication-readiness claims are asserted by these checks.
