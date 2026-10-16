# Reviewer-response map

These are implementation provisions, not assertions that experiments have been completed.

| Review concern | Repository provision | Remaining evidence/action |
|---|---|---|
| Table IV in a single column | Separate compact effectiveness/latency CSV and LaTeX tables | Typeset and verify in the actual conference template |
| Comparison with cited work | Notebook qualitative comparison; REFERENCES.md | Verify bibliography; only compare numerical scores on shared protocols |
| Research-oriented conclusion | Protocol questions, measured tables, limitations checklist | Rewrite conclusion after final results; avoid unsupported deployment claims |
| Relevant references | Primary dataset/model/library references | Check venue style and add directly relevant retrieval research |
| Single English source | IMDB + English/German Amazon product reviews | Execute scenarios; report domain/language confounds and target validation use |
| Threshold selection | Disjoint calibration/tuning, actual gated policy and fallback | Execute sensitivity study; report accuracy/coverage and retrieval consequences |
| Other classifiers | DistilBERT/RoBERTa/VADER, multilingual XLM-R | Execute comparisons; no cross-language fairness claim for English baselines |
| Errors and mixed tone | High-confidence errors, length/class slices, fixed probes, format perturbations | Discuss observed failure causes; handwritten examples remain diagnostic |
| Robustness/statistics | Per-query scores, paired bootstrap, three-seed paper profile | Execute repeated runs; do not infer significance from exploratory comparisons alone |
| Scalability | Real nested subsets, serialized sizes, online stage timing | Execute full profile; concurrency/production testing remains outside scope |
| Retrieval validity | Explicit proxy naming; blinded annotation workflow | Obtain independent annotations and adjudication before human-relevance claims |
