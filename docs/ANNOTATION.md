# Independent relevance assessment

Automated scores alone are insufficient evidence of human relevance. Every scenario/seed
exports `judgments_template.csv` for a stratified, prespecified subset of test queries.
Pool the top 10 from all dense policies/indexes and BM25, plus five random corpus
documents per query. Deduplicate pairs and randomize presentation. Pool provenance and
model scores are in a separate file: annotators must not see that file.

Two annotators should label each pair independently before an adjudicator resolves
disagreements. Use copies of the template for each annotator. Keep identities pseudonymous.
The final CSV requires one row per pair with these fields:

| Field | Values |
|---|---|
| query_id, doc_id | Preserve template identifiers |
| semantic_grade | 0 unrelated; 1 partially relevant/shared meaningful aspect; 2 directly relevant to the query's content |
| sentiment_match | 1 document satisfies the requested overall polarity; 0 otherwise |
| adjudicator | Nonempty pseudonymous identifier for final adjudication |

The template supplies review text and the requested sentiment constraint (0 negative,
1 positive), derived from the held-out review's dataset label. Grade semantic relevance
separately from tone: shared words or polarity alone do not establish relevance. A review
with conflicting aspects can be topically relevant while failing the requested overall
polarity. Record ambiguities in a separate notes file. Do not change query inclusion
after seeing which system wins. This benchmark uses review-as-query inputs; a separately
collected natural-user-query benchmark is needed to claim realistic search-intent coverage.

Compute agreement on the independent copies before adjudication:

```python
import pandas as pd
from sentivec.annotations import annotator_agreement
print(annotator_agreement(pd.read_csv("annotator_a.csv"), pd.read_csv("annotator_b.csv")))
```

Complete every pool pair. The evaluator rejects missing judgments, duplicates, invalid
grades, or missing adjudicator identifiers. It excludes extra, unregistered pairs from
the ideal ranking. Evaluate with `sentivec evaluate-judgments <experiment-directory>
<final-csv>`. Semantic nDCG uses gains `2**semantic_grade - 1`; joint nDCG replaces the
grade with `semantic_grade * sentiment_match`. The ideal list is pool-relative.

Queries with no positive joint judgments remain in nDCG with score zero; their pooled
AP/recall are undefined. Report their number. The evaluator outputs per-query measurements
so paired intervals can be computed. Pool-relative recall/AP must never be presented as
full-collection recall/AP. Annotation quality, pool bias, query count, agreement, and
adjudication details belong in the paper.
