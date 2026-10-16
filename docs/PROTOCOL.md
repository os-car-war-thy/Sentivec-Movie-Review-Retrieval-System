# Experimental protocol

Protocol ID: `sentivec-2026-10-v1`. Register any changes before examining new test results.

## Questions and scope

1. How do hard, gated, and soft sentiment policies affect sentiment alignment and
   agreement with a sentiment-constrained embedding reference?
2. Do independent relevance judgments support those differences?
3. How do calibration, uncertain/mixed-tone inputs, classifier errors, approximate
   indexes, candidate-pool sizes, and corpus sizes affect the trade-off?

The primary automated comparison is IMDB, seed 42, FlatIP, gated filtering minus semantic
retrieval on proxy nDCG@10. Other indexes, seeds, domains, and diagnostic comparisons are
exploratory. Report intervals without treating dozens of comparisons as independent
confirmatory discoveries. No publication-quality result is supplied in advance.

## Partitions and provenance

Clean HTML/whitespace, exclude empty text, identify text with SHA-256, reject conflicting
labels, and deduplicate before partitioning. Kaggle's CSV lacks the original official
train/test identifiers: use a declared stratified 80/10/10 corpus/validation/test split.
Cap partitions reproducibly while maintaining class proportions. Validation is split
50/50 into calibration and tuning. Test queries are never indexed. Text hashes are
disjoint; movie/reviewer-level separation cannot be verified from this CSV's two columns.

Amazon uses the mirror's official train/validation/test files, pinned to a commit and
hashed locally. Duplicate texts in later partitions are removed before sampling. Drop
zero-based star label 2; labels 0/1 become negative, 3/4 positive. Each secondary dataset
uses its own validation labels. Do not describe these experiments as zero-shot transfer.
English IMDB/product scenarios use the same English encoder/classifiers. Non-English
product scenarios use the multilingual encoder/XLM-R classifier. Report results by
scenario: domain and encoder differences must not be attributed to a single cause.

## Long text and model probabilities

Tokenize the whole text into non-overlapping windows that fit each model, retaining every
window. Decode windows for model batching, apply a final tokenizer context guard, and
aggregate vectors/logits with token-count weights. Normalize final embeddings to float32
unit vectors. This is a declared long-text policy, not sentence-level aspect modelling.
Decoding and retokenizing can change boundaries; the final guard may truncate a few
boundary tokens. Report that limitation rather than claiming exact token preservation.

Classifier logits use `[negative, neutral, positive]` order. For binary models the neutral
logit is fixed to -100. Temperature scaling minimizes binary polarity NLL on calibration
data, conditioned on negative/positive. Gating confidence is
`max(P(pos | nonneutral), P(neg | nonneutral)) * (1 - P(neutral))`. This is a validation-
selected decision score, not a calibrated probability of neutrality or correctness.
The binary classifier metrics use conditional polarity probabilities. Neutral/mixed
queries are diagnostic because the primary datasets exclude a neutral class.

## Policies and parameter freezing

Primary classifiers are prespecified: DistilBERT for English, XLM-R for German. RoBERTa
and VADER are comparisons, not test-selected replacements. Predict document metadata
offline with the primary model. Search `min(N, 10 * max(k))` candidates for every dense
method, recompute their cosine scores from original vectors, and:

- semantic: rank by cosine;
- hard filter: retain predicted document polarity matching predicted query polarity;
- gated filter: apply hard filtering only above the validation-selected confidence
  threshold; otherwise retain semantic candidates;
- soft rerank: cosine plus `alpha * (P(same polarity) - 0.5)`;
- oracle metadata: replace document predictions with dataset labels, keeping the query
  prediction. This is a document-metadata upper-bound/control experiment.

Tune threshold from `[.5,.6,.7,.8,.9,.95]` and alpha from `[0,.1,.25,.5]` on tuning
proxy nDCG@10 with FlatIP. Ties prefer higher thresholds/smaller alpha. Write frozen
parameters before any test inference. Candidate/index sensitivity sweeps use validation
only and restore the predeclared final parameters. Short result lists are recorded;
they are not backfilled silently or omitted.

## Indexes and measurements

All indexes use normalized vectors and inner product. IVF cluster count is
`min(100, max(1, N // 40))`, nprobe `min(10,nlist)`. HNSW uses M=32,
efConstruction=80, efSearch=64. IVFPQ uses m=8 and 8 bits and requires at least 9,984
unique training vectors. Set one FAISS thread and IVF/PQ seeds. Retain original vectors
for rescoring; serialized index bytes exclude this extra storage and the models.

ANN recall compares semantic results against exact unconstrained neighbors. Report this
separately from sentiment-aware relevance. Scalability uses nested, distinct corpus
subsets up to the real available size. Do not duplicate vectors to claim larger datasets.

Online timing warms models/indexes, randomizes request order, performs uncached
single-query inference, synchronizes CUDA boundaries, and separates sentiment inference,
encoding, and search/policy time. Report mean/p50/p95 and request counts. It excludes
startup, downloading, transport, UI rendering, and corpus preprocessing. It does not
measure concurrent throughput, peak memory, or production deployment performance.

## Metrics and inference limits

Automated reference: exact top-50 cosine neighbors *within* the gold sentiment subset,
searched over the whole corpus. This reference depends on the evaluated embedding and
sentiment objective. Prefix all resulting metrics with `proxy_`.

Precision@k divides by k, including absent ranks. Recall@k divides by reference size.
AP@k divides summed precision at relevant ranks by `min(k, |R|)`. nDCG uses binary
gains for the proxy. RR@k is zero if no relevant item occurs by k. Report mean AP@k/MRR@k
with the cutoff; these are not untruncated mAP/MRR. Sentiment P@k compares corpus gold
labels with query gold polarity and also uses fixed k.

Bootstrap query-level means/differences with paired query IDs, separately per seed and
scenario. Repeated latency observations are not independent relevance observations.
Summarize repeated seeds descriptively; do not pool them as independent query samples.
Human nDCG uses adjudicated semantic and joint semantic/tone grades over a completely
judged pool. Human recall/AP are explicitly pool-relative. See ANNOTATION.md.

## Deployment and limitations

Offline sentiment/embedding preprocessing must be repeated or incrementally updated as
documents change. Model/version/metadata alignment, language routing, neutral requests,
explicit user sentiment intent, privacy, concurrent load, and cold starts require further
evaluation. Matching a query's expressed sentiment is not always matching the sentiment
the user requests. FAISS provides indexing/search, not a full database or deployment
service. The notebook does not host an app or open a public tunnel.
