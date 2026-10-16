# Results already saved in the original notebook

The original `Enhanced_SentiVec_with_Reviews_Addressed.ipynb` already contains executed
experiment outputs. Neither that notebook nor the new research notebook was executed
during repository preparation. The original file remains unchanged locally and excluded
from Git because it contains an ngrok credential. Its numeric output tables are preserved
in `results/original_notebook/`, with a source-file hash and zero-based cell references.

These results belong to the **original implementation**. They have not been independently
reproduced and must not be attributed to the revised pipeline. Numeric values retain the
precision printed by the source notebook. Where two evaluation cells exist, extraction
uses the last comprehensive evaluation, cell 25.

## Retrieval: 500 test queries

| Index | Reported mAP, semantic | Reported mAP, filtered | Relative mAP gain | P@10, semantic | P@10, filtered | Latency ms, semantic | Latency ms, filtered |
|---|---:|---:|---:|---:|---:|---:|---:|
| FlatL2 | 0.229583 | 0.380866 | 65.89% | 0.7158 | 0.9080 | 16.14 | 318.06 |
| IVF | 0.226433 | 0.374558 | 65.42% | 0.7138 | 0.9080 | 11.14 | 310.81 |
| HNSW | 0.231945 | 0.371041 | 59.97% | 0.7184 | 0.9072 | 10.55 | 308.95 |
| IVFPQ | 0.089460 | 0.126787 | 41.72% | 0.4116 | 0.5012 | 9.81 | 303.48 |

Filtering improves agreement with the source benchmark across all four indexes. FlatL2
has the highest filtered effectiveness. HNSW retains much of that effectiveness.
IVFPQ trades effectiveness for compression; its approximately equal full-request latency
shows that a faster index alone does not remove classifier cost. Filtering is roughly
20–31 times slower than the semantic baseline in this saved evaluation. These full-request
measurements need reconciliation with the substantially lower latency values in the
submitted manuscript.

## Classifier and diagnostic results

| Classifier | Accuracy, 500 reviews | F1 as reported | Average inference ms | Mixed-tone examples correct |
|---|---:|---:|---:|---:|
| DistilBERT | 90.2% | 0.8909 | 353.24 | 7/8 |
| VADER | 66.0% | 0.6898 | 3.27 | 5/8 |
| TextBlob | 65.0% | 0.7107 | 1.49 | 3/8 |

DistilBERT has the strongest observed accuracy/F1 and a large latency cost. Mixed-tone
accuracy describes eight handwritten examples only. One useful failure case is
"Amazing visuals can't save the boring story": DistilBERT predicts positive with
approximately 99.8% confidence despite the example's expected negative polarity.

On the 100-review threshold sample, threshold 0.5 retains all reviews with 92% accuracy;
0.7 retains 98% with about 91.84% accuracy; 0.9 retains 91% with about 95.60% accuracy.
The notebook's custom selection score chooses 0.5, while plots mark 0.7 as selected.

## Interpretation and reporting corrections

- Relevance is constructed from the same embedding search plus gold sentiment. Describe
  these scores as proxy-reference agreement; independent human relevance is unmeasured.
- The original retrieval functions do not apply the analyzed confidence threshold to
  decide filtering. The threshold experiment is a classifier study, not a retrieval
  threshold ablation.
- The threshold cell's quantity named F1 combines accuracy and retention; it is not
  standard classification F1. The separate classifier-comparison F1 uses the usual
  sklearn function and is a different quantity.
- Original AP uses the reference-set denominator while returning only 20 results. Keep
  its original definition explicit; do not compare its numbers directly against the new
  implementation's AP@k convention.
- Original P@k divides by returned-list length. Its short-list behavior differs from
  fixed-k precision. The aggregate saved table does not establish whether short lists
  occurred, so the numerical effect cannot be inferred from that table alone.
- RoBERTa is loaded but not evaluated. Language detection is not multilingual evaluation.
  No cross-domain or collection-size experiments are present in the original outputs.
- The saved results are useful historical evidence. Changes to calibration, metadata,
  embedding handling, filtering, or metrics require new runs to produce new numerical
  results. No rerun is needed to preserve and accurately document these existing tables.
