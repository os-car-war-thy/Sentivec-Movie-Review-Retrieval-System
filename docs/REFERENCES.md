# Primary sources and implementation references

Use the exact source/model revision recorded with each run. The following are qualitative
method references, not evidence that SentiVec outperforms those works.

- [Kaggle: IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
  — the supplied primary data source.
- [Maas et al., Learning Word Vectors for Sentiment Analysis, ACL 2011](https://aclanthology.org/P11-1015/)
  — original Large Movie Review Dataset paper; this project uses the Kaggle CSV split protocol.
- [Reimers and Gurevych, Sentence-BERT, EMNLP 2019](https://aclanthology.org/D19-1410/)
  — sentence embedding framework.
- [all-MiniLM-L6-v2 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  and [multilingual MiniLM model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
  — exact model usage and limitations.
- [DistilBERT SST-2 model card](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
  — English sentiment classifier.
- [CardiffNLP TimeLMs model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
  — English three-class classifier; Twitter training differs from review domains.
- [CardiffNLP XLM-T repository](https://github.com/cardiffnlp/xlm-t)
  — multilingual sentiment classifier and training context.
- [MTEB Amazon Reviews Multi mirror](https://huggingface.co/datasets/mteb/amazon_reviews_multi)
  — secondary data files; [original multilingual corpus paper](https://aclanthology.org/2020.emnlp-main.369/).
- [FAISS metric documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
  — cosine/normalization conventions.
- [FAISS clustering FAQ](https://github.com/facebookresearch/faiss/wiki/FAQ)
  — training-size considerations for quantization.
- [Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)
  — temperature scaling.

This bibliography is a starting point. Verify citation metadata and supplement with
directly relevant sentiment-aware/filtered retrieval work before manuscript submission.
