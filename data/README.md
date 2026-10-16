# Local data

Place `IMDB Dataset.csv` in `data/raw/`. Obtain it from
[the author's Kaggle page](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
or run `sentivec download` after installing research dependencies.

Put final, adjudicated human labels in `data/annotations/judgments.csv`.
See `docs/ANNOTATION.md` for the schema and procedure.

All contents of this directory except this README are ignored by Git. Do not commit
raw reviews, access tokens, credentials, or annotation files containing review text.
