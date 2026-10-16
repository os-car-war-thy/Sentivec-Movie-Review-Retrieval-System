import pandas as pd
import pytest

from sentivec.config import profile
from sentivec.data import assert_disjoint, canonicalize, load_imdb, split_calibration_validation


def test_canonical_identity_handles_html_and_whitespace():
    frame = pd.DataFrame({"text": ["good<br /> film", "good  film", None], "label": [1, 1, 0]})
    result = canonicalize(frame, "fixture")
    assert len(result) == 1
    assert result.iloc[0].text == "good film"


def test_conflicting_duplicate_labels_are_rejected():
    with pytest.raises(ValueError, match="conflicting"):
        canonicalize(pd.DataFrame({"text": ["same", "same"], "label": [0, 1]}), "fixture")


def test_split_reproducibility_and_disjointness(tmp_path):
    path = tmp_path / "fixture.csv"
    pd.DataFrame({"review": [f"Unique review number {i}" for i in range(1000)],
                  "sentiment": ["positive" if i % 2 else "negative" for i in range(1000)]}).to_csv(
                      path, index=False)
    config = profile("smoke")
    first, provenance = load_imdb(path, config, 42)
    second, _ = load_imdb(path, config, 42)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    calibration = split_calibration_validation(first["validation"], 42)
    assert_disjoint({"corpus": first["corpus"], "test": first["test"], **calibration})
    assert len(first["corpus"]) == 400
    assert len(provenance["sha256"]) == 64


def test_leakage_is_rejected():
    frame = canonicalize(pd.DataFrame({"text": ["good", "bad"], "label": [1, 0]}), "fixture")
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint({"one": frame, "two": frame})


def test_unknown_labels_do_not_silently_disappear(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"review": ["review"], "sentiment": ["neutral"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unknown"):
        load_imdb(path, profile("smoke"), 42)
