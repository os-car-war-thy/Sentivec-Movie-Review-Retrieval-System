"""Predeclared diagnostic probes, kept separate from representative test metrics."""

from pathlib import Path

import pandas as pd

from .metrics import polarity

ENGLISH_PROBES = [
    ("This movie was good but I didn't enjoy it", 0, "contrast"),
    ("This movie was bad but I did enjoy it", 1, "contrast"),
    ("Great acting but a terrible plot made this unwatchable", 0, "aspect_conflict"),
    ("Slow pacing but the emotional payoff was worth it", 1, "aspect_conflict"),
    ("I wanted to like this movie but couldn't", 0, "negation"),
    ("Not perfect but definitely worth watching", 1, "negation"),
    ("Amazing visuals can't save the boring story", 0, "aspect_conflict"),
    ("Despite its flaws, I loved every moment", 1, "contrast"),
    ("I cannot recommend this film", 0, "negation"),
    ("I cannot recommend this film enough", 1, "negation"),
    ("I expected a masterpiece and got a tedious mess", 0, "expectation"),
    ("I expected a disaster and was pleasantly surprised", 1, "expectation"),
    ("The trailer was excellent; the actual movie was awful", 0, "target_shift"),
    ("The trailer was awful; the actual movie was excellent", 1, "target_shift"),
    ("A wonderful way to waste two hours of my life", 0, "sarcasm"),
    ("The only good thing was that it finally ended", 0, "sarcasm"),
    ("The acting was good and the story was bad", None, "ambiguous"),
    ("The film runs for two hours", None, "neutral"),
    ("Some scenes worked for me and some did not", None, "ambiguous"),
    ("Show me reviews discussing both strengths and weaknesses", None, "explicit_mixed_intent"),
]
GERMAN_PROBES = [
    ("Der Film war gut, aber ich habe ihn nicht genossen", 0, "contrast"),
    ("Der Film war schlecht, aber ich habe ihn trotzdem genossen", 1, "contrast"),
    ("Die tollen Bilder retten die langweilige Geschichte nicht", 0, "aspect_conflict"),
    ("Nicht perfekt, aber absolut sehenswert", 1, "negation"),
    ("Ich kann diesen Film nicht empfehlen", 0, "negation"),
    ("Der Film dauert zwei Stunden", None, "neutral"),
]


def run_robustness(model, temperature, queries, directory, language="en"):
    probes = ENGLISH_PROBES if language == "en" else GERMAN_PROBES if language == "de" else []
    if probes:
        p, confidence, neutral = polarity(model.scores([item[0] for item in probes]), temperature)
        rows = []
        for i, (text, expected, category) in enumerate(probes):
            rows.append({"text": text, "expected_polarity": expected, "category": category,
                         "prediction": int(p[i] >= 0.5), "gate_confidence": float(confidence[i]),
                         "neutral_probability": float(neutral[i]),
                         "correct": (bool((p[i] >= 0.5) == expected) if expected is not None else None),
                         "evidence_type": "handwritten diagnostic, not a representative benchmark"})
        pd.DataFrame(rows).to_csv(Path(directory) / "mixed_tone_diagnostics.csv", index=False)
    rows = []
    for row in queries.iloc[:20].itertuples():
        variants = [row.text, row.text.upper(), "\n".join(row.text.split())]
        p, confidence, _ = polarity(model.scores(variants), temperature)
        for i, variant in enumerate(["original", "uppercase", "line_breaks"]):
            rows.append({"query_id": row.doc_id, "variant": variant,
                         "prediction": int(p[i] >= 0.5), "gold_label": row.label,
                         "consistent_with_original": bool((p[i] >= 0.5) == (p[0] >= 0.5)),
                         "correct": bool((p[i] >= 0.5) == row.label),
                         "gate_confidence": float(confidence[i])})
    pd.DataFrame(rows).to_csv(Path(directory) / "format_robustness.csv", index=False)
