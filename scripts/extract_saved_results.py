"""Extract numeric tables from existing saved outputs; never execute notebook code."""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Enhanced_SentiVec_with_Reviews_Addressed.ipynb"
DESTINATION = ROOT / "results" / "original_notebook"


def saved_text(notebook, cell_index):
    return "\n".join("".join(output.get("text", []))
                     for output in notebook["cells"][cell_index].get("outputs", [])
                     if output.get("output_type") == "stream" and output.get("name") != "stderr")


def write_csv(name, columns, rows):
    with (DESTINATION / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    retrieval_text = saved_text(notebook, 25).split("EVALUATION SUMMARY TABLE")[-1]
    retrieval = [line.split() for line in retrieval_text.splitlines()
                 if line.strip().startswith(("without_sentiment ", "with_sentiment "))]
    classifiers = [line.split() for line in saved_text(notebook, 16).splitlines()
                   if line.startswith(("DistilBERT ", "VADER ", "TextBlob "))]
    thresholds = [line.split() for line in saved_text(notebook, 14).splitlines()
                  if line.strip().split() and line.strip().split()[0] in
                  {"0.50", "0.60", "0.70", "0.75", "0.80", "0.85", "0.90", "0.95"}]
    if not (len(retrieval) == 8 and all(len(row) == 13 for row in retrieval)
            and len(classifiers) == 3 and all(len(row) == 8 for row in classifiers)
            and len(thresholds) == 8 and all(len(row) == 5 for row in thresholds)):
        raise ValueError("Saved output structure changed; inspect the source before extracting.")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    write_csv("retrieval.csv", ["approach", "index", "num_queries", "avg_latency_ms",
                              "std_latency_ms", "reported_mAP", "reported_MRR", "P@5", "R@5",
                              "P@10", "R@10", "P@20", "R@20"], retrieval)
    write_csv("classifiers.csv", ["classifier", "accuracy", "precision", "recall", "f1",
                                "avg_confidence", "avg_latency_ms", "std_latency_ms"], classifiers)
    write_csv("thresholds.csv", ["threshold", "retained_queries", "retention_rate", "accuracy",
                               "avg_confidence"], thresholds)
    write_csv("mixed_tone.csv", ["classifier", "correct", "num_examples", "accuracy"],
              [["DistilBERT", 7, 8, .875], ["VADER", 5, 8, .625], ["TextBlob", 3, 8, .375]])
    metadata = {"source_notebook": SOURCE.name,
                "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                "source_cells_zero_based": {"retrieval": 25, "classifiers": 16,
                                             "thresholds": 14, "mixed_tone": 18},
                "extraction": "Read notebook JSON and existing text outputs; no code execution.",
                "status": "Previously saved results, not independently reproduced.",
                "new_implementation_results": False,
                "note": "Printed precision retained. Retrieval uses the final duplicate evaluation cell."}
    (DESTINATION / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                  encoding="utf-8")
    print("Extracted saved numeric results without executing the notebook.")


if __name__ == "__main__":
    main()
