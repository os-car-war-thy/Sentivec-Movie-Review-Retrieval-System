"""Clear code-cell outputs and widget state without changing notebook source."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook.get("metadata", {}).pop("widgets", None)
    args.notebook.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    print(f"Cleared {args.notebook}")


if __name__ == "__main__":
    main()
