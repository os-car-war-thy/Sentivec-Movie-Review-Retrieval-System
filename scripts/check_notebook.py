"""Validate schema, Python syntax, and clean output state without running research."""

import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "notebooks" / "SentiVec_Reproducible_Study.ipynb"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{path.name}:cell-{i}")
            if cell.outputs or cell.execution_count is not None:
                raise ValueError(f"Clear outputs/execution counts before committing cell {i}.")
    print(f"Valid notebook: {len(notebook.cells)} cells; Python syntax valid; no saved outputs.")


if __name__ == "__main__":
    main()
