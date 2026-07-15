"""Execute all Week 4 notebook code cells without requiring a Jupyter server."""

from __future__ import annotations

import json
from pathlib import Path


def display(*_objects: object) -> None:
    """No-op replacement for notebook display during automated checks."""


def execute_notebook(notebook_path: Path) -> int:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"display": display, "__name__": "__notebook__"}
    code_count = 0
    for position, cell in enumerate(notebook["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        code_count += 1
        source = "".join(cell["source"])
        exec(compile(source, f"{notebook_path}#cell-{position}", "exec"), namespace)
    return code_count


def main() -> None:
    for path in [
        Path("notebooks/01-ai-ml-problem-framing.ipynb"),
        Path("notebooks/02-metrics-baselines-operating-point.ipynb"),
    ]:
        count = execute_notebook(path)
        print(f"PASS — {path}: {count} code cells")


if __name__ == "__main__":
    main()

