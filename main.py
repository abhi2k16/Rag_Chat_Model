"""
Main executable entry point for the local RAG chat model.

Run:
    python main.py

This delegates to generation_rag.py, which builds the retrieval pipeline and
starts the interactive chat loop.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    generation_script = project_dir / "generation_rag.py"

    if not generation_script.exists():
        raise FileNotFoundError(f"Could not find {generation_script}")

    runpy.run_path(str(generation_script), run_name="__main__")


if __name__ == "__main__":
    main()
