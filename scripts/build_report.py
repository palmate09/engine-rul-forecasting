"""Build a submission-ready DOCX from the Markdown report.

Requires the optional `pypandoc-binary` package:
    uv pip install --python .venv/bin/python pypandoc-binary

Usage:
    python scripts/build_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
SOURCE = REPORT_DIR / "PROJECT_REPORT.md"
OUTPUT = REPORT_DIR / "PROJECT_REPORT.docx"


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Report not found: {SOURCE}")

    try:
        import pypandoc
    except ImportError as exc:  # pragma: no cover - optional tooling
        raise SystemExit(
            "pypandoc is not installed. Run:\n"
            "  uv pip install --python .venv/bin/python pypandoc-binary"
        ) from exc

    # The Markdown report already contains a static, linked table of contents.
    # Keeping it avoids an empty Word TOC field that only fills after a manual
    # "Update field" action.
    text = SOURCE.read_text(encoding="utf-8")
    pypandoc.convert_text(
        text,
        "docx",
        format="markdown+tex_math_single_backslash+pipe_tables+task_lists",
        outputfile=str(OUTPUT),
        extra_args=[
            f"--resource-path={REPORT_DIR}",
            "--standalone",
        ],
    )
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
