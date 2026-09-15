"""Verification script for lazy loading and import safety.

Ensures heavy AI dependencies (torch, faiss, transformers) are NOT eagerly loaded
at application startup.
"""

from __future__ import annotations

import sys


def main() -> int:
    import app.main  # noqa: F401

    forbidden = ["torch", "faiss", "sentence_transformers", "transformers"]
    loaded = [mod for mod in forbidden if mod in sys.modules]

    if loaded:
        print(f"FAILED: Eager import detected for: {', '.join(loaded)}", file=sys.stderr)
        return 1

    print("Import safety passed: No heavy ML packages were imported eagerly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
