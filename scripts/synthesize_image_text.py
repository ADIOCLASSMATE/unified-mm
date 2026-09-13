#!/usr/bin/env python3
"""Current B512 pipeline: source reuse -> SII repair -> retry-exhausted Codex fallback.

Run as `python -m scripts.synthesize_image_text --help`.
Historical sol-judge runs live under scripts.legacy and are not the default route.
"""
from data_synthesis.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
