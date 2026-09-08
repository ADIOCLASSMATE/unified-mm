"""Run audited V4 extraction with F's own architecture/configuration."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import probe_unified_semantics_v2 as v2

v2.RUN = ROOT / "output/unified-f-on-b-0p6b-100b-imagenet-split-s42-r1"

from scripts.extract_unified_geometry_v4 import main

if __name__ == "__main__":
    main()
