#!/usr/bin/env python3
"""Dedicated offline FID/IS entry for Dynamic-XT checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pretrain.train_selfless_flow_dynamic_xt import (
    load_dynamic_xt_model_tokenizer,
)
from scripts.evaluate_single_stream_fid_is import main

if __name__ == "__main__":
    main(model_loader=load_dynamic_xt_model_tokenizer)
