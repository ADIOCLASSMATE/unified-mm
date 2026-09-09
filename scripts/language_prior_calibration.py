#!/usr/bin/env python3
"""Shared fixed-alpha language-prior calibration for generative VLM scores."""

from __future__ import annotations

import math

import torch



from utils.evaluation.calibration import (  # noqa: F401 -- compatibility exports
    LANGUAGE_PRIOR_ALPHA,
    LANGUAGE_PRIOR_ESTIMATOR,
    language_prior_debiased_scores,
)
