#!/usr/bin/env python3
"""Shared fixed-alpha language-prior calibration for generative VLM scores."""

from __future__ import annotations

import math

import torch


LANGUAGE_PRIOR_ALPHA = 1.0
LANGUAGE_PRIOR_ESTIMATOR = "candidate_image_logmeanexp"


def language_prior_debiased_scores(
    conditional_mean_token_loglikelihood: torch.Tensor,
    *,
    alpha: float = LANGUAGE_PRIOR_ALPHA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``log P(t|i) - alpha * logmeanexp_i(log P(t|i))``.

    Rows are candidate images and columns are candidate texts.  The input is
    the length-normalized VisualGPTScore in log space.  The marginal must be
    computed in probability space, hence ``logsumexp`` rather than a column
    mean of log scores.
    """

    scores = conditional_mean_token_loglikelihood
    if scores.ndim != 2:
        raise ValueError("conditional scores must be a rank-two image-text matrix")
    if int(scores.shape[0]) <= 0 or int(scores.shape[1]) <= 0:
        raise ValueError("conditional score matrix must be non-empty")
    if not math.isfinite(float(alpha)) or float(alpha) != LANGUAGE_PRIOR_ALPHA:
        raise ValueError("the formal protocol fixes language-prior alpha to 1.0")
    scores = scores.float()
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("conditional score matrix contains non-finite values")
    text_log_prior = torch.logsumexp(scores, dim=0) - math.log(int(scores.shape[0]))
    calibrated = scores - float(alpha) * text_log_prior.unsqueeze(0)
    return calibrated, text_log_prior

