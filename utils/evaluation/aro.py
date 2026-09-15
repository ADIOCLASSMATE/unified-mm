"""ARO ranks candidate texts by conditional likelihood, retaining text priors."""
from collections import defaultdict
import math
import statistics

ARO_TASKS = frozenset({'aro_vg_relation', 'aro_vg_attribution'})
CONDITIONAL_SCORE = 'conditional_mean_token_loglikelihood'
ARO_PRIMARY = 'conditional_pairwise.win_rate'
ARO_SCORE_VARIANT = 'conditional_aro_debiased_other_tasks'
DEBIASED_SCORE = 'language_prior_debiased_mean_token_loglikelihood'


def conditional_score(candidate):
    """Read direct scores or add back the prior saved by older evaluations."""
    if CONDITIONAL_SCORE in candidate:
        value = candidate[CONDITIONAL_SCORE]
    elif 'normalized_loglikelihood' in candidate:
        value = candidate['normalized_loglikelihood']
    else:
        alpha = candidate.get('language_prior_alpha')
        if alpha != 1.0:
            raise ValueError('ARO score recovery requires recorded language_prior_alpha=1')
        value = candidate[DEBIASED_SCORE] + alpha * candidate['estimated_language_prior_log_score']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('ARO requires a finite conditional likelihood')
    return float(value)


def conditional_pairwise_metrics(rows):
    margins = []
    for row in rows:
        candidates = row['candidate_scores']
        label = row['label']
        if len(candidates) != 2 or label not in (0, 1):
            raise ValueError('ARO requires two candidates and a valid positive label')
        margins.append(conditional_score(candidates[label]) - conditional_score(candidates[1-label]))
    if not margins:
        raise ValueError('cannot summarize empty ARO predictions')
    wins = sum(m > 0 for m in margins)
    ties = sum(m == 0 for m in margins)
    return {'records': len(margins), 'primary_metric': ARO_PRIMARY,
            'primary_candidate_score': CONDITIONAL_SCORE, 'language_prior_included': True,
            'conditional_pairwise': {'win_rate': wins/len(margins), 'tie_rate': ties/len(margins),
                'loss_rate': (len(margins)-wins-ties)/len(margins),
                'mean_margin': statistics.mean(margins), 'median_margin': statistics.median(margins)}}


def update_aro_metrics(metrics, rows):
    """Keep previous diagnostics and replace only the ARO primary selection."""
    result = {**metrics, **conditional_pairwise_metrics(rows)}
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get('category') or 'uncategorized')].append(row)
    result['categories'] = {category: {**metrics.get('categories', {}).get(category, {}),
                                      **conditional_pairwise_metrics(group)}
                            for category, group in sorted(groups.items())}
    return result
