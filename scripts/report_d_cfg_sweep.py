#!/usr/bin/env python3
"""Validate and plot B / D CFG results; optionally refresh as jobs finish."""
from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import fcntl
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.build_evaluation_report import check_not_invalidated, read, report_model_specs, require, within, write


from scripts.report_s2_cfg_sweep import score as common_score


def score(root, relative, cfg, spec):
    result = common_score(root, relative, cfg, spec)
    if result is not None and spec['id'] == 'd_on_b':
        data = read(within(root, relative))
        guard = data['model_source_load'].get('post_load_validation', {})
        expected = {f'model.backbone_flow_time_embedder.mlp.{layer}.{kind}'
                    for layer in (0, 2) for kind in ('weight', 'bias')}
        require(guard.get('schema') == 'dynamic_xt_hf_time_embedding_values_v1'
                and guard.get('complete') is True
                and guard.get('comparison') == 'exact_after_model_dtype_cast'
                and set(guard.get('checked_parameters', [])) == expected,
                f'{relative}: D time embedding load has not passed exact validation')
        require(set(data['strategies']) == {'spatial_halton'}, f'{relative}: wrong generation order')
    return result


def collect(root, selection):
    selected = selection['d_cfg_sweep']
    gallery = read(root / selection['qualitative'] / 'manifest.json')
    specs = {m['id']: m for m in report_model_specs(root, selection, gallery['models'])}
    baseline = next(m for m in selection['flow_head_scale']['models'] if m['id'] == 'b_x0')['metrics']
    rows = [{'cfg': cfg, 'b': score(root, baseline[str(cfg)], cfg, specs['b_x0']),
             'd': score(root, selected['sources'][str(cfg)], cfg, specs['d_on_b'])}
            for cfg in selected['cfg_values']]
    completed = sum(row['d'] is not None for row in rows)
    jobs = []
    protocol = read(root / selected['root'] / 'protocol.json')
    launch = Path(protocol['launch_root'])
    for arm in protocol['arms']:
        progress = launch / f"{arm['kind']}-progress.json"
        state = read(progress) if progress.exists() else {'status': 'SUBMITTED'}
        jobs.append({'name': arm['job_name'], 'status': state['status'], 'results': state.get('results', [])})
    return {'schema': 'd_cfg_comparison_v1', 'rows': rows, 'jobs': jobs,
            'completed': completed, 'total': len(rows), 'complete': completed == len(rows),
            'protocol': {'samples': 50000, 'heun_steps': 10, 'seed': 42,
                         'b_batch': 4096, 'd_batch': 4096, 'd_generation': 'dynamic_xt', 'order': 'spatial_halton'}}


def plot(study, directory):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    cfgs = [row['cfg'] for row in study['rows']]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    for ax, key, ylabel in zip(axes, ('fid', 'is'), ('FID (lower is better)', 'Inception Score (higher is better)')):
        for model, label, color in [('b', 'B', '#12655f'), ('d', 'D', '#b65d2e')]:
            values = [row[model][key] if row[model] else math.nan for row in study['rows']]
            ax.plot(cfgs, values, 'o-', label=label, color=color, linewidth=2, markersize=5)
            if key == 'is':
                std = [row[model]['is_std'] if row[model] else math.nan for row in study['rows']]
                ax.fill_between(cfgs, [v-s for v,s in zip(values,std)], [v+s for v,s in zip(values,std)], color=color, alpha=.12)
        ax.set(xlabel='Classifier-free guidance (CFG)', ylabel=ylabel, xticks=cfgs)
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"B / D CFG sweep | D: {study['completed']}/{study['total']} complete\n"
                 'ImageNet-val 50K | final EMA step 95415 | Heun 10 | seed 42', fontsize=13)
    for extension in ('png', 'pdf', 'svg'):
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.'+extension, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            fig.savefig(temporary, dpi=180, bbox_inches='tight', facecolor='white')
            os.replace(temporary, directory / f'cfg-sweep.{extension}')
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(fig)


def refresh():
    root = REPO / 'output/evaluation'
    selection = read(REPO / 'configs/protocols/evaluation_report.json')
    directory = within(root, selection['d_cfg_sweep']['output'])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.refresh.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        study = collect(root, selection)
        previous = read(directory / 'comparison.json') if (directory / 'comparison.json').exists() else {}
        unchanged = all(previous.get(k) == v for k,v in study.items())
        if not unchanged or not all((directory / f'cfg-sweep.{ext}').exists() for ext in ('png', 'pdf', 'svg')):
            plot(study, directory)
            stream = io.StringIO()
            writer = csv.writer(stream)
            writer.writerow(['cfg','model','fid','is','is_std','source'])
            for row in study['rows']:
                for model in ('b', 'd'):
                    value = row[model] or {}
                    writer.writerow([row['cfg'], model, *[value.get(k, '') for k in ('fid','is','is_std','source')]])
            write(directory / 'cfg-sweep.csv', stream.getvalue())
            study['updated_at'] = datetime.now(UTC).isoformat(timespec='seconds')
            write(directory / 'comparison.json', json.dumps(study, ensure_ascii=False, indent=2)+'\n')
            print(json.dumps({'updated_at':study['updated_at'],'completed':study['completed'],'total':study['total']}), flush=True)
        return study


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=60)
    parser.add_argument('--timeout', type=float, default=172800)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            study = refresh()
            terminal = all(j['status'] in ('SUCCEEDED','FAILED') for j in study['jobs'])
            if not args.watch or study['complete'] or terminal:
                break
        except Exception as error:
            if not args.watch:
                raise
            print(json.dumps({'error':str(error),'at':datetime.now(UTC).isoformat()}), flush=True)
        if time.monotonic() >= deadline:
            raise SystemExit('CFG report watcher timed out; restart to continue refreshing.')
        time.sleep(args.interval)
