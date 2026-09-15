#!/usr/bin/env python3
"""Validate and plot B / S2-single CFG results; optionally refresh as jobs finish."""
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


def score(root, relative, cfg, spec):
    path = within(root, relative)
    check_not_invalidated(path, root)
    if not path.exists():
        return None
    data = read(path)
    s2 = spec['architecture'] == 'showo2_unified'
    expected = {'schema': 'selfless_imagenet_val_t2i_fid_is_v2', 'project_formal_protocol': True,
        'runtime_hashing_enabled': False, 'samples_requested': 50000, 'samples_evaluated': 50000,
        'split': 'val', 'cfg': cfg, 'seed': 42, 'flow_solver': 'heun', 'cfg_schedule': 'constant',
        'temperature': 1.0, 'parallel_rate': 1, 'backbone_kv_cache': not s2,
        'batch_size': 2048 if s2 else 4096}
    for key, value in expected.items():
        require(data.get(key) == value, f'{path}: {key} differs from the frozen protocol')
    require(str(data['sampling_steps']) == '10', f'{path}: wrong Heun steps')
    source = data['evaluation_model_source']
    require(source['kind'] == 'hf_final_ema' and source['global_step'] == spec['source']['global_step']
            and source['floating_dtype'] == 'float32'
            and Path(source['path']).resolve() == Path(spec['checkpoint']).resolve(), f'{path}: checkpoint mismatch')
    require(data['distributed']['world_size'] == 16, f'{path}: wrong world size')
    precision = data['precision_protocol']
    require(precision['model_dtype'] == 'bf16' and precision['vae_dtype'] == 'fp32'
            and precision['flow_integrator_dtype'] == 'fp32', f'{path}: precision mismatch')
    contracts = data['implementation_contracts']
    require(contracts['canonical_initial_noise_enabled'] and contracts['paired_sample_count'] == 50000
            and contracts['ordered_sample_count'] == 50000, f'{path}: sample pairing mismatch')
    require(contracts['backbone_attention']['dual_stream_attention_contract'] == spec['backbone_attention'],
            f'{path}: attention mismatch')
    if s2:
        require(contracts.get('full_image_refresh_each_velocity') is True
                and contracts.get('image_generation_order_applicable') is False,
                f'{path}: S2 requires full-image flow')
    metric = data['metric_protocol']
    require(metric['protocol_name'] == 'imagenet_val_fid50k_torch_fidelity_stratified_is'
            and metric['reference_distribution'] == 'imagenet_val_50000' and metric['is_splits'] == 10,
            f'{path}: metric protocol mismatch')
    require(Path(data['real_stats_path']).resolve() ==
            (REPO / 'public/datasets/imagenet_full/fid_stats/inception_v3_2048_imagenet_val50000_256.pt').resolve(),
            f'{path}: real reference mismatch')
    require(data['mechanism_diagnostics']['generated_latent_finite_rate'] == 1.0,
            f'{path}: nonfinite generation')
    result = data['strategies']['spatial_halton']
    require(result['count'] == 50000, f'{path}: incomplete generation')
    values = {'fid': result['fid'], 'is': result['inception_score_mean'], 'is_std': result['inception_score_std']}
    require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values.values()),
            f'{path}: invalid score')
    return {**values, 'source': relative}


def collect(root, selection):
    selected = selection['s2_cfg_sweep']
    gallery = read(root / selection['qualitative'] / 'manifest.json')
    specs = {m['id']: m for m in report_model_specs(root, selection, gallery['models'])}
    baseline = next(m for m in selection['flow_head_scale']['models'] if m['id'] == 'b_x0')['metrics']
    rows = [{'cfg': cfg, 'b': score(root, baseline[str(cfg)], cfg, specs['b_x0']),
             's2': score(root, selected['sources'][str(cfg)], cfg, specs['s2_single'])}
            for cfg in selected['cfg_values']]
    completed = sum(row['s2'] is not None for row in rows)
    jobs = []
    protocol = read(root / selected['root'] / 'protocol.json')
    launch = Path(protocol['launch_root'])
    for arm in protocol['arms']:
        progress = launch / f"{arm['kind']}-progress.json"
        state = read(progress) if progress.exists() else {'status': 'SUBMITTED'}
        jobs.append({'name': arm['job_name'], 'status': state['status'], 'results': state.get('results', [])})
    return {'schema': 's2_single_cfg_comparison_v1', 'rows': rows, 'jobs': jobs,
            'completed': completed, 'total': len(rows), 'complete': completed == len(rows),
            'protocol': {'samples': 50000, 'heun_steps': 10, 'seed': 42,
                         'b_batch': 4096, 's2_batch': 2048, 's2_generation': 'full_image_flow'}}


def plot(study, directory):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    cfgs = [row['cfg'] for row in study['rows']]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    for ax, key, ylabel in zip(axes, ('fid', 'is'), ('FID (lower is better)', 'Inception Score (higher is better)')):
        for model, label, color in [('b', 'B', '#12655f'), ('s2', 'S2-single', '#b65d2e')]:
            values = [row[model][key] if row[model] else math.nan for row in study['rows']]
            ax.plot(cfgs, values, 'o-', label=label, color=color, linewidth=2, markersize=5)
            if key == 'is':
                std = [row[model]['is_std'] if row[model] else math.nan for row in study['rows']]
                ax.fill_between(cfgs, [v-s for v,s in zip(values,std)], [v+s for v,s in zip(values,std)], color=color, alpha=.12)
        ax.set(xlabel='Classifier-free guidance (CFG)', ylabel=ylabel, xticks=cfgs)
        ax.grid(alpha=.25)
        ax.legend(frameon=False)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f"B / S2-single CFG sweep | S2: {study['completed']}/{study['total']} complete\n"
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
    directory = within(root, selection['s2_cfg_sweep']['output'])
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
                for model in ('b', 's2'):
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
