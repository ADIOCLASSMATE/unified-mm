#!/usr/bin/env python3
"""16-NPU acceptance: train, full validation, save, resume, and reload exports."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.launch_unified_flow_head_scaling import launch_plan


def override(command, values):
    result = [part for part in command if part.split('=', 1)[0] not in values]
    return result + [f'{key}={value}' for key, value in values.items()]


def read(path):
    return json.loads(Path(path).read_text())


def checked_run(command, path):
    print(json.dumps({'event': 'lifecycle_stage', 'log': str(path), 'command': command}), flush=True)
    with path.open('w') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--depth', type=int, choices=(16, 30), required=True)
    parser.add_argument('--label', default='validation-r2')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference-generation-file', type=Path)
    args = parser.parse_args()
    os.chdir(ROOT)
    report = args.output_dir.resolve()
    report.mkdir(parents=True, exist_ok=True)
    plan = launch_plan(args.depth, smoke=True, label=args.label, steps=5, environment=dict(os.environ))
    run = ROOT / plan['output_root']
    if (run / 'config.yaml').exists():
        raise FileExistsError(f'acceptance must start fresh: {run}')
    # Validate twice to exercise cold and warm caches, then train and save again.
    # The periodic checkpoint/model pair is saved immediately before validation.
    fresh = override(plan['command'], {'experiment.val_every': 2, 'experiment.save_every': 2,
                                      'experiment.save_ema_eval_every': 2})
    (report / 'launch-plan.json').write_text(json.dumps({**plan, 'command': fresh}, indent=2) + '\n')
    checked_run(plan['preflight'], report / 'preflight.json')
    checked_run(fresh, report / 'fresh-training.log')
    runtime = read(run / 'training_runtime_metrics.json')
    assert runtime['global_step'] == 5 and runtime['run_start_global_step'] == 0
    assert runtime['finite_loss_microbatches_checked'] == 20
    assert math.isfinite(runtime['last_logged_loss'])
    validation = ROOT / 'output/evaluation/training-validation' / run.name
    validations = []
    fixed_subset = None
    for step in (2, 4):
        whole = read(validation / f'validation_summary_step_{step}.json')
        loss = read(validation / f'validation_unified_loss_metrics_step_{step}.json')
        downstream = read(validation / f'downstream_validation/step-{step}/summary.json')
        subset = read(validation / f'downstream_validation/step-{step}/subset.json')
        assert whole['step'] == loss['global_step'] == downstream['step'] == step
        assert whole['complete'] and loss['complete'] and downstream['complete']
        assert downstream['within_time_budget'] and len(downstream['tasks']) == 11
        assert all(row['complete'] for row in downstream['tasks'].values())
        assert downstream['prepare_cache_hit'] is (step == 4)
        assert downstream['weight_source'] == 'ema' and downstream['ema']['step'] == step
        assert loss['model_mode'] == 'train_no_grad' and loss['model_weights'] == 'current'
        assert loss['imagenet_subset'] == subset['imagenet_subset']
        assert loss['imagenet_subset']['sample_ids'] == subset['sample_ids']['imagenet']
        assert Counter(loss['imagenet_subset']['class_indices']) == {i: 2 for i in range(1000)}
        if fixed_subset is None:
            fixed_subset = loss['imagenet_subset']
        assert loss['imagenet_subset'] == fixed_subset
        assert loss['subsets']['t2i']['samples'] == loss['subsets']['i2t']['samples'] == 2000
        assert loss['subsets']['climbmix']['samples'] == 400
        assert loss['pure_text']['independence'] in ('source_rows_excluded_since_training_start',
                                                   'may_have_been_seen_in_training')
        assert math.isclose(loss['metrics']['val/loss'], sum(loss['metrics'][f'val/weighted_contribution_{s}']
                           for s in ('t2i','i2t','climbmix')), rel_tol=1e-6)
        validations.append(whole)
    from pretrain.train_selfless_flow import _validate_checkpoint_complete
    for step in (2, 4, 5):
        _validate_checkpoint_complete(run / f'checkpoint-{step}', expected_global_step=step)
    for step in (2, 4):
        assert read(run / f'hf_model-{step}-eval-pair.json')['global_step'] == step
    resume = override(plan['command'], {'experiment.resume_from_checkpoint': run / 'checkpoint-5',
                                       'training.stop_after_steps': 6, 'experiment.val_every': 0})
    checked_run(resume, report / 'resumed-training.log')
    resumed = read(run / 'training_runtime_metrics.json')
    assert resumed['run_start_global_step'] == 5 and resumed['global_step'] == 6
    assert resumed['steps_this_run'] == 1 and resumed['finite_loss_microbatches_checked'] == 4
    assert math.isfinite(resumed['last_logged_loss'])
    _validate_checkpoint_complete(run / 'checkpoint-6', expected_global_step=6)
    for kind, checkpoint in (('current', 'hf_model-final'), ('ema', 'hf_model-final-ema')):
        reference_args = (
            ['--reference-generation-file', str(args.reference_generation_file)]
            if kind == 'ema' and args.reference_generation_file is not None else []
        )
        checked_run([sys.executable, 'scripts/smoke_unified_flow_head_generation.py', '--depth', str(args.depth),
                     '--checkpoint', str(run / checkpoint), '--weights', kind,
                     '--output-dir', str(report / f'{kind}-reload'), *reference_args], report / f'{kind}-reload.log')
        assert read(report / f'{kind}-reload/report.json')['passed']
    result = {'schema':'training_validation_lifecycle_smoke_v1', 'passed':True, 'depth':args.depth,
              'source':str(ROOT), 'run':str(run), 'world_size':16,
              'fresh_runtime':runtime, 'resumed_runtime':resumed,
              'validations':validations, 'cold_and_warm_validation_passed':True,
              'imagenet_samples':2000, 'imagenet_per_class':2,
              'climbmix_samples':400, 'complete_downstream_tasks':11,
              'climbmix_independence':loss['pure_text']['independence'],
              'checkpoint_steps_verified':[2,4,5,6], 'periodic_model_pair_steps':[2,4],
              'current_and_ema_reload_generation_passed':True, 'runtime_hashing_enabled':False}
    (report / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (report / 'SMOKE_PASSED').write_text('passed\n')
    print(json.dumps({'event':'lifecycle_acceptance_passed', 'depth':args.depth,
                      'validation_seconds':[row['wall_seconds'] for row in validations],
                      'report':str(report / 'report.json')}), flush=True)


if __name__ == '__main__':
    main()
