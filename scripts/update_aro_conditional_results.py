#!/usr/bin/env python3
"""Recover ARO conditional scores from saved predictions and update all summaries."""
import argparse
from copy import deepcopy
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.build_evaluation_report import read, write, check_not_invalidated
from utils.evaluation.aro import ARO_TASKS, ARO_PRIMARY, CONDITIONAL_SCORE, ARO_SCORE_VARIANT, update_aro_metrics


def propagate(value, changes, parent=''):
    if isinstance(value, list):
        return [propagate(item, changes, parent) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key in changes and isinstance(item, dict) and 'metrics' in item:
            result[key] = deepcopy(item)
            metrics = deepcopy(changes[key]['metrics'])
            if 'categories' not in item['metrics']:
                metrics.pop('categories', None)
            result[key]['metrics'] = metrics
            result[key]['aro_metric_revision'] = changes[key]['aro_metric_revision']
        elif key in changes and parent == 'primary_metrics':
            result[key] = changes[key]['metrics']['conditional_pairwise']['win_rate']
        else:
            result[key] = propagate(item, changes, key)
    if 'reported_score_variant' in result and 'primary_candidate_score' in result:
        result['reported_score_variant'] = ARO_SCORE_VARIANT
        result['task_primary_candidate_scores'] = {task: CONDITIONAL_SCORE for task in sorted(ARO_TASKS)}
    if 'image_text_matching_score_variant' in result:
        result['image_text_matching_score_variant'] = ARO_SCORE_VARIANT
        result['aro_primary_candidate_score'] = CONDITIONAL_SCORE
    return result


def migrate(root, destination, apply=False):
    root, destination = root.resolve(), destination.resolve()
    paths = [Path(p) for p in subprocess.check_output(
        ['rg','--files',str(root),'-g','aro_vg_relation.jsonl','-g','aro_vg_attribution.jsonl'], text=True).splitlines()]
    staged, audit, skipped = {}, [], []

    def stage(path, payload):
        original = staged.get(path, read(path))
        if payload != original:
            staged[path] = payload

    for prediction in sorted(paths):
        try:
            check_not_invalidated(prediction, root)
        except ValueError as error:
            skipped.append({'source':str(prediction.relative_to(root)), 'reason':str(error)})
            continue
        task = prediction.stem
        directory = prediction.parent.parent
        summary_path = directory / 'summaries' / f'{task}.json'
        if not summary_path.exists():
            skipped.append({'source':str(prediction.relative_to(root)), 'reason':'no task summary'})
            continue
        with prediction.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        original = read(summary_path)
        if original['metrics']['records'] != len(rows):
            raise ValueError(f'ARO record count mismatch: {prediction}')
        if len({row['item_index'] for row in rows}) != len(rows) or any(row['task'] != task for row in rows):
            raise ValueError(f'duplicate or mixed ARO predictions: {prediction}')
        updated = {**original, 'metrics': update_aro_metrics(original['metrics'], rows),
            'aro_metric_revision': {'schema':'aro_conditional_primary_v1',
                'prediction_source':str(prediction.relative_to(root)),
                'candidate_score':CONDITIONAL_SCORE, 'language_prior_included':True,
                'recovery':'saved conditional score, or debiased score + recorded text prior',
                'aggregation':'strict pairwise win rate over all retained records; ties are incorrect'}}
        stage(summary_path, updated)
        targets = [directory/'summary.json', directory/'manifest.json']
        # Only task-local containers and their native-suite parents; no unrelated snapshots.
        for parent in (directory.parent, directory.parent.parent):
            targets.extend(parent/name for name in ('pretraining_native_understanding_summary.json', 'native_full_evaluation_summary.json'))
        for path in targets:
            if path.exists():
                data = staged.get(path, read(path))
                stage(path, propagate(data, {task:updated}))
        audit.append({'source':str(prediction.relative_to(root)), 'records':len(rows),
            'old_primary':original['metrics']['primary_metric'],
            'old_debiased':original['metrics'].get('language_prior_debiased_pairwise',{}).get('win_rate'),
            'new_primary':ARO_PRIMARY, **updated['metrics']['conditional_pairwise']})
    destination.mkdir(parents=True, exist_ok=True)
    for path, data in staged.items():
        write(destination/'updated'/path.relative_to(root), json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    report = {'schema':'aro_conditional_migration_v1','at':datetime.now(UTC).isoformat(),
        'prediction_files':len(paths),'updated_task_results':len(audit),'updated_json_files':len(staged),
        'raw_predictions_modified':False,'applied':apply,'tasks':audit,'skipped':skipped}
    write(destination/'audit.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if apply:
        for path in staged:
            backup = destination/'originals'/path.relative_to(root)
            if not backup.exists():
                backup.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,backup)
        for path, data in staged.items():
            write(path,json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('tasks','skipped')},ensure_ascii=False),flush=True)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO/'output/evaluation')
    parser.add_argument('--destination',type=Path,default=REPO/'output/evaluation/migrations/20260915-aro-conditional')
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    migrate(args.root,args.destination,args.apply)
