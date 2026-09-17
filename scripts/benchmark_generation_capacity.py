#!/usr/bin/env python3
"""Find NPU generation batch limits with real inference, then time stable maxima.

Each trial gets a fresh process and one exclusive NPU. OOM is recorded and the
process exits, so allocator state from failed probes cannot affect measurements.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import html
import math
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.evaluation.model_contracts import S2_ATTENTION_CONTRACTS

GIB = 1024 ** 3
MODELS = ('b_x0', 's2_single')


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def bounds(rows):
    ok = [r['batch_size'] for r in rows if r['status'] == 'ok']
    oom = [r['batch_size'] for r in rows if r['status'] == 'oom']
    if any(r['status'] not in ('ok', 'oom') for r in rows):
        raise RuntimeError('non-OOM trial failure; inspect trial JSON/log before resuming')
    low = max(ok, default=0)
    high = min(oom, default=None)
    if high is not None and low >= high:
        raise RuntimeError('nonmonotonic memory limit; rerun conflicting trials')
    return low, high


def refine_batches(low, high, grain=8):
    values = range(low + grain, high, grain)
    values = list(values)
    if len(values) <= 8:
        return values
    return sorted({values[round(i * (len(values) - 1) / 7)] for i in range(8)})


def worker(args):
    import torch
    import torch_npu
    from omegaconf import OmegaConf
    from scripts.generate_unified_qualitative import (BASE_CONFIG, build_t2i_item, noise_for,
        validate_generation_trace, check_loaded_values, save_png)
    from utils.evaluation_model_source import (configure_model_source,
        resolve_evaluation_model_source, load_model_source_weights)
    from utils.utils import load_model_tokenizer
    from utils.imagenet_flow_batching import collate_imagenet_flow_cache
    from utils.image_generation_io import load_vae, decode_latents

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = torch.device('npu', args.device)
    manifest = read(args.root / 'manifest.json')
    spec = next(s for s in manifest['models'] if s['id'] == args.model)
    result = dict(model=args.model, batch_size=args.batch_size, device=args.device,
        phase=args.phase, started_at=utc(), status='running', hostname=os.uname().nodename,
        cfg=manifest['contract']['model_cfg'][args.model], solver='heun', steps=10,
        prompt_count=len(manifest['samples']['t2i']), prompt_batch='cyclic frozen gallery order',
        warmup_batches=args.warmup, measured_repeats=args.repeats,
        torch_version=torch.__version__, torch_npu_version=torch_npu.__version__)
    write(args.result, result)
    try:
        cfg = OmegaConf.load(BASE_CONFIG)
        cfg.training.runtime_hashing_enabled = False
        source = resolve_evaluation_model_source(spec['checkpoint'])
        configure_model_source(cfg, source)
        cfg.model.image_flow_num_sampling_steps = '10'
        model, tokenizer = load_model_tokenizer(cfg, model_dtype=torch.bfloat16)
        result['weight_source'] = load_model_source_weights(model, source)
        if args.device == 0:
            result['weight_check'] = check_loaded_values(model, spec['checkpoint'])
        model.to(device).eval()
        cfg.experiment.validation_vae_module_root = 'public/code/mar'
        cfg.experiment.validation_vae_path = 'public/vae/mar-kl16/kl16.ckpt'
        cfg.experiment.validation_vae_scaling_factor = .2325
        vae = load_vae(cfg, device, 'fp32')
        use_cache = spec['backbone_attention'] not in S2_ATTENTION_CONTRACTS
        rows = manifest['samples']['t2i']
        order = spec['image_order']
        items, noises = [], []
        for j in range(args.batch_size):
            idx = j % len(rows)
            seed = 42 + j // len(rows)
            items.append(build_t2i_item(tokenizer, model, rows[idx]['prompt'], idx, seed, order))
            noises.append(noise_for(idx, seed))
        batch = collate_imagenet_flow_cache(items, pad_to_length=512)
        noise = torch.stack(noises)
        spans = [(j, item['image_start'], item['image_start'] + 256) for j, item in enumerate(items)]
        props = torch.npu.get_device_properties(device)
        result['device_properties'] = str(props)
        result['device_total_memory_bytes'] = int(props.total_memory)
        result['resident_allocated_bytes'] = torch.npu.memory_allocated(device)

        probe_steps = 1 if args.model == 's2_single' and args.phase.startswith('refine-') else 10
        result['capacity_probe_steps'] = probe_steps
        result['warmup_steps'] = 1

        @torch.inference_mode()
        def generate(steps):
            return model.generate('t2i', input_ids=batch['input_ids'].to(device),
                token_types=batch['token_types'].to(device), sigma=batch['sigma'].to(device),
                spans=spans, image_latent_dim=16, initial_noise_bank=noise,
                flow_temperature=1., flow_cfg=result['cfg'], flow_cfg_schedule='constant',
                flow_solver='heun', flow_num_steps=steps, parallel_rate=1,
                order_strategy='spatial_halton', use_cache=use_cache, return_trace=True)

        with torch.inference_mode():
            # Both models keep the same warmed VAE resident, as in the gallery.
            preview = decode_latents(vae, torch.zeros(4, 16, 16, 16, device=device), .2325)
            del preview
            torch.npu.synchronize()
            torch.npu.empty_cache()
            torch.npu.reset_peak_memory_stats(device)
            times, peaks = [], []
            for repeat in range(args.warmup + args.repeats):
                torch.npu.synchronize()
                started = time.perf_counter()
                latents, trace = generate(1 if repeat < args.warmup else probe_steps)
                torch.npu.synchronize()
                duration = time.perf_counter() - started
                validate_generation_trace(trace, use_cache=use_cache, task='t2i')
                if latents.shape != (args.batch_size, 16, 16, 16) or not bool(torch.isfinite(latents).all()):
                    raise ValueError('invalid generated latent')
                if repeat >= args.warmup:
                    times.append(duration)
                peaks.append(dict(allocated_bytes=torch.npu.max_memory_allocated(device),
                                  reserved_bytes=torch.npu.max_memory_reserved(device)))
                if repeat == args.warmup + args.repeats - 1:
                    preview_latents = latents[:2].clone()
                del latents
                result.update(completed_calls=repeat + 1, generation_seconds=times,
                              memory_peaks=peaks)
                write(args.result, result)
            # Decode real outputs to verify that a completed trial produces images.
            preview = decode_latents(vae, preview_latents.float(), .2325)
            for j, img in enumerate(preview):
                save_png(img, args.result.parent / (args.result.stem + f'-sample-{j}.png'))
            result.update(status='ok', completed_at=utc(), generation_seconds=times,
                trace={k: v for k, v in trace.items() if v is None or isinstance(v, (str, int, float, bool))},
                mean_seconds_per_image=mean(times) / args.batch_size,
                median_seconds_per_image=median(times) / args.batch_size,
                median_images_per_second=args.batch_size / median(times),
                peak_allocated_bytes=max(p['allocated_bytes'] for p in peaks),
                peak_reserved_bytes=max(p['reserved_bytes'] for p in peaks))
        write(args.result, result)
    except Exception as error:
        message = str(error)
        oom = isinstance(error, torch.OutOfMemoryError) or 'out of memory' in message.lower()
        result.update(status='oom' if oom else 'error', completed_at=utc(),
                      error=message[-6000:], traceback=traceback.format_exc()[-8000:])
        write(args.result, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if not oom:
            raise
    print(json.dumps({k:result[k] for k in ('model','batch_size','device','phase','status')}), flush=True)


def run_wave(args, plans, phase, warmup=0, repeats=1):
    def run(plan):
        model, batch, device = plan
        path = args.root / phase / f'{model}-b{batch:04d}-d{device:02d}.json'
        if path.exists() and read(path).get('status') in ('ok', 'oom'):
            return read(path)
        cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--root', str(args.root),
               '--model', model, '--batch-size', str(batch), '--device', str(device),
               '--result', str(path), '--phase', phase, '--warmup', str(warmup), '--repeats', str(repeats)]
        path.parent.mkdir(parents=True, exist_ok=True)
        env = {k:v for k,v in os.environ.items() if k not in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE')}
        with path.with_suffix('.log').open('w') as log:
            proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        if not path.exists():
            raise RuntimeError(f'worker did not produce result: {path}')
        result = read(path)
        if result['status'] not in ('ok','oom'):
            raise RuntimeError(f'worker failed ({proc.returncode}): {path}')
        return result
    write(args.root/'progress.json', dict(phase=phase, plans=plans, updated_at=utc()))
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(run, plans))


def orchestrate(args):
    rows = {mid: [] for mid in MODELS}
    coarse = [8, 64, 128, 192, 256, 384, 512, 768]
    plans = [(mid, batch, m * 8 + i) for m, mid in enumerate(MODELS) for i, batch in enumerate(coarse)]
    for row in run_wave(args, plans, 'coarse'):
        rows[row['model']].append(row)
    for iteration in range(8):
        plans = []
        for m, mid in enumerate(MODELS):
            low, high = bounds(rows[mid])
            if high is None:
                batches = [low * k for k in (2, 3, 4, 6)]
            elif low == 0:
                batches = [1, 2, 4]
            else:
                batches = refine_batches(low, high)
            plans.extend((mid, batch, m * 8 + i) for i, batch in enumerate(batches))
        if not plans:
            break
        for row in run_wave(args, plans, f'refine-{iteration}'):
            rows[row['model']].append(row)
    else:
        raise RuntimeError('memory search did not converge')
    selected = {}
    for mid in MODELS:
        low, high = bounds(rows[mid])
        if not low or high is None or high - low > 8:
            raise RuntimeError('memory limit not bracketed')
        selected[mid] = dict(candidate_batch=low, first_oom_batch=high)
    write(args.root/'capacity_search.json', dict(complete=True, grain=8, models=selected))
    measured = {}
    # Full-node execution, both models measured on all 16 of the same devices.
    for mid in reversed(MODELS):
        batch = selected[mid]['candidate_batch']
        for attempt in range(8):
            trials = run_wave(args, [(mid,batch,d) for d in range(16)], f'measure-{mid}-b{batch}', warmup=1, repeats=3)
            if all(r['status']=='ok' for r in trials):
                measured[mid] = trials
                break
            batch -= 8
            if batch <= 0:
                raise RuntimeError('no stable batch across devices')
        else:
            raise RuntimeError('maximum-batch verification did not converge')
    write(args.root/'measurements.json', dict(complete=True, completed_at=utc(), models=measured,
        search=selected, protocol=read(args.root/'manifest.json')['capacity_benchmark']))
    write(args.root/'progress.json', dict(phase='complete', complete=True, updated_at=utc()))


def report(root):
    data=read(root/'measurements.json')
    if not data.get('complete'):
        raise ValueError('measurements incomplete')
    models, raw={}, []
    for mid, rows in data['models'].items():
        if len(rows)!=16 or {r['device'] for r in rows}!=set(range(16)):
            raise ValueError('expected all 16 devices')
        batches={r['batch_size'] for r in rows}
        if len(batches)!=1 or any(r['status']!='ok' or len(r['generation_seconds'])!=3 or r['warmup_batches']!=1 for r in rows):
            raise ValueError('inconsistent final measurements')
        batch=batches.pop()
        times=[t for r in rows for t in r['generation_seconds']]
        models[mid]=dict(batch_size_per_device=batch,
            search_first_oom_batch=data['search'][mid]['first_oom_batch'],
            cfg=rows[0]['cfg'], mean_seconds_per_image=mean(times)/batch,
            median_seconds_per_image=median(median(r['generation_seconds']) for r in rows)/batch,
            median_device_images_per_second=median(batch/median(r['generation_seconds']) for r in rows),
            median_batch_seconds=median(median(r['generation_seconds']) for r in rows),
            min_peak_allocated_gib=min(r['peak_allocated_bytes'] for r in rows)/GIB,
            max_peak_allocated_gib=max(r['peak_allocated_bytes'] for r in rows)/GIB,
            max_peak_reserved_gib=max(r['peak_reserved_bytes'] for r in rows)/GIB,
            device_total_memory_gib=rows[0]['device_total_memory_bytes']/GIB,
            cache_enabled=rows[0]['trace']['backbone_kv_cache_enabled'])
        for row in rows:
            for repeat, seconds in enumerate(row['generation_seconds']):
                raw.append(dict(model=mid,device=row['device'],repeat=repeat,batch_size=batch,
                    seconds=seconds,seconds_per_image=seconds/batch,images_per_second=batch/seconds,
                    peak_allocated_gib=row['peak_allocated_bytes']/GIB,
                    peak_reserved_gib=row['peak_reserved_bytes']/GIB))
    summary=dict(complete=True,updated_at=utc(),protocol=data['protocol'],models=models,
        s2_over_b_throughput=models['s2_single']['median_device_images_per_second']/models['b_x0']['median_device_images_per_second'])
    write(root/'summary.json',summary)
    with (root/'timings.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
    table=''.join(f'<tr><td>{mid}</td><td>{m["cfg"]}</td><td>{m["batch_size_per_device"]}</td>'
        f'<td>{m["min_peak_allocated_gib"]:.2f}–{m["max_peak_allocated_gib"]:.2f}</td>'
        f'<td>{m["max_peak_reserved_gib"]:.2f}</td><td>{m["mean_seconds_per_image"]:.4f}</td>'
        f'<td>{m["median_seconds_per_image"]:.4f}</td><td>{m["median_device_images_per_second"]:.3f}</td></tr>'
        for mid,m in models.items())
    trials=[]
    for p in sorted(root.glob('*/**/*.json')):
        if p.name.endswith('.tmp.json'):continue
        d=read(p)
        if 'batch_size' in d and 'device' in d and 'status' in d:
            trials.append((d,p))
    audit=''.join(f'<tr><td>{html.escape(d["phase"])}</td><td>{d["model"]}</td><td>{d["batch_size"]}</td>'
        f'<td>{d["device"]}</td><td>{d["status"]}</td><td><a href="{p.relative_to(root)}">JSON</a></td></tr>' for d,p in trials)
    (root/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>B / S2 显存容量与吞吐对照</title><style>body{{font:16px/1.65 system-ui;margin:32px;color:#17202a}}table{{border-collapse:collapse}}td,th{{padding:10px;border:1px solid #ccc}}code{{overflow-wrap:anywhere}}</style>
<h1>B / S2-single：接近显存上限的生成吞吐</h1>
<p>同一 16×910B 节点，每张卡独立生成；B CFG=2.0、S2 CFG=1.5，256×256、Heun 10、BF16，FP32 VAE 常驻。B 启用 KV cache，S2 使用原生整图 flow。复用固定 64 条提示，循环取样并改变种子。两者各自选择最大可稳定运行 batch，以 8 张为搜索粒度。</p>
<p>每个最终 batch 在全部 16 张卡上用同 batch、1 步 Heun 预热，再完整测量 10 步 Heun 三次，同步设备后计时。平均秒/图 = 48 次批次耗时之和 / (48 × batch)。这是批量生成折算，排除模型加载、VAE 解码和写盘；不是单图请求延迟或整机实测吞吐。实际分配峰值与 allocator 保留峰值分别报告，保留显存不等于有效计算占用。</p>
<table><tr><th>模型</th><th>CFG</th><th>每卡 batch</th><th>实际分配峰值 / GiB（各卡范围）</th><th>最大保留 / GiB</th><th>平均秒/图</th><th>中位秒/图</th><th>每卡图/秒（中位数）</th></tr>{table}</table>
<p>S2 / B 中位吞吐比：{summary['s2_over_b_throughput']:.3f}×。</p>
<p><a href="summary.json">汇总 JSON</a> · <a href="timings.csv">96 条正式计时 CSV</a> · <a href="measurements.json">完整正式测量</a> · <a href="capacity_search.json">容量边界</a></p>
<details><summary>全部探测记录与 OOM 边界</summary><p>coarse/refine 阶段用于确定容量，未预热，不能与正式速度直接比较。S2 的 refine 使用 1 步 Heun 筛查固定形状的整图 flow 显存峰值，最终以全部 16 卡的完整 10 步、三次测量验证容量。B 容量探测保留完整 256 token / Heun 10 步。</p><table><tr><th>阶段</th><th>模型</th><th>batch</th><th>设备</th><th>结果</th><th>原始文件</th></tr>{audit}</table></details></html>''')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['worker','run','report'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', choices=MODELS)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--phase', default='smoke')
    p.add_argument('--result', type=Path)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--repeats', type=int, default=3)
    a=p.parse_args()
    a.root=a.root.resolve()
    if a.action=='worker':
        if a.result is None or a.model is None:
            p.error('worker requires --model and --result')
        worker(a)
    elif a.action=='run':
        orchestrate(a)
    else:
        report(a.root)


if __name__=='__main__':
    main()
