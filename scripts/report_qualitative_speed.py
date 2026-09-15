#!/usr/bin/env python3
"""Summarize paired, warmed per-device generation timings from a complete gallery."""
import argparse
import csv
import html
import json
import math
from pathlib import Path
from statistics import mean, median


def summarize(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    if not json.loads((root / 'COMPLETED.json').read_text()).get('complete'):
        raise ValueError('gallery must be complete')
    models, raw = {}, []
    contracts = {}
    for spec in manifest['models']:
        paths = sorted((root / 'models' / spec['id'] / 'timing').glob('*.json'))
        if not paths:
            continue
        rows = [json.loads(p.read_text()) for p in paths]
        if len(rows) != 16 or {r['rank'] for r in rows} != set(range(16)):
            raise ValueError('expected one full timed batch per each of 16 devices')
        contracts[spec['id']] = {r['rank']: r['samples'] for r in rows}
        device_medians = []
        for row, path in zip(rows, paths):
            times = row['generation_seconds']
            if row['batch_size'] != 8 or row['world_size'] != 16 or len(times) != 3:
                raise ValueError('expected batch 8, 16 devices, three timed repetitions')
            if row['timing_contract']['warmup_batches'] != 1:
                raise ValueError('missing warmup')
            if any(not math.isfinite(t) or t <= 0 for t in times):
                raise ValueError('invalid duration')
            if row['cfg'] != manifest['contract'].get('model_cfg', {}).get(spec['id'], manifest['contract']['cfg']):
                raise ValueError('CFG differs from frozen manifest')
            device_medians.append(median(times))
            for repeat, duration in enumerate(times):
                raw.append(dict(model=spec['id'], cfg=row['cfg'], rank=row['rank'], repeat=repeat,
                    batch_size=8, seconds=duration, images_per_second=8 / duration, source=str(path.relative_to(root))))
        models[spec['id']] = dict(label=spec['label'], cfg=rows[0]['cfg'], images=128,
            device_count=16, measured_batches=48, batch_size_per_device=8,
            median_batch_seconds=median(device_medians),
            mean_seconds_per_image=mean(t for r in rows for t in r['generation_seconds']) / 8,
            median_seconds_per_image=median(device_medians) / 8,
            mean_vae_decode_seconds_per_image=mean(r['vae_decode_seconds'] for r in rows) / 8,
            mean_png_save_seconds_per_image=mean(r['png_save_seconds'] for r in rows) / 8,
            median_device_images_per_second=median([8 / t for t in device_medians]),
            min_device_images_per_second=min(8 / t for t in device_medians),
            max_device_images_per_second=max(8 / t for t in device_medians),
            median_vae_decode_batch_seconds=median(r['vae_decode_seconds'] for r in rows),
            median_png_save_batch_seconds=median(r['png_save_seconds'] for r in rows),
            cache_enabled=rows[0]['trace']['backbone_kv_cache_enabled'])
    if set(models) != {'b_x0', 's2_single'} or contracts['b_x0'] != contracts['s2_single']:
        raise ValueError('expected exactly paired B and S2-single timings')
    b, s2 = models['b_x0'], models['s2_single']
    summary = dict(complete=True, models=models, protocol=manifest['contract']['timing'],
        s2_over_b_device_throughput=s2['median_device_images_per_second'] / b['median_device_images_per_second'],
        note=f'Per-device warmed throughput. B CFG={b["cfg"]}; S2 CFG={s2["cfg"]}. Heun 10, BF16, batch 8, same prompts/noise/hardware. VAE and PNG IO excluded. Not end-to-end cluster throughput.')
    (root / 'speed.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    with (root / 'speed.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(raw[0]))
        writer.writeheader()
        writer.writerows(raw)
    table = ''.join(f'<tr><td>{html.escape(m["label"])}</td><td>{m["cfg"]}</td>'
        f'<td>{m["mean_seconds_per_image"]:.4f}</td><td>{m["median_seconds_per_image"]:.4f}</td><td>{m["median_device_images_per_second"]:.3f}</td>'
        f'<td>{m["min_device_images_per_second"]:.3f}–{m["max_device_images_per_second"]:.3f}</td>'
        f'<td>{m["median_vae_decode_batch_seconds"]:.3f}</td><td>{m["median_png_save_batch_seconds"]:.3f}</td></tr>'
        for m in models.values())
    (root / 'speed.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>B / S2-single 生成速度</title><style>body{{font:16px/1.7 system-ui;margin:40px;max-width:1100px}}td,th{{border:1px solid #ccc;padding:12px}}table{{border-collapse:collapse}}</style>
<h1>B / S2-single 图像生成速度</h1><p>同一 16 卡 Ascend 910B Job，每卡 batch=8，256×256，Heun 10 步，BF16 模型、FP32 VAE。B CFG={b['cfg']}；S2-single CFG={s2['cfg']}。固定同提示和同噪声。</p>
<p>每模型每卡先预热 1 批，再测量同批 3 次；每次调用前后同步设备。平均秒/图 = 48 次批次耗时之和 ÷ (48 × 8)；中位秒/图先取每卡 3 次中位数，再取 16 卡中位数，除以 8。生成计时包含输入搬运，排除模型加载、VAE 和图片写盘。VAE 与写盘仅计保存样例的一次，不作为预热后的微基准。</p>
<table><tr><th>模型</th><th>CFG</th><th>平均秒/图</th><th>中位秒/图</th><th>每卡图/秒（中位数）</th><th>各卡图/秒范围</th><th>VAE / 8 张 / 秒</th><th>PNG 保存 / 8 张 / 秒</th></tr>{table}</table>
<p>S2 / B 每卡吞吐比：{summary['s2_over_b_device_throughput']:.3f}×。工作进程独立运行，未将各卡速度相加当作实测整机吞吐；这也不是单张请求延迟。</p>
<p><a href="speed.json">汇总 JSON</a> · <a href="speed.csv">全部 96 条原始计时 CSV</a> · <a href="index.html">定性样例</a></p></html>''')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(summarize(parser.parse_args().root), ensure_ascii=False, indent=2))
