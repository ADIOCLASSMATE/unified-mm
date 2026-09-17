"""Render the architecture research narrative from final and in-training evidence."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

RUNS = {
    "S2-single · 100B": "unified-s2-single-0p6b-100b-imagenet-split-s42-r1",
    "S2-single": "unified-s2-single-0p6b-33b-imagenet-split-s42-r2",
    "S2-single · text two-stream": "unified-s2-single-text-two-stream-0p6b-33b-imagenet-split-s42-r2",
    "Z": "unified-z-0p6b-33b-imagenet-split-s42-r2",
}
COLORS = ["#a67535", "#6876b8", "#12655f"]
FIELDS = [("text_mean", "文本八项均分"), ("mmlu", "MMLU"), ("imagenet", "ImageNet Top-1"),
          ("aro_vg_relation", "ARO Relation"), ("sugarcrepe", "SugarCrepe")]


def render_forward_architecture() -> str:
    return (Path(__file__).parent / "assets/evaluation_forward.html").read_text(encoding="utf-8")


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def score(value):
    return '—' if value is None else f'{value * 100:.2f}%'


def link_score(point, key):
    return f'<a href="{escape(point["source"])}">{score(point.get(key))}</a>'


def collect(root):
    runs = {}
    for label, run in RUNS.items():
        points = []
        for path in (root / 'training-validation' / run).glob('downstream_validation/step-*/summary.json'):
            data = json.loads(path.read_text())
            if not data.get('complete') or data.get('weight_source') != 'ema':
                continue
            points.append({"step": data['step'], "source": path.relative_to(root).as_posix(),
                           **{key: data.get('text_mean') if key == 'text_mean' else data['tasks'].get(key, {}).get('primary') for key, _ in FIELDS}})
        runs[label] = sorted(points, key=lambda point: point['step'])
    return runs


def chart(runs, key, title, lo, hi, max_step):
    # A shared axis per panel; every measured point retains its source and exact value.
    def x(step):
        return 58 + step / max_step * 602

    def y(value):
        return 230 - (value * 100 - lo) / (hi - lo) * 200

    parts = [f'<div class="loss-chart"><h3>{escape(title)}</h3><svg viewBox="0 0 710 285" role="img" aria-label="{escape(title)}；完整数值见验证记录表"><title>{escape(title)}（百分比，越高越好）</title>']
    for tick in range(5):
        value = lo + (hi - lo) * tick / 4
        py = y(value / 100)
        parts.append(f'<path d="M58 {py}H660" stroke="#dce5e6"/><text x="48" y="{py + 4}" text-anchor="end" font-size="12" fill="#61747c">{value:g}</text>')
    for tick in range(4):
        step = round(max_step * tick / 3)
        px = x(step)
        parts.append(f'<text x="{px}" y="253" text-anchor="middle" font-size="12" fill="#61747c">{step:,}</text>')
    parts.append('<text x="660" y="278" text-anchor="end" font-size="12" fill="#61747c">optimizer step</text>')
    for index, (label, points) in enumerate(runs.items()):
        valid = [p for p in points if p.get(key) is not None]
        coords = ' '.join(f'{x(p["step"]):.2f},{y(p[key]):.2f}' for p in valid)
        color = COLORS[index % len(COLORS)]
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for p in valid:
            parts.append(f'<a href="{escape(p["source"])}"><circle cx="{x(p["step"]):.2f}" cy="{y(p[key]):.2f}" r="4" fill="{color}"><title>{escape(label)} · step {p["step"]:,} · {score(p[key])}；点击查看原始记录</title></circle></a>')
    parts.append('</svg><div class="content-links">')
    for index, label in enumerate(runs):
        parts.append(f'<span><i class="loss-dot" style="background:{COLORS[index % len(COLORS)]}"></i> {escape(label)}</span>')
    return ''.join(parts) + '</div></div>'


def render_research_status(root: Path, models: list[dict]) -> str:
    runs = collect(root)
    old = runs['S2-single · 100B']
    short = {label: points for label, points in runs.items() if label != 'S2-single · 100B'}
    shared = set.intersection(*(set(p['step'] for p in points) for points in short.values()))
    common = max(shared) if shared else None
    model_map = {m['id']: m for m in models}

    def final_table(ids, fields):
        rows = []
        for mid in ids:
            m = model_map.get(mid)
            if not m:
                continue
            row = [escape(m['label'])]
            for key, _ in fields:
                metric = m['metrics'].get(key)
                row.append(f'<a href="{escape(metric["source"])}">{score(metric["value"])}</a>' if metric else '—')
            rows.append(row)
        return table(['模型'] + [label + ' ↑' for _, label in fields], rows)

    def validation_table(series):
        return table(['模型', 'Step'] + [name + ' ↑' for _, name in FIELDS], [
            [escape(label), f'{p["step"]:,}'] + [link_score(p, key) for key, _ in FIELDS]
            for label, points in series.items() for p in points])

    paired = {label: [p for p in points if p['step'] == common] for label, points in short.items()}
    pair_note = '<p class="note">三组尚无共同完整验证点。</p>'
    if common:
        s2, two, z = (paired[label][0] for label in short)
        pair_note = f'''<p>在共同的 <strong>{common:,} 步</strong>，Z 的文本均分为 <strong>{score(z['text_mean'])}</strong>，比 S2-single 高 {(z['text_mean'] - s2['text_mean']) * 100:.2f} 个百分点；MMLU 为 {score(z['mmlu'])}，比 S2-single 高 {(z['mmlu'] - s2['mmlu']) * 100:.2f} 个百分点。ImageNet Top-1 为 <strong>{score(z['imagenet'])}</strong>（S2-single：{score(s2['imagenet'])}），SugarCrepe 为 {score(z['sugarcrepe'])}（S2-single：{score(s2['sugarcrepe'])}），图像理解已跟上 S2-single，部分指标更高；ARO Relation 为 {score(z['aro_vg_relation'])}，略低于 S2-single 的 {score(s2['aro_vg_relation'])}。</p>
<p>只消除文本 shift 的 two-stream 版本，文本均分为 {score(two['text_mean'])}，较 S2-single 提高 {(two['text_mean'] - s2['text_mean']) * 100:.2f} 个百分点；MMLU 为 {score(two['mmlu'])}，接近 S2-single 的 {score(s2['mmlu'])}。这支持同位置预测的方向，但仅改文本双流仍未消除 backbone 参与时间步迭代时的文本退化。</p>'''
    old_note = ''
    if old:
        first, last = old[0], old[-1]
        old_note = f'''<p>S2-single 在 {first['step']:,} → {last['step']:,} 步的 MMLU 从 <strong>{link_score(first, 'mmlu')} → {link_score(last, 'mmlu')}</strong>，现有验证点逐轮下降；文本八项均分从 {link_score(first, 'text_mean')} → {link_score(last, 'text_mean')}，整体下降、期间有小幅波动。图像能力上升的同时，纯文本能力没有被保住。</p>'''
    latest_note = ''
    if short['Z']:
        first, last = short['Z'][0], short['Z'][-1]
        latest_note = f'''<p class="note">Z 已记录至 {last['step']:,} 步：文本均分 {score(last['text_mean'])}、MMLU {score(last['mmlu'])}、ImageNet Top-1 {score(last['imagenet'])}。文本均分相对首个 {first['step']:,} 步验证点 {score(first['text_mean'])} 变化较小，但 MMLU 仍从 {score(first['mmlu'])} 降至 {score(last['mmlu'])}；“影响较小”不等于所有文本能力无损。不同训练进度的末点不用于三组直接对比。</p>'''
    return f'''<section class="panel" id="research-status" hidden>
<div class="section-head"><div><div class="section-kicker">RESEARCH STATUS / B → S2 → Z</div><h2>现状与架构结论</h2><p class="note">证据快照：2026-09-17 · 从旧版 100B 完整评测，到新一轮约 33B 配方的训练中 EMA 验证。百分比指标均为越高越好。</p></div><a class="back-overview" href="#overview">返回研究总览 ↑</a></div>
{render_forward_architecture()}
<div class="abstract"><h2>让图像双向可见，让 backbone 退出时间步迭代，让文本回到同位置预测</h2><p>训练 B、基于 B 的消融和 S2-single 后，我们看到 S2-single 的图像理解表现较好，由此判断：<strong>I2T 中的图像双向注意力值得保留</strong>。但 D、S2-single 让 backbone 随 noisy image 进入大循环，纯文本表现明显受损。因此开展 S2-single、S2-single 文本双流与 Z 的对照，希望一个 unified model 同时具备图像双向注意力、backbone 不进入大循环，以及消除文本一位 shift 的能力。</p><p><strong>当前 Z 的验证结果支持这一组合：</strong>文本能力受影响较小，图像理解已达到 S2-single 的水平。以下分别列出观察、消融和结论依据。</p></div>
<div class="box"><div class="section-kicker">01 / 最初的观察</div><h3>B 与基于 B 的消融 → S2-single：图像双向注意力是重要线索</h3><p>B 系列的消融帮助定位结构差异；S2-single 让图像块内部双向可见，在 COCO / Flickr30K 检索和内部视觉问答上优于 B，尤其 Flickr30K I2T R@1 从 38.30% 提高到 43.50%。这促使我们将图像双向注意力作为 I2T 的重要设计，而不是只关注图像生成端。</p>
{final_table(['b_x0', 'c_on_b', 'd_on_b', 'e_on_b', 'f_on_b', 's2_single'], [('top1', 'ImageNet Top-1'), ('coco_i2t_r1', 'COCO I2T R@1'), ('flickr_i2t_r1', 'Flickr I2T R@1'), ('mmbench_dev_en', 'MMBench'), ('seed_bench_image', 'SEED image')])}
<p class="note">此表为旧版 95,415-step final EMA，各列沿用完整评测协议；MMBench / SEED 为内部指标。S2-single 并非每项领先，例如 ImageNet Top-1 低于 B；这一步提供方向性证据，B 与 S2 还同时存在文本预测、flow 和 backbone 路径差异。</p></div>
<div class="box"><div class="section-kicker">02 / 代价与诊断</div><h3>D / S2-single：backbone 进入大循环，文本能力明显退化</h3><p>这里的“大循环”指图像生成的 flow / ODE 时间步迭代：D 使用随 x_t 更新的 backbone 条件；S2-single 每次随 noisy image 刷新 backbone。这个观察提示，应让去噪 head 承担时间步迭代，并让 backbone 提供可复用的条件。</p>{old_note}
{chart({'S2-single · 100B': old}, 'mmlu', '旧 S2-single · MMLU 逐轮下降', 20, 50, 90000)}
<h3 style="margin-top:24px">最终文本成绩：退化也出现在完整评测中</h3>
{final_table(['b_x0', 'd_on_b', 's2_single'], [('text_macro', '文本八项均分'), ('mmlu', 'MMLU')])}
<p class="note">D / S2-single 的 final MMLU 分别为 26.06% / 26.99%，均明显低于 B 的 35.36%。上方逐轮曲线来自 S2-single；现有归档未提供 D 同协议的训练中下游曲线，因此 D 在此仅列完整终点评测。</p>
<details><summary>旧 S2-single 的全部训练验证点与原始记录</summary>{validation_table({'S2-single · 100B': old})}</details></div>
<div class="box"><div class="section-kicker">03 / 针对性消融</div><h3>S2-single → S2-single two-stream → Z</h3><p>新一轮采用相同的 31,800-step（约 33B）预算，从基座 step 0 开始，每 3,180 步验证。以共同 step 比较文本保留和图像理解，避免将旧 100B 终点与新 33B 过程分数混排。</p>
{table(['架构', '图像块双向注意力', 'Backbone 进入时间步迭代', '文本预测与对角线'], [
['S2-single', '是', '是；随 noisy image 刷新', 'hidden[i−1] → token[i]，一位 shift'],
['S2-single · text two-stream', '是', '是；图像 flow 路径保持不变', 'query[i] → token[i]；文本 query 只看 j &lt; i'],
['Z', '是；同一图像共享 sigma', '否；一次 backbone，迭代复用固定条件', '同位置双流预测；query 不读目标 content']])}
<p class="note">“消除对角线”特指文本 query → content 的目标位置对角线，防止读取待预测 token；content 流仍保留自身，图像块内部仍双向可见。该设计消除 next-token hidden shift，不是删除所有 attention 对角线。Z 的 backbone 仍参与训练并接收梯度，只是不随去噪时间步反复前向。</p>
<div class="content-links"><a href="../../docs/S2_TEXT_TWO_STREAM.md">文本双流定义 →</a><a href="../../docs/Z_EXPERIMENT.md">Z 架构与固定条件 →</a><a href="../../docs/TRAINING.md#当前四项短预算消融">短预算训练设置 →</a></div></div>
<div class="box"><div class="section-kicker">04 / 当前验证结果</div><h3>Z 保住更多文本能力，同时跟上 S2-single 的图像理解</h3>{pair_note}
{validation_table(paired)}
<div class="loss-grid">{chart(short, 'text_mean', '文本八项均分 · Z 更稳定', 45, 57, 31800)}{chart(short, 'mmlu', 'MMLU · Z 的下降较缓', 25, 55, 31800)}{chart(short, 'imagenet', 'ImageNet Top-1 · Z 已跟上', 0, 60, 31800)}{chart(short, 'sugarcrepe', 'SugarCrepe · 保留图像理解', 55, 85, 31800)}</div>
<p class="note">折线显示全部已完成验证点，不外推尚无结果的区间；悬停查看数值，点击数据点打开原始 JSON。文本为完整八任务（34,507 题），ImageNet 为固定 2,000 图，ARO / SugarCrepe 各固定 512 条。训练中 grounding 使用校准协议；不能与上方完整评测直接作差。</p>{latest_note}
<details><summary>新一轮全部验证记录 · 检查每个 step 的五项指标</summary>{validation_table(short)}</details></div>
<div class="experiment-conclusion"><h3>当前判断：三个设计方向值得同时保留</h3><div class="unified-findings">
<div class="unified-finding gain"><strong>图像双向注意力是好的</strong><p>S2-single 的图像理解优势提供起点；Z 在保留双向图像可见性的同时，达到相近或更高的多项理解分数，支持在 unified model 中保留这一设计。</p></div>
<div class="unified-finding gain"><strong>消除文本一位 shift 是好的</strong><p>文本双流将预测放回同一位置，新一轮文本均分有小幅改善。现有证据支持这一方向，但改善有限，单独消除 shift 尚不足以解决文本退化。</p></div>
<div class="unified-finding gain"><strong>Backbone 不进入时间步迭代是好的</strong><p>D / S2-single 的文本退化与 Z 的较好保留共同支持：backbone 提供固定条件，去噪迭代交给 head，是当前更好的文本与图像能力折中。</p></div></div>
<p>因此，当前优先沿 <strong>Z：图像双向注意力 + 同位置文本预测 + backbone 不进入大循环</strong> 推进。已有结果支持这一架构组合；Z 相比 S2 还改变了条件和 head 路径，三项收益尚不能全部视为独立、严格的单因素因果证明。</p></div>
<div class="content-links"><a href="#matrix/showo2">旧 B / S2 完整评测 →</a><a href="#training">训练与验证 Loss →</a><a href="../../docs/TRAINING_DOWNSTREAM_VALIDATION.md">训练中下游验证协议 →</a></div>
</section>'''
