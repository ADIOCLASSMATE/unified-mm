#!/usr/bin/env python3
"""Frozen paired inference probes; no finetuning, image editing or best-of selection."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SCHEMA = 'style_instruction_probe_v1'
SUBJECTS = [
    ('fox', 'a red fox sitting in snow', 'a red fox'),
    ('teapot', 'a white porcelain teapot on a wooden table', 'a teapot'),
    ('lighthouse', 'a white lighthouse above a blue ocean', 'a lighthouse'),
    ('cat', 'a striped tabby cat sitting on a blue cushion', 'a tabby cat'),
]
STYLES = {
    'pixel': ('pixel art', 'small square blocks on a visible coarse grid, crisp stair-step outlines, a limited palette of flat colors, and no smooth gradients'),
    'watercolor': ('watercolor art', 'translucent washes of pigment on textured white paper, soft bleeding edges, uneven brush strokes, and pale overlapping colors'),
}
CONDITIONS = {
    'name': '风格名称', 'description': '仅视觉特征', 'name_description': '名称 + 特征',
    'nonce': '新名称，无定义', 'definition': '新名称 + 定义',
    'definition_swapped': '新名称 + 反向定义', 'examples': '两例文本 ICL',
    'examples_swapped': '两例反向 ICL', 'scrambled': '同词打乱',
    'filler': '等长无关前文 + 名称', 'photo': '照片基线',
    'binding_definition': '双名称定义，词频平衡', 'binding_definition_swapped': '双名称定义，反向绑定',
    'binding_examples': '双名称两例 ICL，词频平衡', 'binding_examples_swapped': '双名称两例 ICL，反向绑定',
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def style_prompts(subject, style):
    name, desc = STYLES[style]
    opposite = 'watercolor' if style == 'pixel' else 'pixel'
    other_desc = STYLES[opposite][1]
    # The arbitrary label is identical for both mappings, avoiding semantic label cues.
    alias = 'dax'
    definition = f'In this task, {alias} means an image made of {desc}. Render {subject} in {alias} style.'
    words = definition.split()
    # Scramble context only and preserve the final subject request.
    context = f'In this task, {alias} means an image made of {desc}.'.split()
    random.Random(1701).shuffle(context)
    filler_words = ('This paragraph describes a catalog entry stored in a folder with a number and a date '
                    'The next entry is displayed after the previous entry and the list is sorted by its title').split()
    filler = ' '.join((filler_words * 4)[:len(words) - len(f'A {name} image of {subject}.'.split())])

    def examples(features):
        return (f'Example 1. Request: a rose in dax style. Image description: a rose depicted using {features}.\n'
                f'Example 2. Request: a sailboat in dax style. Image description: a sailboat depicted using {features}.\n'
                f'Now render {subject} in dax style.')

    def binding(first, second, fewshot=False):
        if fewshot:
            return (f'Example 1. Request: a rose in dax style. Description: a rose made of {first}.\n'
                    f'Example 2. Request: a sailboat in wug style. Description: a sailboat made of {second}.\n'
                    f'Now render {subject} in dax style.')
        return (f'In this task, dax means an image made of {first}. '
                f'Wug means an image made of {second}. Render {subject} in dax style.')

    return {
        'name': (f'A {name} image of {subject}.', style),
        'description': (f'An image of {subject}, made of {desc}.', style),
        'name_description': (f'A {name} image of {subject}, made of {desc}.', style),
        'nonce': (f'An image of {subject} in dax style.', None),
        'definition': (definition, style),
        'definition_swapped': (f'In this task, {alias} means an image made of {other_desc}. Render {subject} in {alias} style.', opposite),
        'examples': (examples(desc), style),
        'examples_swapped': (examples(other_desc), opposite),
        'scrambled': (' '.join(context) + f' Render {subject} in dax style.', style),
        'filler': (filler + f'. A {name} image of {subject}.', style),
        'binding_definition': (binding(desc,other_desc),style),
        'binding_definition_swapped': (binding(other_desc,desc),opposite),
        'binding_examples': (binding(desc,other_desc,True),style),
        'binding_examples_swapped': (binding(other_desc,desc,True),opposite),
    }


def prepare(args):
    from scripts.generate_unified_qualitative import model_inventory
    root = args.root.resolve()
    if (root / 'manifest.json').exists():
        raise FileExistsError('Frozen manifest already exists')
    inventory = {s['id']: s for s in model_inventory(REPO)}
    model_ids = ['b_x0', 's2_single', 'b_t2i_only_matched']
    text_ids = model_ids + ['b_i2t_only_matched', 'b_text_only']
    prompts = []
    for si, (sid, subject, scorer_subject) in enumerate(SUBJECTS):
        prompts.append(dict(id=f'{sid}_photo', subject_id=sid, subject=subject, scorer_subject=scorer_subject,
                            noise_index=si, style='photo', expected_style='photo', condition='photo',
                            prompt=f'A detailed color photograph of {subject}.', family='style'))
        for style in STYLES:
            for condition, (prompt, target) in style_prompts(subject, style).items():
                prompts.append(dict(id=f'{sid}_{style}_{condition}', subject_id=sid, subject=subject,
                                    scorer_subject=scorer_subject, noise_index=si, style=style,
                                    expected_style=target, condition=condition, prompt=prompt, family='style'))
    color_prompts = {
        'red': ('A red teapot on a plain white table.', 'red'),
        'blue': ('A blue teapot on a plain white table.', 'blue'),
        'nonce': ('A dax-colored teapot on a plain white table.', None),
        'definition_red': ('For this task, dax means red and wug means blue. A dax-colored teapot on a plain white table.', 'red'),
        'definition_blue': ('For this task, dax means blue and wug means red. A dax-colored teapot on a plain white table.', 'blue'),
        'examples_red': ('Examples: dax-colored ball = red ball; wug-colored cup = blue cup. Now show a dax-colored teapot on a plain white table.', 'red'),
        'examples_blue': ('Examples: dax-colored ball = blue ball; wug-colored cup = red cup. Now show a dax-colored teapot on a plain white table.', 'blue'),
    }
    for condition, (prompt, expected) in color_prompts.items():
        prompts.append(dict(id=f'color_{condition}', subject_id='color_teapot', subject='a teapot on a plain white table',
                            scorer_subject='a teapot', noise_index=4, style='color', expected_style=expected,
                            condition=condition, prompt=prompt, family='color'))
    schedule = []
    for spec in model_ids:
        for p in prompts:
            cfg_seeds = [(2., s) for s in [42, 43, 44, 45]]
            if p['family'] == 'style' and p['condition'] in ('photo', 'name', 'definition', 'examples'):
                cfg_seeds += [(c, s) for c in [1., 3.5] for s in [42, 43]]
            for cfg, seed in cfg_seeds:
                schedule.append(dict(model=spec, prompt_id=p['id'], cfg=cfg, seed=seed,
                                     key=f'{spec}/{p["id"]}-cfg{cfg:g}-s{seed}'))
    manifest = dict(schema=SCHEMA, created_at=now(), repo_root=str(REPO), models=[inventory[k] for k in model_ids],
                    text_models=[inventory[k] for k in text_ids], prompts=prompts, schedule=schedule,
                    conditions=CONDITIONS, subjects=SUBJECTS,
                    contract=dict(steps=10, solver='heun', primary_cfg=2., sensitivity_cfg=[1., 3.5],
                                  primary_seeds=[42, 43, 44, 45], sensitivity_seeds=[42, 43],
                                  noise='CPU FP32 seed + 1000003 * subject_index, shared across conditions/models/CFG',
                                  dtype='bfloat16', vae_dtype='float32', image_size=256, context=512,
                                  text_icl_only=True, image_demonstrations=False, runtime_hashing_enabled=False,
                                  all_outputs_retained=True, no_parameter_update=True,
                                  caveat='Final-checkpoint observational comparison; equal T2I exposure does not match total compute or task gradient fractions.'))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(REPO / 'public/models/Qwen--Qwen3-0.6B-Base'), local_files_only=True)
    for p in prompts:
        p['text_tokens'] = len(tokenizer.encode('Generate an image matching this description: ' + p['prompt'], add_special_tokens=False))
        if p['text_tokens'] + 259 > 512:
            raise ValueError('Prompt would exceed context: ' + p['id'])
    manifest['max_sequence_length'] = max(p['text_tokens'] + 259 for p in prompts)
    write(root / 'manifest.json', manifest)
    print(json.dumps(dict(prompts=len(prompts), images=len(schedule), max_sequence_length=manifest['max_sequence_length'])), flush=True)


def load_model(spec, device):
    import torch
    from omegaconf import OmegaConf
    from scripts.generate_unified_qualitative import BASE_CONFIG, check_loaded_values
    from utils.evaluation_model_source import configure_model_source, load_model_source_weights, resolve_evaluation_model_source
    from utils.utils import load_model_tokenizer
    cfg = OmegaConf.load(BASE_CONFIG)
    cfg.training.runtime_hashing_enabled = False
    source = resolve_evaluation_model_source(spec['checkpoint'])
    configure_model_source(cfg, source)
    model, tokenizer = load_model_tokenizer(cfg, model_dtype=torch.bfloat16)
    report = load_model_source_weights(model, source)
    if int(os.environ.get('RANK', '0')) == 0:
        report['full_checkpoint_value_check'] = check_loaded_values(model, spec['checkpoint'])
    model.to(device).eval()
    return model, tokenizer, report, cfg


def run(args):
    import gc
    import torch
    import torch_npu
    from utils.image_generation_io import build_t2i_item, noise_for, save_png, load_vae, decode_latents, T2I_PREFIX
    from utils.imagenet_flow_batching import collate_imagenet_flow_cache
    from scripts.generate_unified_qualitative import validate_generation_trace
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.set_num_threads(1)
    torch.npu.set_device(local_rank)
    device = torch.device('npu', local_rank)
    root, m = args.root.resolve(), read(args.root / 'manifest.json')
    prompts = {p['id']: p for p in m['prompts']}
    for spec in m['models']:
        if args.model and spec['id'] != args.model:
            continue
        work = [s for s in m['schedule'] if s['model'] == spec['id']][rank::world]
        if args.limit:
            work = work[:args.limit]
        model, tokenizer, report, cfg = load_model(spec, device)
        write(root / 'load_reports' / spec['id'] / f'rank-{rank:02}.json', report)
        cfg.experiment.validation_vae_module_root = 'public/code/mar'
        cfg.experiment.validation_vae_path = 'public/vae/mar-kl16/kl16.ckpt'
        vae = load_vae(cfg, device, 'fp32')
        use_cache = spec['id'] != 's2_single'
        for cfg_value in [2., 1., 3.5]:
            jobs = [s for s in work if s['cfg'] == cfg_value]
            for off in range(0, len(jobs), 8):
                subset = jobs[off:off + 8]
                rows = [prompts[j['prompt_id']] for j in subset]
                items = [build_t2i_item(tokenizer, model, p['prompt'], p['noise_index'], j['seed'], spec['image_order'])
                         for p, j in zip(rows, subset)]
                batch = collate_imagenet_flow_cache(items, pad_to_length=512)
                noise = torch.stack([noise_for(p['noise_index'], j['seed']) for p, j in zip(rows, subset)])
                start = time.monotonic()
                with torch.inference_mode():
                    latents, trace = model.generate('t2i', input_ids=batch['input_ids'].to(device),
                        token_types=batch['token_types'].to(device), sigma=batch['sigma'].to(device),
                        spans=[(b, item['image_start'], item['image_start'] + 256) for b, item in enumerate(items)],
                        image_latent_dim=16, initial_noise_bank=noise, flow_temperature=1., flow_cfg=cfg_value,
                        flow_cfg_schedule='constant', flow_solver='heun', flow_num_steps=10, parallel_rate=1,
                        order_strategy='spatial_halton', use_cache=use_cache, return_trace=True)
                    torch.npu.synchronize()
                    elapsed = time.monotonic() - start
                    validate_generation_trace(trace, use_cache=use_cache, task='t2i')
                    if not torch.isfinite(latents).all():
                        raise ValueError('Non-finite generation')
                    for begin in range(0, len(rows), 4):
                        decoded = decode_latents(vae, latents[begin:begin+4].float(), .2325)
                        for k, im in enumerate(decoded):
                            p, j = rows[begin+k], subset[begin+k]
                            image_path = root / 'images' / (j['key'] + '.png')
                            save_png(im, image_path)
                            write(image_path.with_suffix('.json'), dict(**j, **p,
                                image=str(image_path.relative_to(root)), serialized_prompt=f'{T2I_PREFIX} {p["prompt"]}',
                                noise_seed=j['seed'] + 1000003*p['noise_index'], generation_seconds=elapsed,
                                batch_size=len(rows), rank=rank, trace={k:v for k,v in trace.items() if v is None or isinstance(v,(str,int,float,bool))}))
                print(json.dumps(dict(rank=rank, model=spec['id'], cfg=cfg_value, completed=off+len(rows), total=len(jobs))), flush=True)
        del model, tokenizer, vae
        gc.collect()
        torch.npu.empty_cache()
    write(root / 'progress' / f'rank-{rank:02}.json', dict(rank=rank, world_size=world, complete=True,
          limited=bool(args.limit or args.model), finished_at=now(), torch=str(torch.__version__), torch_npu=str(torch_npu.__version__)))


def text_probes():
    probes = []
    for style, (name, desc) in STYLES.items():
        probes += [dict(id=f'{style}_knowledge', prompt=f'{name.capitalize()} is a visual style characterized by', expected=style),
                   dict(id=f'{style}_alias', prompt=f'In this task, dax means an image with {desc}. Therefore, a dax picture of a teapot should show', expected=style)]
    for target, other in [('red','blue'),('blue','red')]:
        probes.append(dict(id=f'color_{target}', prompt=f'In this task, dax means {target} and wug means {other}. The color of a dax teapot is', expected=target))
    probes.append(dict(id='pixel_fewshot', prompt='Request: a rose in dax style. Description: a rose made of crisp square blocks with a limited color palette.\nRequest: a sailboat in dax style. Description: a sailboat made of crisp square blocks with a limited color palette.\nRequest: a teapot in dax style. Description:', expected='pixel'))
    return probes


def run_text(args):
    import gc
    import torch
    import torch_npu
    from scripts.generate_unified_qualitative import decode_suffix
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(1)
    torch.npu.set_device(0)
    device = torch.device('npu', 0)
    root, m = args.root.resolve(), read(args.root / 'manifest.json')
    specs = m['text_models'] + [dict(id='qwen_base', checkpoint=str(REPO/'public/models/Qwen--Qwen3-0.6B-Base'))]
    results = []
    for spec in specs:
        if spec['id'] == 'qwen_base':
            tokenizer = AutoTokenizer.from_pretrained(spec['checkpoint'], local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(spec['checkpoint'], dtype=torch.bfloat16,
                    attn_implementation='eager', local_files_only=True).to(device).eval()
        else:
            model, tokenizer, report, cfg = load_model(spec, device)
            write(root/'text_load_reports'/f'{spec["id"]}.json', report)
        for p in text_probes():
            ids = torch.tensor([tokenizer.encode(p['prompt'], add_special_tokens=False)], device=device)
            stops = [tokenizer.eos_token_id]
            if getattr(model.config, 'im_end_token_id', None) is not None:
                stops.append(model.config.im_end_token_id)
            with torch.inference_mode():
                if spec['id'] == 'qwen_base':
                    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=80,
                        do_sample=False, eos_token_id=stops, pad_token_id=tokenizer.eos_token_id)
                else:
                    out = model.generate('text', input_ids=ids, max_new_tokens=80, temperature=0.,
                        eos_token_id=stops, use_cache=spec['id'] != 's2_single')
            row = dict(model=spec['id'], **p, **decode_suffix(tokenizer,out[0,ids.shape[1]:].cpu().tolist(),stops))
            results.append(row)
            write(root/'text_probes.json', dict(complete=False, rows=results))
        del model, tokenizer
        gc.collect()
        torch.npu.empty_cache()
    write(root/'text_probes.json', dict(complete=True, rows=results, protocol='80 tokens, greedy continuation, no chat template', finished_at=now()))


def audit(args):
    from utils.imagenet_synthetic_text_index import ImageNetSyntheticTextIndex
    index = ImageNetSyntheticTextIndex(REPO/'public/datasets/imagenet1k_synthetic_v1/indexed/train/manifest.json')
    chosen = random.Random(1609).sample(range(index.row_count), 1024)
    counts, examples = Counter(), []
    for idx in chosen:
        row = index.read_t2i(idx)
        variants = row['model_result']['prompts']
        counts.update(p['style'] for p in variants)
        if len(examples) < 8:
            examples.append(dict(manifest_index=idx, image_id=row['image_id'], prompts=variants,
                                 visual_description=row['model_result'].get('visual_description')))
    index.close()
    write(args.root/'training_audit.json', dict(complete=True, population=index.row_count, sample=1024, seed=1609,
        style_counts=dict(counts), examples=examples, scope='Deterministic sample, not exhaustive corpus scan',
        source='public/datasets/imagenet1k_synthetic_v1/indexed/train/manifest.json'))


def score(args):
    """Independent SigLIP similarity is a diagnostic proxy, never a success label."""
    import torch
    import torch_npu
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    torch.set_num_threads(1)
    rank, world = int(os.environ.get('RANK', '0')), int(os.environ.get('WORLD_SIZE', '1'))
    device = torch.device('npu', int(os.environ.get('LOCAL_RANK', '0')))
    torch.npu.set_device(device)
    root, m = args.root.resolve(), read(args.root/'manifest.json')
    checkpoint = str(REPO/'public/models/google--siglip-so400m-patch14-384')
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModel.from_pretrained(checkpoint, dtype=torch.bfloat16, attn_implementation='eager', local_files_only=True).to(device).eval()
    texts, text_keys = [], []
    for sid, subject, scorer_subject in SUBJECTS + [('color_teapot', 'a teapot', 'a teapot')]:
        for label, text in [('photo', f'A detailed color photograph of {scorer_subject}.'),
                            ('pixel', f'A pixel art illustration of {scorer_subject}, with a coarse grid of square pixels and a limited palette.'),
                            ('watercolor', f'A watercolor painting of {scorer_subject}, with translucent brush strokes on white paper.'),
                            ('red', f'A red {scorer_subject.removeprefix("a ")}.'),
                            ('blue', f'A blue {scorer_subject.removeprefix("a ")}.')]:
            text_keys.append((sid,label));texts.append(text)
    with torch.inference_mode():
        batch=processor(text=texts,padding='max_length',truncation=True,return_tensors='pt')
        tf=model.get_text_features(**{k:v.to(device) for k,v in batch.items()})
        tf=getattr(tf,'pooler_output',tf).float()
        tf=tf/tf.norm(dim=-1,keepdim=True)
        rows=m['schedule'][rank::world]
        if args.limit:rows=rows[:args.limit]
        scores=[]
        for off in range(0,len(rows),16):
            selected=rows[off:off+16]
            records=[read(root/'images'/(s['key']+'.json')) for s in selected]
            images=[]
            for r in records:
                with Image.open(root/r['image']) as im:images.append(im.convert('RGB'))
            inputs=processor(images=images,return_tensors='pt')
            vf=model.get_image_features(pixel_values=inputs['pixel_values'].to(device,dtype=torch.bfloat16))
            vf=getattr(vf,'pooler_output',vf).float()
            vf=vf/vf.norm(dim=-1,keepdim=True)
            cosine=(vf@tf.T).cpu().tolist()
            for r, values in zip(records,cosine):
                sims={label:values[i] for i,(sid,label) in enumerate(text_keys) if sid==r['subject_id']}
                scores.append(dict(key=r['key'], cosine=sims, pixel_minus_photo=sims['pixel']-sims['photo'],
                    watercolor_minus_photo=sims['watercolor']-sims['photo'], red_minus_blue=sims['red']-sims['blue']))
            print(json.dumps(dict(scoring_rank=rank,completed=off+len(selected),total=len(rows))),flush=True)
    write(root/'scores'/f'rank-{rank:02}.json', dict(complete=True,rank=rank,world_size=world,limited=bool(args.limit),
        checkpoint=checkpoint,texts=[dict(subject=s,label=l,text=t) for (s,l),t in zip(text_keys,texts)],
        metric='Cosine similarity; not calibrated probability or human judgment',rows=scores))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','text','audit','score'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--model',default='')
    args=parser.parse_args()
    {'prepare':prepare,'run':run,'text':run_text,'audit':audit,'score':score}[args.action](args)


if __name__=='__main__':
    main()
