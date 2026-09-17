#!/usr/bin/env python3
"""Validate every frozen probe, summarize paired contrasts, and publish a gallery."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import html
import json
from pathlib import Path
import statistics

REPO=Path(__file__).resolve().parents[1]


def read(p):return json.loads(Path(p).read_text())


def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


def average(values):return statistics.mean(values) if values else None


def collect(root):
    from PIL import Image
    m=read(root/'manifest.json')
    scheduled={j['key']:j for j in m['schedule']}
    prompts={p['id']:p for p in m['prompts']}
    expected=set(scheduled)
    assert len(expected)==len(m['schedule'])
    records={}
    for p in sorted((root/'images').glob('*/*.json')):
        r=read(p);assert r['key'] not in records
        assert r['key'] in expected
        assert all(r[k]==v for k,v in scheduled[r['key']].items())
        assert all(r[k]==v for k,v in prompts[r['prompt_id']].items())
        assert r['noise_seed']==r['seed']+1000003*r['noise_index']
        assert r['text_tokens']+259<=512
        im_path=root/r['image']
        with Image.open(im_path) as im:
            assert im.size==(256,256)
            im.verify()
        records[r['key']]=r
    assert set(records)==expected,(len(records),len(expected))
    scores={}
    for rank in range(16):
        p=read(root/'progress'/f'rank-{rank:02}.json')
        assert p['rank']==rank and p['world_size']==16 and p['complete'] and not p['limited']
        s=read(root/'scores'/f'rank-{rank:02}.json')
        assert s['rank']==rank and s['world_size']==16 and s['complete'] and not s['limited']
        for row in s['rows']:
            assert row['key'] not in scores
            scores[row['key']]=row
    assert set(scores)==expected
    for spec in m['models']:
        for rank in range(16):
            l=read(root/'load_reports'/spec['id']/f'rank-{rank:02}.json')
            assert l['global_step']==spec['source']['global_step']
            if rank==0:assert l['full_checkpoint_value_check']['complete']
    text=read(root/'text_probes.json');assert text['complete'] and len(text['rows'])==7*(len(m['text_models'])+1)
    audit=read(root/'training_audit.json');assert audit['complete']
    for r in records.values():
        r.update(scores[r['key']])
    rows=list(records.values())
    grouped=defaultdict(list)
    for r in rows:
        grouped[r['model'],r['cfg'],r['family'],r['style'],r['condition']].append(r)
    aggregates=[]
    for (model,cfg,family,style,condition),rs in grouped.items():
        metric='red_minus_blue' if family=='color' else 'pixel_minus_photo' if style=='pixel' else 'watercolor_minus_photo'
        values=[r[metric] for r in rs]
        aggregates.append(dict(model=model,cfg=cfg,family=family,style=style,condition=condition,n=len(rs),
            metric=metric,mean=average(values),minimum=min(values),maximum=max(values),positive=sum(v>0 for v in values)))
    lookup={(r['model'],r['subject_id'],r['style'],r['condition'],r['cfg'],r['seed']):r for r in rows}
    contrasts=[]
    for spec in m['models']:
        mid=spec['id']
        for label,a,b in [('single_definition','definition','definition_swapped'),
                          ('single_examples','examples','examples_swapped'),
                          ('balanced_definition','binding_definition','binding_definition_swapped'),
                          ('balanced_examples','binding_examples','binding_examples_swapped')]:
            pairs=[]
            for sid,_,_ in m['subjects']:
                for seed in m['contract']['primary_seeds']:
                    ra=lookup[mid,sid,'pixel',a,2.,seed];rb=lookup[mid,sid,'pixel',b,2.,seed]
                    delta=(ra['cosine']['pixel']-ra['cosine']['watercolor'])-(rb['cosine']['pixel']-rb['cosine']['watercolor'])
                    win=lambda r,target:max(('photo','pixel','watercolor'),key=lambda k:r['cosine'][k])==target
                    pairs.append(dict(subject=sid,seed=seed,delta=delta,both_proxy_targets=win(ra,'pixel') and win(rb,'watercolor')))
            contrasts.append(dict(model=mid,contrast=label,n=len(pairs),mean_delta=average([p['delta'] for p in pairs]),
                positive=sum(p['delta']>0 for p in pairs),both_proxy_targets=sum(p['both_proxy_targets'] for p in pairs),pairs=pairs))
        for label,a,b in [('color_direct','red','blue'),('color_definition','definition_red','definition_blue'),('color_examples','examples_red','examples_blue')]:
            pairs=[]
            for seed in m['contract']['primary_seeds']:
                ra=lookup[mid,'color_teapot','color',a,2.,seed];rb=lookup[mid,'color_teapot','color',b,2.,seed]
                pairs.append(dict(seed=seed,delta=ra['red_minus_blue']-rb['red_minus_blue'],
                    both_proxy_targets=ra['red_minus_blue']>0 and rb['red_minus_blue']<0))
            contrasts.append(dict(model=mid,contrast=label,n=len(pairs),mean_delta=average([p['delta'] for p in pairs]),
                positive=sum(p['delta']>0 for p in pairs),both_proxy_targets=sum(p['both_proxy_targets'] for p in pairs),pairs=pairs))
    keep=['key','model','prompt_id','cfg','seed','id','subject_id','style','expected_style','condition','prompt','family',
          'image','noise_seed','text_tokens','cosine','pixel_minus_photo','watercolor_minus_photo','red_minus_blue']
    result=dict(schema='style_instruction_report_v1',complete=True,updated_at=datetime.now(timezone.utc).isoformat(),
        models=m['models'],conditions=m['conditions'],subjects=m['subjects'],contract=m['contract'],
        expected_images=len(expected),verified_images=len(rows),primary_images=sum(r['cfg']==2 for r in rows),
        sensitivity_images=sum(r['cfg']!=2 for r in rows),records=[{k:r[k] for k in keep} for r in rows],
        aggregates=aggregates,contrasts=contrasts,text_probes=text,training_audit=audit,
        assessment=read(root/'assessment.json') if (root/'assessment.json').exists() else None,
        scorer='google/siglip-so400m-patch14-384; normalized cosine, BF16. A diagnostic proxy, not a calibrated probability or human success rate.')
    return result


def plots(root,d):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions=['name','description','name_description','nonce','definition','examples','scrambled','filler','binding_definition','binding_examples']
    fig,axes=plt.subplots(1,2,figsize=(12,7),layout='constrained')
    models=[m['id'] for m in d['models']]
    labels=[m.get('plot_label',m['label']) for m in d['models']]
    for ax,style in zip(axes,['pixel','watercolor']):
        data=np.array([[next(r['mean'] for r in d['aggregates'] if r['model']==mid and r['condition']==c and r['style']==style and r['cfg']==2) for mid in models] for c in conditions])
        im=ax.imshow(data,cmap='RdBu_r',vmin=-.1,vmax=.1,aspect='auto')
        ax.set_xticks(range(len(models)),labels);ax.set_yticks(range(len(conditions)),conditions)
        ax.set_title(style+' vs photograph')
        for (i,j),v in np.ndenumerate(data):ax.text(j,i,f'{v:+.3f}',ha='center',va='center',color='white' if abs(v)>.065 else 'black',fontsize=9)
    fig.colorbar(im,ax=axes,label='Mean SigLIP cosine(style) - cosine(photo); 4 subjects x 4 seeds')
    fig.suptitle('Text-defined style probes | CFG 2, Heun 10 | proxy, not human judgment')
    for ext in ['png','svg','pdf']:fig.savefig(root/f'style-margin.{ext}',dpi=160)
    plt.close(fig)


def contact_sheets(root,d):
    from PIL import Image,ImageDraw,ImageFont
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    models=[m['id'] for m in d['models']]
    lookup={(r['model'],r['id'],r['cfg'],r['seed']):r for r in d['records']}
    for sid,_,_ in d['subjects']:
        for style in ['pixel','watercolor']:
            for seed in [42,43,44,45]:
                conditions=['photo','name','description','definition','examples','scrambled','binding_definition','binding_examples']
                sheet=Image.new('RGB',(len(models)*256+180,len(conditions)*278+36),'white');draw=ImageDraw.Draw(sheet)
                for j,mid in enumerate(models):draw.text((180+256*j,8),mid,fill='black',font=font)
                for i,c in enumerate(conditions):
                    pid=f'{sid}_photo' if c=='photo' else f'{sid}_{style}_{c}'
                    draw.text((4,40+278*i),f'{sid} / {style}\n{c}\nseed {seed}',fill='black',font=font)
                    for j,mid in enumerate(models):
                        r=lookup[mid,pid,2.,seed]
                        with Image.open(root/r['image']) as im:sheet.paste(im,(180+j*256,36+i*278))
                path=root/'contact_sheets'/f'{sid}-{style}-s{seed}.jpg';path.parent.mkdir(exist_ok=True)
                sheet.save(path,quality=91)
    # Every balanced color rule, without selecting seeds by output quality.
    conditions=['red','blue','nonce','definition_red','definition_blue','examples_red','examples_blue']
    for seed in [42,43,44,45]:
        sheet=Image.new('RGB',(len(models)*256+180,len(conditions)*278+36),'white');draw=ImageDraw.Draw(sheet)
        for j,mid in enumerate(models):draw.text((180+j*256,8),mid,fill='black',font=font)
        for i,c in enumerate(conditions):
            draw.text((4,40+i*278),c,fill='black',font=font)
            for j,mid in enumerate(models):
                r=lookup[mid,f'color_{c}',2.,seed]
                with Image.open(root/r['image']) as im:sheet.paste(im,(180+j*256,36+i*278))
        sheet.save(root/'contact_sheets'/f'color-s{seed}.jpg',quality=91)


def publish(root,d):
    write(root/'summary.json',d)
    with (root/'results.jsonl').open('w') as f:
        for r in d['records']:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (root/'aggregates.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(d['aggregates'][0]));w.writeheader();w.writerows(d['aggregates'])
    css=(REPO/'scripts/assets/evaluation_style_instruction.css').read_text()
    js=(REPO/'scripts/assets/evaluation_style_instruction.js').read_text()
    encoded=json.dumps(d,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Instruction 与风格泛化实验</title><style>body{{font:15px/1.65 system-ui,sans-serif;color:#203b3d;background:#f5f8f6;margin:0}}main{{max-width:1440px;margin:auto;padding:28px}}a{{color:#087966}}button,select{{font:inherit;padding:6px}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #dbe4df;padding:10px;text-align:left;vertical-align:top}}img{{max-width:100%}}details{{margin:12px 0}}summary{{cursor:pointer}}{css}</style><main><a href="../../index.html#style-instruction">← 返回评测首页</a><h1>Instruction 与风格泛化</h1><div id="style-report"></div></main><script type="application/json" id="style-data">{encoded}</script><script>{js}\nrenderStyleInstruction(JSON.parse(document.getElementById('style-data').textContent),document.getElementById('style-report'),'');</script></html>'''
    (root/'index.html').write_text(page)
    write(root/'COMPLETED.json',dict(complete=True,images=d['verified_images'],text_probes=len(d['text_probes']['rows']),
          exact_schedule_coverage=True,all_pngs_decoded=True,all_16_workers_complete=True,
          checkpoint_load_gate=True,score_coverage_exact=True,updated_at=d['updated_at']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--plots',action='store_true');parser.add_argument('--contact-sheets',action='store_true')
    a=parser.parse_args();d=collect(a.root)
    if a.plots:plots(a.root,d)
    if a.contact_sheets:contact_sheets(a.root,d)
    publish(a.root,d)
    print(json.dumps({k:d[k] for k in ['complete','verified_images','primary_images','sensitivity_images']}))
