// One joint model versus each task's independently trained only model.
const unifiedTaskLabels={understanding:'视觉理解 · I2T-only',generation:'图像生成 · T2I-only',text:'纯文本 · text-only'};
const unifiedOverviewKeys=new Set(['top1','top5','sugarcrepe','aro_vg_relation','aro_vg_attribution','coco_mr','flickr_mr','fid','is','text_macro']);
function unifiedTaskRows(study,task,cfg=3.5){return study.rows.filter(r=>r.group===task&&!r.aggregate&&(task!=='generation'||r.cfg===cfg))}
function unifiedMetric(study,key,cfg=3.5){return study.rows.find(r=>r.key===key&&(r.task!=='generation'||r.cfg===cfg))}
function unifiedDifference(row){
 if(row?.delta==null)return '—';
 return (row.delta>0?'+':'')+row.delta.toFixed(row.key==='fid'?3:2)+(row.baseline.percent?' pp':'');
}
function unifiedFindings(study,cfg=3.5){
 return Object.entries(unifiedTaskLabels).map(([task,label])=>{
  const rows=unifiedTaskRows(study,task,cfg),done=rows.filter(r=>r.only),wins=done.filter(r=>r.winner==='baseline').length;
  const complete=rows.length>0&&done.length===rows.length;
  let title=complete?`联合 B：${wins} / ${rows.length} 项更优`:'等待该任务的完整对照';
  let detail='已完成的单项结果如下；缺失分数保留空白。';
  if(complete&&task==='understanding'){
   title=wins===rows.length?'视觉理解：联合 B 全部更高':`视觉理解：联合 B ${wins} / ${rows.length} 项更高`;
   detail=`ImageNet Top-1 ${unifiedDifference(unifiedMetric(study,'top1'))}；SugarCrepe ${unifiedDifference(unifiedMetric(study,'sugarcrepe'))}。`;
  }
  if(complete&&task==='generation'){
   const fid=unifiedMetric(study,'fid',cfg),is=unifiedMetric(study,'is',cfg);
   title=fid.winner!==is.winner?(fid.winner==='baseline'?'图像生成：B 的 FID 更好，IS 较低':'图像生成：B 的 IS 更高，FID 较差'):`图像生成：联合 B ${wins} / ${rows.length} 项更优`;
   detail=`CFG=${cfg.toFixed(1)}，B − T2I-only：FID ${unifiedDifference(fid)}；IS ${unifiedDifference(is)}。`;
  }
  if(task==='generation'&&!complete)detail=`CFG=${cfg.toFixed(1)} 的 T2I-only 结果待完成；下方保留已完成 CFG 的分数。`;
  if(complete&&task==='text'){
   title=done.every(r=>r.winner==='only')?'纯文本：text-only 全部更高':`纯文本：联合 B ${wins} / ${rows.length} 项更高`;
   detail=`B − text-only：八任务均分 ${unifiedDifference(unifiedMetric(study,'text_macro'))}；MMLU ${unifiedDifference(unifiedMetric(study,'mmlu'))}。`;
  }
  return {task,label,title,detail,complete,wins,total:rows.length,tone:complete?(wins===rows.length?'gain':wins===0?'cost':'mixed'):'pending'};
 });
}
function unifiedTrainingOverview(){
 const study=D.unified_training_ablation;
 if(!study)return '';
 const rows=unifiedFindings(study).map(f=>[esc(f.label),esc(f.title),esc(f.detail)]);
 const sweep=study.generation_sweep;
 if(sweep){
  const row=rows[1],b=sweep.baseline,o=sweep.only;
  row[1]=`CFG sweep：${sweep.completed} / ${sweep.total} 档配对完成`;
  row[2]=b&&o?`各自已测最低 FID：B ${b.best_fid.fid.toFixed(3)}（CFG ${b.best_fid.cfg.toFixed(1)}）；T2I-only ${o.best_fid.fid.toFixed(3)}（CFG ${o.best_fid.cfg.toFixed(1)}）。`:'等待完整生成对照。';
 }
 return summaryCard('02 / 联合数据与单任务对照','Unified 训练消融','unified-training',
  summaryTable(['单任务参照','联合训练是否更好？','联合 B − only'],rows),
  `已完成 ${study.completed} / ${study.total} 组对应任务评测（生成要求 ${study.generation_cfg_values.length} 档 CFG）。按对应任务数据曝光量比较；生成固定 Heun=10，详情包含 FID / IS 折线图、同 CFG 对照和各自选优结果。`,
  '<a href="#unified-training">完整指标、数据曝光量与比较范围 →</a>',true);
}
function renderUnifiedCfgSweep(study){
 const sweep=study.generation_sweep;
 if(!sweep)return;
 const cfgs=study.generation_cfg_values;
 $('unified-cfg-protocol').textContent=`CFG ${cfgs[0].toFixed(1)}–${cfgs.at(-1).toFixed(1)}，共 ${cfgs.length} 档，与 B 使用相同扫描网格。每档 ImageNet-val 50K、Heun=10、Halton、seed 42、final EMA；T2I-only 已完成 ${sweep.completed} / ${sweep.total} 档。`;
 const plots=sweep.plots,version='?v='+encodeURIComponent(study.updated_at);
 $('unified-cfg-plots').innerHTML=['fid','is'].map(key=>plots[`cfg-${key}_svg`]?`<figure><img src="${esc(plots[`cfg-${key}_svg`]+version)}" alt="B 与 T2I-only 的 ${key.toUpperCase()} 随 CFG 变化折线图"><figcaption class="note">${key==='fid'?'FID 越低越好':'IS 越高越好；阴影为 split 标准差'}</figcaption></figure>`:'').join('')||'<p class="note">曲线等待更新；下表保留当前已完成结果。</p>';
 $('unified-cfg-downloads').innerHTML=[['cfg-sweep_png','PNG'],['cfg-sweep_pdf','PDF'],['cfg-sweep_svg','SVG']].filter(([key])=>plots[key]).map(([key,label])=>`<a href="${esc(plots[key]+version)}" download>下载折线图 ${label} →</a>`).join('')+`<a href="${esc(sweep.csv)}" download>下载逐点 CSV →</a>`;
 $('unified-cfg-tradeoff').innerHTML=plots['fid-is-tradeoff_svg']?`<img class="unified-tradeoff" src="${esc(plots['fid-is-tradeoff_svg']+version)}" alt="B 与 T2I-only 在不同 CFG 下的 FID 与 IS 权衡曲线"><p class="note">向右表示 IS 更高，向下表示 FID 更低；连线顺序为 CFG 递增。</p><div class="content-links"><a href="${esc(plots['fid-is-tradeoff_png']+version)}" download>下载 PNG →</a><a href="${esc(plots['fid-is-tradeoff_pdf']+version)}" download>下载 PDF →</a></div>`:'<p class="note">权衡曲线等待更新。</p>';
 const completeText=sweep.complete?'扫描范围内的完整结果':'当前已测点，扫描尚未完成';
 const wins=sweep.paired_wins;
 const sameCfg=`已配对的 ${sweep.completed} 档中，B 的 FID 有 ${wins.baseline.fid} 档更低，T2I-only 有 ${wins.only.fid} 档更低；B 的 IS 有 ${wins.baseline.is} 档更高，T2I-only 有 ${wins.only.is} 档更高。`;
 const bestText=['baseline','only'].filter(side=>sweep[side]).map(side=>{
  const value=sweep[side],label=side==='baseline'?'B':'T2I-only';
  return `${label} 最低 FID 为 ${value.best_fid.fid.toFixed(3)}（CFG ${value.best_fid.cfg.toFixed(1)}），最高 IS 为 ${value.best_is.is.toFixed(3)}（CFG ${value.best_is.cfg.toFixed(1)}）。`;
 }).join('');
 $('unified-cfg-conclusion').innerHTML=`<h3>${esc(completeText)}</h3><p>${esc(sameCfg)}</p><p>${esc(bestText)}</p>`;
 $('unified-cfg-best').innerHTML='<table aria-label="B 与 T2I-only 各自扫描 CFG 后的最优结果"><thead><tr><th>模型</th><th>完成档位</th><th>最低 FID ↓</th><th>对应 CFG</th><th>最高 IS ↑</th><th>对应 CFG</th></tr></thead><tbody>'+['baseline','only'].map(side=>{
  const value=sweep[side],label=side==='baseline'?'联合 B':'T2I-only';
  if(!value)return `<tr><td>${label}</td><td>0 / ${sweep.total}</td><td colspan="4">待评测</td></tr>`;
  const fid=value.best_fid,is=value.best_is;
  return `<tr><td>${label}</td><td>${value.completed} / ${value.total}</td><td class="metric">${metricLink({value:fid.fid,source:fid.source})}</td><td>${fid.cfg.toFixed(1)}</td><td class="metric">${metricLink({value:is.is,std:is.is_std,source:is.source})}</td><td>${is.cfg.toFixed(1)}</td></tr>`;
 }).join('')+'</tbody></table>';
}
function renderUnifiedTrainingMetrics(){
 const study=D.unified_training_ablation;
 if(!study)return;
 const group=$('unified-metric-group').value;
 const cfg=Number($('unified-cfg').value);
 $('unified-cfg').disabled=!['overview','generation'].includes(group);
 const rows=study.rows.filter(row=>(group==='overview'?unifiedOverviewKeys.has(row.key):row.group===group)&&(row.task!=='generation'||row.cfg===cfg));
 const notes={
  overview:'核心结果分别来自对应任务的 only 模型。百分数指标的差值单位为百分点（pp）；FID / IS 使用原始单位。',
  understanding:'ImageNet 分类、组合理解与跨数据集检索。mR 为图到文和文到图两个方向的 R@1、R@5、R@10 共六项平均。',
  generation:`ImageNet-val 50K；当前 CFG=${cfg.toFixed(1)}、Heun=10、Halton、seed 42，相同参考分布、精度与逐样本初始噪声。按相同 CFG 比较 B 与 T2I-only。`,
  text:'MMLU 为 5-shot，其余为 0-shot；两组的文本评分协议、样本数和归一化方式一致。八任务均分仅作内部摘要。',
  retrieval:'I2T 表示以图检索文本，T2I 表示以文本检索图像；这里是检索方向。COCO 为 5K test，Flickr30K 为 1K test。',
  diagnostics:'MMBench circular / SEED-Image 使用候选答案似然评分，作为内部消融诊断；不属于官方自由生成评测成绩。'
 };
 $('unified-metric-note').textContent=notes[group];
 $('unified-metrics').innerHTML='<table aria-label="联合 B 与对应 only 的任务指标比较"><thead><tr><th scope="col">任务 / 单任务参照</th><th scope="col">指标</th><th scope="col" class="metric">联合 B</th><th scope="col" class="metric">单任务 only</th><th scope="col" class="metric">差值 B − only</th><th scope="col">本项较优</th></tr></thead><tbody>'+rows.map(row=>{
  const winner=row.winner==='baseline'?'联合 B':row.winner==='only'?study.controls.find(c=>c.task===row.task).label:row.winner==='tie'?'持平':'待评测';
  const tone=row.winner==='baseline'?'unified-gain':row.winner==='only'?'unified-cost':'';
  return `<tr data-unified-metric="${esc(row.key)}"><td>${esc(unifiedTaskLabels[row.task])}</td><td>${esc(row.label)}${/[↑↓]/.test(row.label)?'':row.lower_is_better?' ↓':' ↑'}${row.cfg!==undefined?`<span class="status">CFG ${row.cfg.toFixed(1)}</span>`:''}${row.aggregate?'<span class="status">内部跨任务摘要</span>':''}</td><td class="metric${row.winner==='baseline'?' best':''}">${metricLink(row.baseline)}</td><td class="metric${row.winner==='only'?' best':''}">${metricLink(row.only)}</td><td class="metric ${tone}">${unifiedDifference(row)}</td><td class="${tone}">${esc(winner)}</td></tr>`;
 }).join('')+'</tbody></table>';
}
function renderUnifiedTrainingFindings(){
 const study=D.unified_training_ablation;
 if(!study)return;
 const cfg=Number($('unified-cfg').value),findings=unifiedFindings(study,cfg);
 $('unified-findings').innerHTML=findings.map(f=>`<article class="unified-finding ${f.tone}"><small>${esc(f.label)}</small><h3>${esc(f.title)}</h3><p>${esc(f.detail)}</p><span class="note">${f.complete?`联合 B 在 ${f.total} 个单项指标中有 ${f.wins} 项更优`:'该组评测待补齐'}</span></article>`).join('');
 const jointVisual=findings.find(f=>f.task==='understanding'),onlyText=findings.find(f=>f.task==='text');
 const visualGain=jointVisual?.complete&&jointVisual.wins===jointVisual.total;
 const textCost=onlyText?.complete&&unifiedTaskRows(study,'text').every(row=>row.winner==='only');
 $('unified-conclusion').innerHTML=`<h3>${visualGain&&textCost?'联合训练有视觉理解收益，也有纯文本性能代价':'联合训练的效果需要按任务判断'}</h3><p>${findings.map(f=>esc(f.title)+'。').join('')}生成结论对应当前 CFG=${cfg.toFixed(1)}；不同 CFG 分别比较。这些结果比较的是当前联合训练配方与单任务配方，不能概括为所有能力都提升。</p><p class="note">I2T-only 同时去掉 ClimbMix 和 T2I，因此视觉理解的提升还不能单独归因于生成任务；需要分别移除数据源的对照才能拆分贡献。</p>`;
}
function renderUnifiedTraining(){
 const study=D.unified_training_ablation;
 if(!study){$('unified-coverage').textContent='尚未收录联合训练对照。';return}
 const generation=study.controls.find(c=>c.task==='generation');
 $('unified-cfg').innerHTML=study.generation_cfg_values.map(cfg=>`<option value="${cfg.toFixed(1)}"${cfg===3.5?' selected':''}>${cfg.toFixed(1)}</option>`).join('');
 $('unified-coverage').textContent=`已完成 ${study.completed} / ${study.total} 组对应任务评测 · T2I-only 已完成 ${generation.completed_cfgs.length} / ${generation.cfg_total} 档 CFG。均使用各自训练结束时的 final EMA，训练步数与数据曝光量见下表。`;
 const models=[{...study.baseline,complete:true},...study.controls];
 $('unified-exposure').innerHTML='<table aria-label="训练任务及对应数据曝光量"><thead><tr><th scope="col">模型</th><th scope="col">ClimbMix 纯文本</th><th scope="col">ImageNet I2T</th><th scope="col">ImageNet T2I</th><th scope="col">训练步数</th><th scope="col">对应任务评测</th></tr></thead><tbody>'+models.map(model=>`<tr><td><strong>${esc(model.label)}</strong></td>${['text','i2t','t2i'].map(task=>{const n=model.physical_positions[task];return `<td title="${n.toLocaleString()} 个物理位置">${n?(n/1e9).toFixed(3):'<span class="muted">未使用</span>'}</td>`}).join('')}<td>${model.checkpoint_step.toLocaleString()}</td><td>${model.cfg_total?`${model.completed_cfgs.length} / ${model.cfg_total} 档 CFG 已完成`:model.complete?'已完成':'待完成'}</td></tr>`).join('')+'</tbody></table>';
 $('unified-cfg-matrix').innerHTML='<table aria-label="B 与 T2I-only 的完整 CFG 生成对照"><thead><tr><th scope="col">CFG</th><th scope="col">B · FID ↓</th><th scope="col">T2I-only · FID ↓</th><th scope="col">Δ FID</th><th scope="col">B · IS ↑</th><th scope="col">T2I-only · IS ↑</th><th scope="col">Δ IS</th><th scope="col">状态</th></tr></thead><tbody>'+study.generation_cfg_values.map(cfg=>{
  const fid=unifiedMetric(study,'fid',cfg),is=unifiedMetric(study,'is',cfg);
  const cells=[fid,is].map(row=>`<td class="metric${row.winner==='baseline'?' best':''}">${metricLink(row.baseline)}</td><td class="metric${row.winner==='only'?' best':''}">${metricLink(row.only)}</td><td class="metric">${unifiedDifference(row)}</td>`).join('');
  return `<tr data-unified-cfg="${cfg.toFixed(1)}"><th scope="row">${cfg.toFixed(1)}</th>${cells}<td>${fid.only&&is.only?'已完成':'待评测'}</td></tr>`;
 }).join('')+'</tbody></table>';
 $('unified-download').href=study.source;
 $('unified-metric-group').addEventListener('change',renderUnifiedTrainingMetrics);
 $('unified-cfg').addEventListener('change',()=>{renderUnifiedTrainingFindings();renderUnifiedTrainingMetrics()});
 renderUnifiedCfgSweep(study);
 renderUnifiedTrainingFindings();
 renderUnifiedTrainingMetrics();
}
