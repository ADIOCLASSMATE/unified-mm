// Page-level navigation, fixed-protocol model metrics and the research abstract.
const metricGroups={
 overview:['fid','is','top1','text_macro','mmlu','mmbench_dev_en'],
 generation:['fid','is'],
 text:['arc_easy','arc_challenge','hellaswag','piqa','winogrande','boolq','openbookqa','mmlu','text_macro'],
 understanding:['top1','top5','sugarcrepe','aro_vg_relation','aro_vg_attribution','mmbench_dev_en','seed_bench_image'],
 diagnostics:['t2i_loss','i2t_loss','i2t_ppl']
};
for(const dataset of ['coco','flickr'])metricGroups[dataset]=['i2t','t2i'].flatMap(dir=>[1,5,10].map(k=>`${dataset}_${dir}_r${k}`));
const fmt=metric=>metric?(metric.percent?(metric.value*100).toFixed(2)+'%':metric.value.toFixed(3))+(metric.std===undefined?'':` ± ${metric.std.toFixed(2)}`):'—';
const smallerIsBetter=key=>key==='fid'||key.endsWith('loss')||key.endsWith('ppl');

function modelMetric(model,key,protocol='cfg2'){
 if(!['fid','is'].includes(key)||protocol==='cfg3p5')return model.metrics[key]??null;
 const entry=D.matrix_sweeps?.[0]?.models.find(m=>m.model===model.id);
 const result=entry?.strategies[entry.native_strategy];
 // Missing CFG=2 results must never silently fall back to CFG=3.5.
 if(result?.status!=='done'||!Number.isFinite(result[key]))return null;
 return {value:result[key],source:result.metrics_path,...(key==='is'?{std:result.is_std}:{})};
}
function metricLeader(models,key,protocol='cfg2'){
 const available=models.map(model=>({model,metric:modelMetric(model,key,protocol)})).filter(r=>r.metric);
 available.sort((a,b)=>(smallerIsBetter(key)?1:-1)*(a.metric.value-b.metric.value));
 return available[0]??null;
}
function metricLink(metric){return metric?`<a href="${esc(metric.source)}" title="查看原始结果">${fmt(metric)}</a>`:'<span class="missing">—</span>'}
function inModelGroup(model,group){
 return group==='all'||model.group===group||(group==='ablation'&&model.id==='b_x0');
}
function metricScope(scope){
 return D.models.filter(m=>scope==='formal'?(Object.keys(m.metrics).length>0||modelMetric(m,'fid')):inModelGroup(m,scope));
}

function renderMetrics(){
 const group=$('metric-group').value,protocol=$('metric-protocol').value,models=metricScope($('metric-scope').value),keys=metricGroups[group];
 const includesGeneration=keys.some(k=>['fid','is'].includes(k));
 $('metric-protocol').disabled=!includesGeneration;
 $('metric-protocol-note').textContent=includesGeneration?
  `当前 ${models.length} 个模型。FID / IS：CFG=${protocol==='cfg2'?'2.0':'3.5'}、Heun=10、ImageNet-val 50K；E 使用原生 Sequential，其余使用 Halton。文本与图像理解分数来自同一模型最终 checkpoint 的各自评测。`:
  `当前 ${models.length} 个模型。${group==='diagnostics'?'此处为最终评测诊断，不是训练过程中的验证曲线。':'此指标组使用自身的固定评测协议，不受 CFG / Heun 选择影响。'}`;
 const best=Object.fromEntries(keys.map(k=>[k,metricLeader(models,k,protocol)?.metric.value]));
 $('metrics').innerHTML='<table><thead><tr><th class="model-name">模型 / 完成情况</th>'+keys.map(k=>`<th class="metric">${esc(D.labels[k])}${['fid','is'].includes(k)?`<span class="status">CFG ${protocol==='cfg2'?'2.0':'3.5'}</span>`:''}</th>`).join('')+'</tr></thead><tbody>'+models.map(m=>{
  const order=m.id==='e_on_b'?'Sequential':'Halton';
  return `<tr data-model="${esc(m.id)}" class="${mainModelIds.has(m.id)?'focus-row':''}"><td class="model-name">${esc(m.label)}<span class="status">${esc(m.status)}${includesGeneration&&modelMetric(m,'fid',protocol)?' · '+order:''}</span></td>`+keys.map(k=>{
   const metric=modelMetric(m,k,protocol);
   return `<td data-metric="${esc(k)}" class="metric ${metric&&metric.value===best[k]?'best':''}">${metricLink(metric)}</td>`;
  }).join('')+'</tr>';
 }).join('')+'</tbody></table>';
 const highlights=group==='overview'?['fid','text_macro','top1']:group==='text'?['text_macro','mmlu','hellaswag']:keys.slice(0,3);
 $('metric-findings').innerHTML=highlights.map(key=>{
  const winner=metricLeader(models,key,protocol);
  return `<div class="metric-finding"><small>${esc(D.labels[key])} · ${smallerIsBetter(key)?'最低':'最高'}</small><strong>${winner?fmt(winner.metric):'—'}</strong><span class="note">${winner?esc(winner.model.label):'尚无有效记录'}</span></div>`;
 }).join('');
 const findings=highlights.map(key=>{
  const winner=metricLeader(models,key,protocol);
  return winner?`${esc(D.labels[key])} 的${smallerIsBetter(key)?'最低':'最高'}值为 ${metricLink(winner.metric)}（${esc(winner.model.label)}）`:null;
 }).filter(Boolean);
 $('model-metric-conclusion').innerHTML='<strong>当前范围的结果</strong><p>'+(findings.length?findings.join('；')+'。':'尚无足够的已完成记录。')+'</p>'+`<p class="note">${includesGeneration?'生成指标、文本能力与图像理解的领先模型可能不同。固定生成协议后的分数用于结构消融；参数变化带来的收益和代价见“采样与解码消融”。':'这些结果对应当前选中的模型范围。不同任务分别报告，不把跨任务平均值当作统一质量评分。'}</p>`;
}

function summaryTable(headers,rows){
 return '<div class="scroll"><table><thead><tr>'+headers.map(h=>`<th>${esc(h)}</th>`).join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(cell=>`<td>${cell}</td>`).join('')+'</tr>').join('')+'</tbody></table></div>';
}
function summaryCard(number,title,route,body,note,links='',wide=false){
 return `<article class="summary-card${wide?' wide':''}" data-summary="${route.split('/')[0]}"><div class="section-kicker">${number}</div><div class="card-heading"><h2>${title}</h2><a href="#${route}">查看详情 →</a></div>${body}<p class="summary-note">${note}</p>${links?`<div class="summary-links">${links}</div>`:''}</article>`;
}

function renderResearchOverview(){
 const models=metricScope('main'),matrix=D.matrix_sweeps?.[0],sweep=D.sampling_sweeps?.[0],order=D.order_sweeps?.[0],training=D.training;
 const fid=metricLeader(models,'fid'),text=metricLeader(models,'text_macro'),top1=metricLeader(models,'top1');
 const resultText=fid&&text&&top1?
  `固定 CFG=2.0、Heun=10 并保留各模型原生生成顺序后，${esc(fid.model.label)} 的 FID 最低（${fmt(fid.metric)}）；${esc(text.model.label)} 的文本八项均分最高（${fmt(text.metric)}），${esc(top1.model.label)} 的 ImageNet Top-1 最高（${fmt(top1.metric)}）。`:
  '已完成的结果按评测协议分别汇总，尚无有效分数的项目保留空白。';
 const samplingText=sweep?.conclusion&&order?.conclusion?
  `B 的采样扫描表明，最低 FID 与最高 IS 对应不同设置；固定 CFG=2.0、Heun=10 后，${esc(orderLabels[order.conclusion.best_fid.strategy])} 得到当前顺序实验中的最低 FID ${order.conclusion.best_fid.result.fid.toFixed(4)}。`:
  '采样参数与解码顺序实验单独汇总，避免将推理设置的收益混入模型结构排名。';
 $('research-abstract').innerHTML=`<div class="section-kicker">RESEARCH OVERVIEW / 研究摘要</div><h2>正式 A / B · 当前主线</h2><p>A 与 B 均使用 XT-query / X0-content 条件；A 的 content attention 严格排除自身，B 包含自身对角线。首页、指标、曲线与样例默认比较这两个模型。${resultText}</p><p>${samplingText}</p><p class="note">C–F 是在正式 B 上的消融，可按组查看；旧版 A/B 与单任务对照另列历史。全库共 ${training.run_count} 个训练实验、${D.formal_complete_models} 个完成评测的模型、${D.qualitative_records.toLocaleString()} 条定性输出。</p>`;

 const modelRows=models.map(m=>[
  `<span>${esc(m.label)}</span>`,metricLink(modelMetric(m,'fid')),metricLink(modelMetric(m,'is')),
  metricLink(modelMetric(m,'text_macro')),metricLink(modelMetric(m,'top1'))
 ]);
 const modelNote=`<strong>当前比较范围：正式 A / B。</strong> ${resultText}本表 FID / IS 使用 CFG=2.0、Heun=10、Halton。模型评测页可切换 B 上的 C–F 消融或历史实验，并查看文本八任务、图像理解、双向检索与历史 CFG=3.5 分数。`;

 const samplingRows=[];
 if(sweep){const c=sweep.conclusion;samplingRows.push([
  '<a href="#sampling/parameters">CFG / 采样步数</a>',`${sweep.completed} / ${sweep.total}`,
  c?`最低 FID <strong>${c.best_fid.result.fid.toFixed(4)}</strong><br><span class="note">CFG ${c.best_fid.cfg.toFixed(1)} · Heun ${c.best_fid.steps}</span><br>最高 IS <strong>${c.best_is.result.is.toFixed(2)}</strong><br><span class="note">CFG ${c.best_is.cfg.toFixed(1)} · Heun ${c.best_is.steps}</span>`:'尚无完整结论'
 ])}
 if(order){const c=order.conclusion;samplingRows.push([
  '<a href="#sampling/strategies">B 解码策略</a>',`${order.completed} / ${order.total}`,
  c?`最低 FID <strong>${c.best_fid.result.fid.toFixed(4)}</strong><br><span class="note">${esc(orderLabels[c.best_fid.strategy])} · 较 Halton 降低 ${c.fid_reduction.toFixed(4)}</span>`:'尚无完整结论'
 ])}
 if(matrix){const c=matrix.conclusion;samplingRows.push([
  '<a href="#sampling/cross-model">跨模型采样与换序</a>',`${matrix.completed} / ${matrix.total}`,
  c?`Stability：${c.improved_fid_models.length} / ${matrix.models.length} 个模型 FID 降低<br><span class="note">Random：${c.random_improved_fid_models?.length??0} / ${matrix.models.length} 个模型 FID 降低</span>`:'尚无完整结论'
 ])}
 const samplingNote=sweep?.conclusion&&matrix?.conclusion?
  '<strong>采样设置影响分数，也影响模型排名。</strong> FID 与 IS 的最优设置不同，换序收益因模型而异；E 的 Sequential 原生对照单列。解码探测的开销及 BF16 数值影响见细节，小幅差异不等于统计显著。':
  '各阶段分别报告完成情况与已测结果；实验完成后再给出该范围的最优结论。';

 const trainRows=Object.entries(training.groups).map(([group,label])=>{
  const runs=training.runs.filter(r=>r.group===group);
  return [esc(label),String(runs.length),`${runs.filter(r=>r.records).length} / ${runs.length}`,`${runs.filter(r=>r.validation.records).length} / ${runs.length}`];
 });
 const mainRuns=training.runs.filter(r=>r.group==='main');
 const pureValidation=training.runs.reduce((sum,r)=>sum+r.validation.available.climbmix,0);
 const trainNote=`<strong>${mainRuns.filter(r=>r.complete).length} / ${mainRuns.length} 个主线模型已到训练目标步数。</strong> 默认展示正式 A / B。全库共 ${training.records.toLocaleString()} 个训练记录、${training.validation_records.toLocaleString()} 个验证点；T2I、I2T 与纯文本分别查看。${pureValidation?`已收录 ${pureValidation} 个纯文本验证点。`:'纯文本验证代码已补充，当前快照尚无对应验证点。'}新验证协议与训练同口径；历史验证单独标注。`;

 const taskLabels={t2i:'T2I · 图像生成',i2t:'I2T · 图像描述',text:'纯文本续写'};
 const qualRows=['t2i','i2t','text'].map(task=>[
  `<a href="#qualitative/${task}">${taskLabels[task]}</a>`,
  `${D.samples[task].length} ${task==='text'?'个前缀':'个输入'}${task==='t2i'?' × 2 seed':task==='text'?' × 2 种解码':''}`,
  D.records.filter(r=>r.task===task).length.toLocaleString()
 ]);
 const qualitativeNote=`<strong>${D.qualitative_models} 个模型使用固定输入配对查看。</strong> 图像生成核对提示词遵循与结构，I2T 核对描述准确性，纯文本核对续写表现；所有输出保留。参考 caption 未进入 I2T 模型输入，单任务模型未训练的任务会标注。`;

 const provenance=D.data_provenance;
 const dataRows=[
  ['旧消融图像训练 / 验证','ImageNet-1K','1,281,167 / 50,000 张 · 256px'],
  ['I2T 训练 caption','Qwen3.6 + MiniMax-M3','每图 3 + 3 条'],
  ['T2I prompt / 验证 caption','gpt-5.6-luna',`每图 ${provenance.styles.length} 风格 / 验证固定描述`],
  ['当前大规模训练计划','已有 caption / prompt 优先；SII API 仅补缺','完整 ImageNet + 至少 200 万非 ImageNet 原图目标 · 512px'],
  ['两套共用纯文本','ClimbMix','100 个本地分片']
 ].map(row=>row.map(esc));
 const dataNote=`<strong>非 ImageNet 图库至少 200 万张，各来源配额为起点，合格图可超过；尚未完成。</strong> 当前优先确定来源，复用已有 caption / prompt，缺失或错误样本再由通过前置验收的 SII API 处理；不逐图调用 sol，仅在 SII 单图重试耗尽后由 Codex CLI 有界兜底。旧消融成绩、9 月 12 日首轮已验收产物与本次目标配额分别记录，ClimbMix 沿用现有语料。`;

 $('research-sections').innerHTML=
  summaryCard('01 / 当前主线','正式 A / B 评测','matrix',summaryTable(['模型','FID ↓ · CFG 2.0','IS ↑ · CFG 2.0','文本八项均分 ↑','ImageNet Top-1 ↑'],modelRows),modelNote,'<a href="#matrix">完整评测指标与历史消融 →</a><a href="#qualitative/t2i">同输入样例 →</a>',true)+
  summaryCard('02 / 推理时的实验变量','采样与解码消融','sampling/parameters',summaryTable(['实验','完成组数','主要结果'],samplingRows),samplingNote,'<a href="#sampling/parameters">CFG / 步数 →</a><a href="#sampling/strategies">解码策略 →</a><a href="#sampling/cross-model">跨模型换序 →</a>')+
  summaryCard('03 / 优化过程与验证覆盖','训练与验证 Loss','training',summaryTable(['实验组','模型数','有训练记录','有验证记录'],trainRows),trainNote,'<a href="#training">全部模型与任务曲线 →</a>')+
  summaryCard('04 / 同输入、同设置的输出对照','定性样例','qualitative/t2i',summaryTable(['任务','每模型输入','已保存输出'],qualRows),qualitativeNote,'<a href="#qualitative/t2i">图像生成 →</a><a href="#qualitative/i2t">图像描述 →</a><a href="#qualitative/text">文本续写 →</a>')+
  summaryCard('05 / 数据来源与复现依据','数据与协议','sources',summaryTable(['用途','来源','规模 / 约定'],dataRows),dataNote,'<a href="#sources">数据来源、合成模板与文件 →</a>');
}

function normalizeReportRoute(route){
 const aliases={sweep:'sampling/parameters',order:'sampling/strategies',sampling:'sampling/parameters',
  t2i:'qualitative/t2i',i2t:'qualitative/i2t',text:'qualitative/text',qualitative:'qualitative/t2i'};
 route=aliases[route]??route;
 return ['overview','matrix','training','sources','sampling/parameters','sampling/strategies','sampling/cross-model',
  'qualitative/t2i','qualitative/i2t','qualitative/text'].includes(route)?route:'overview';
}
function showTab(requested,{scroll=true}={}){
 const route=normalizeReportRoute(requested),[tab,section]=route.split('/');
 if(location.hash!=='#'+route)history.replaceState(null,'','#'+route);
 const previousTask=state.task;state.tab=tab;
 for(const panel of document.querySelectorAll('main > .panel'))panel.hidden=panel.id!==tab;
 for(const link of document.querySelectorAll('[data-nav]')){
  const active=link.dataset.nav===tab;link.classList.toggle('active',active);
  if(active)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');
 }
 for(const pane of document.querySelectorAll('[data-pane]'))pane.hidden=pane.dataset.pane!==section;
 for(const link of document.querySelectorAll('[data-sampling],[data-task]')){
  const active=(tab==='sampling'?link.dataset.sampling:tab==='qualitative'?link.dataset.task:null)===section&&Boolean(section);
  link.classList.toggle('active',active);if(active)link.setAttribute('aria-current','page');else link.removeAttribute('aria-current');
 }
 if(tab==='qualitative'){
  state.task=section;
  if(previousTask!==section||!state.galleryInitialized){state.page=0;setupTask();state.galleryInitialized=true}
  renderGallery();
 }
 if(tab==='training')renderLossCharts();
 if(scroll)$(tab).scrollIntoView({block:'start'});
}
