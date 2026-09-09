const T=D.training;
const lossState={selected:new Set(T.runs.filter(r=>r.group==='main').map(r=>r.id))};
const lossTitles={total:'总 Loss · 已加权',t2i:'T2I · 图像生成',i2t:'I2T · 图像描述',climbmix:'纯文本 · ClimbMix'};
const lossValue=v=>v===null||v===undefined?'—':Number(v).toPrecision(6);
const lossNumber=v=>Math.abs(v)>=1000?Math.round(v).toLocaleString():Number(v.toPrecision(4)).toLocaleString(undefined,{maximumFractionDigits:5});
const lossSplitLabel={train:'训练',validation:'验证'};
const lossIndependence={may_have_been_seen_in_training:'可能见过训练文本',source_rows_excluded_since_training_start:'训练已排除这些源记录',external_data_declared_independent:'独立外部文本（配置声明）'};
const isUnifiedLoss=detail=>detail?.loss_protocol?.name==='unified_schedule_microbatch_mean_v1';
const lossProtocolLabel=detail=>isUnifiedLoss(detail)?'与训练同口径 · 当前权重':detail?.loss_protocol?.name==='legacy_text_only'?'历史独立纯文本口径':'历史图文口径';

function lossSeries(run,column,windowSize,axis,split='train'){
 const history=split==='validation'?run.validation:run;
 if(split==='validation')windowSize=1;
 const values=[],queue=[];let sum=0,lastStep=null;
 for(const [index,row] of history.points.entries()){
  const step=row[0],detail=history.details?.[index],previous=history.details?.[index-1];
  const selectedProtocol=$('loss-val-protocol').value;
  const included=split!=='validation'||selectedProtocol==='all'||(selectedProtocol==='unified')===isUnifiedLoss(detail);
  const raw=included?row[column]:null;
  const protocolChange=previous&&(previous.validation_seed!==detail.validation_seed||JSON.stringify(previous.target_counts)!==JSON.stringify(detail.target_counts)||JSON.stringify(previous.loss_protocol)!==JSON.stringify(detail.loss_protocol)||JSON.stringify(previous.imagenet_subset)!==JSON.stringify(detail.imagenet_subset)||previous.model_weights!==detail.model_weights);
  const gap=lastStep!==null&&(step-lastStep>history.log_every*1.5||Boolean(protocolChange));
  if(raw===null||gap){queue.length=0;sum=0}
  if(raw!==null){queue.push(raw);sum+=raw;if(queue.length>windowSize)sum-=queue.shift()}
  const x=axis==='progress'?(run.target_steps?100*step/run.target_steps:null):step;
  values.push({x,step,raw,value:raw===null?null:sum/queue.length,gap,detail});lastStep=step;
 }
 return values;
}

// Keep first/last and each bucket's extrema. Nulls and log gaps form separate
// paths, so decimation cannot invent continuity or hide a short isolated run.
function lossSegments(points,budget=1000){
 const segments=[];let segment=[];
 for(const p of points){
  if(p.gap||p.value===null){if(segment.length)segments.push(segment);segment=[]}
  if(p.value!==null)segment.push(p);
 }
 if(segment.length)segments.push(segment);
 return segments.map(rows=>{
  if(rows.length<=budget)return rows;
  const stride=Math.ceil(rows.length/(budget/4)),picked=[];
  for(let i=0;i<rows.length;i+=stride){
   const end=Math.min(rows.length,i+stride);let lo=i,hi=i;
   for(let j=i+1;j<end;j++){if(rows[j].value<rows[lo].value)lo=j;if(rows[j].value>rows[hi].value)hi=j}
   for(const j of [...new Set([i,lo,hi,end-1])].sort((a,b)=>a-b))picked.push(rows[j]);
  }
  return picked;
 });
}

function renderLossSelectors(){
 $('loss-selectors').innerHTML=T.runs.map(r=>`<label class="chip" title="${esc(T.groups[r.group])} · ${esc(r.status)}"><input type="checkbox" value="${esc(r.id)}" ${lossState.selected.has(r.id)?'checked':''}><span class="loss-dot" style="background:${r.color}"></span>${esc(r.label)}${!r.records?' · 待日志':''}</label>`).join('');
 $('loss-selectors').querySelectorAll('input').forEach(el=>el.addEventListener('change',()=>{el.checked?lossState.selected.add(el.value):lossState.selected.delete(el.value);updateLossPresetButtons();renderLossCharts();renderLossInventory()}));
 updateLossPresetButtons();
}
function updateLossPresetButtons(){
 $('loss-presets').querySelectorAll('button').forEach(el=>{
  const runs=T.runs.filter(r=>inModelGroup(r,el.dataset.group));
  el.classList.toggle('active',runs.length===lossState.selected.size&&runs.every(r=>lossState.selected.has(r.id)));
 });
}

function renderLossInventory(){
 const headers=['模型 / 日志来源','组别','状态 / 已到步数','末次记录 / 目标步数','总 loss','T2I','I2T','纯文本'];
 const cells=T.runs.map(r=>{
  const last=r.points.at(-1),link=r.source?`<a href="${esc(r.source)}">${esc(r.label)}</a>`:esc(r.label);
  const notes=[r.partial_tail?'末行正在写入，留待刷新':null,r.replaced_points?`${r.replaced_points} 个旧点被续训记录替换`:null,r.nonfinite_values?`${r.nonfinite_values} 个非有限值`:null,r.gaps?`${r.gaps} 处记录缺口`:null].filter(Boolean).join('；');
  return `<tr class="${lossState.selected.has(r.id)?'':'loss-unselected'}"><td class="model-name"><span class="loss-dot" style="background:${r.color}"></span> ${link}<span class="status">${r.records.toLocaleString()} 个记录点${r.config?` · <a href="${esc(r.config)}">训练配置</a>`:''}</span><span class="status">${esc(r.schedule.join(' → '))}</span>${notes?`<span class="status">${esc(notes)}</span>`:''}</td><td>${esc(T.groups[r.group])}</td><td>${esc(r.status)}<span class="status">${r.reached_step.toLocaleString()} 步</span></td><td>${r.last_logged_step?.toLocaleString()??'—'} / ${r.target_steps?.toLocaleString()??'未记录'}</td>`+[1,2,3,4].map(i=>`<td class="metric">${lossValue(last?.[i])}</td>`).join('')+'</tr>';
 }).join('');
 $('loss-inventory').innerHTML='<table><thead><tr>'+headers.map(h=>`<th>${h}</th>`).join('')+'</tr></thead><tbody>'+cells+'</tbody></table>';
 const valHeaders=['模型 / 验证来源','验证记录 / 范围','验证协议与覆盖','末次验证步数','总 loss','T2I','I2T','纯文本'];
 const valCells=T.runs.map(r=>{
  const v=r.validation,last=v.points.at(-1),detail=v.details.at(-1),counts=detail?.target_counts;
  const link=detail?`<a href="${esc(detail.source)}">${esc(r.label)}</a>`:esc(r.label);
  const coverage=counts?`<strong>${lossProtocolLabel(detail)}</strong><br>图像 ${counts.image?.toLocaleString()??'—'} / caption ${counts.caption?.toLocaleString()??'—'} / 纯文本 ${counts.climbmix?.toLocaleString()??'—'} targets`:esc(v.missing_reason);
  const pureText=detail?.pure_text_source?`<span class="status"><a href="${esc(detail.pure_text_source)}">纯文本验证 JSON</a> · ${esc(lossIndependence[detail.pure_text_independence]??'未注明独立性')}</span>`:'';
  const notes=[v.gaps?`${v.gaps} 处验证缺口`:null,v.incomplete_files.length?`${v.incomplete_files.length} 个文件正在写入`:null,v.nonfinite_values?`${v.nonfinite_values} 个非有限值`:null].filter(Boolean).join('；');
  return `<tr class="${lossState.selected.has(r.id)?'':'loss-unselected'}"><td class="model-name"><span class="loss-dot" style="background:${r.color}"></span> ${link}<span class="status">${esc(T.groups[r.group])}${r.config?` · <a href="${esc(r.config)}">训练配置</a>`:''}</span></td><td>${v.records.toLocaleString()} 个验证点<span class="status">${v.first_logged_step?.toLocaleString()??'—'} → ${v.last_logged_step?.toLocaleString()??'—'}</span>${notes?`<span class="status">${esc(notes)}</span>`:''}</td><td>${coverage}${detail?`<span class="status">${esc(detail.model_weights==='current'?'当前训练权重':v.model_weights)} · seed ${detail.validation_seed??'未记录'}</span><span class="status">${isUnifiedLoss(detail)?'固定全局 microbatch，按训练任务比例汇总':detail.pure_text_source===detail.source?'固定纯文本记录':v.validation_max_batches===0?'全部 batch':v.validation_max_batches?`最多 ${v.validation_max_batches} batch / rank`:'batch 上限未记录'} · 间隔配置 ${v.log_every.toLocaleString()} 步</span>`:''}${pureText}</td><td>${v.last_logged_step?.toLocaleString()??'—'}</td>`+[1,2,3,4].map(i=>`<td class="metric">${lossValue(last?.[i])}</td>`).join('')+'</tr>';
 }).join('');
 $('loss-validation-inventory').innerHTML='<table><thead><tr>'+valHeaders.map(h=>`<th>${h}</th>`).join('')+'</tr></thead><tbody>'+valCells+'</tbody></table>';
}

function renderLossCharts(){
 const runs=T.runs.filter(r=>lossState.selected.has(r.id)),mode=$('loss-mode').value,split=$('loss-split').value,windowSize=Number($('loss-smooth').value),axis=$('loss-axis').value,log=$('loss-log').checked;
 $('loss-smooth').disabled=split==='validation';
 const pureTextCount=T.runs.reduce((sum,r)=>sum+r.validation.available.climbmix,0);
 $('loss-climbmix-status').textContent=pureTextCount?`已记录 ${pureTextCount.toLocaleString()} 个纯文本验证点；各实验是否排除验证源记录见下表。`:'当前快照尚无纯文本验证 loss。进程加载新版代码并执行三任务验证后，重新生成报告即可收录。';
 const unifiedCount=T.runs.reduce((sum,r)=>sum+(r.validation.unified_records??0),0);
 $('loss-protocol-status').textContent=`新协议 ${unifiedCount.toLocaleString()} 个验证点：任务组成、模型权重、microbatch 平均和调度比例与训练一致。其余 ${T.validation_records-unifiedCount} 个为历史记录，总 loss 不与新协议直接比较；协议切换处断开。`;
 $('loss-val-protocol').disabled=split==='train';
 $('loss-coverage').textContent=`${T.run_count} 个实验：训练 ${T.runs_with_history} 个 / ${T.records.toLocaleString()} 个记录点；验证 ${T.runs_with_validation} 个 / ${T.validation_records.toLocaleString()} 个记录点。当前选中 ${runs.length} 个。快照：${T.updated_at.replace('T',' ').replace('+00:00',' UTC')}。`;
 $('loss-charts').innerHTML=Object.entries(lossTitles).map(([key,title])=>`<div class="loss-chart" id="loss-chart-${key}"><div class="chart-head"><h3>${title}</h3><button class="save-loss-svg" data-key="${key}" aria-label="下载 ${title} SVG">SVG ↓</button></div><p class="note"></p><div class="loss-svg"></div><div class="loss-tooltip" hidden></div></div>`).join('');
 for(const [key,title] of Object.entries(lossTitles)){
  const column=T.columns.indexOf(key==='total'||mode==='raw'?key:'weighted_'+key);
  const chart=$('loss-chart-'+key),unit=key==='total'?'训练 step_loss / 验证 val/loss（新协议同口径；历史记录另标）':mode==='weighted'?'加权贡献（新协议含训练调度比例；历史口径另标）':key==='t2i'?'Flow velocity MSE':'有效目标 token 平均 CE';
  let count=0;
  const splits=split==='both'?['train','validation']:[split];
  const series=runs.flatMap(run=>splits.map(kind=>({run,split:kind,interval:kind==='validation'?run.validation.log_every:run.log_every,points:lossSeries(run,column,windowSize,axis,kind)})));
  let xmax=0;for(const s of series)for(const p of s.points)if(p.x!==null)xmax=Math.max(xmax,p.x);
  const xmin=$('loss-min').value===''?0:Number($('loss-min').value);
  xmax=$('loss-max').value===''?Math.max(xmax,1e-6):Number($('loss-max').value);
  const displayNote=splits.map(kind=>kind==='validation'?'验证原始点':windowSize===1?'训练原始点':`训练 ${windowSize} 点均值`).join(' · ');
  chart.querySelector('.note').textContent=unit+' · '+displayNote;
  const area=chart.querySelector('.loss-svg');
  if(xmax<=xmin){area.innerHTML='<p class="empty">横轴终点必须大于起点。</p>';continue}
  let low=Infinity,high=-Infinity;
  for(const s of series){
   s.points=s.points.filter(p=>p.x!==null&&p.x>=xmin&&p.x<=xmax).map(p=>log&&p.value!==null&&p.value<=0?{...p,value:null}:p);
   for(const p of s.points)if(p.value!==null){low=Math.min(low,p.value);high=Math.max(high,p.value);count++}
  }
  const trainCount=series.filter(s=>s.split==='train'&&s.points.some(p=>p.value!==null)).length,valCount=series.filter(s=>s.split==='validation'&&s.points.some(p=>p.value!==null)).length;
  if(splits.includes('validation')&&!valCount)chart.querySelector('.note').textContent+=' · 当前选择与范围内无验证 loss';
  if(!count){area.innerHTML=`<p class="empty">${key==='climbmix'&&split==='validation'?'尚未记录 ClimbMix 纯文本验证 loss；val/loss_text 属于 I2T。':'所选模型在此任务与范围内没有 '+(split==='both'?'训练或验证':lossSplitLabel[split])+' loss 记录。'}</p>`;chart.querySelector('button').disabled=true;continue}
  const W=720,H=350,L=66,R=18,U=18,B=48,pw=W-L-R,ph=H-U-B,transform=v=>log?Math.log10(v):v;
  let lo=transform(low),hi=transform(high),padding=(hi-lo||Math.abs(hi)*.1||.1)*.07;lo-=padding;hi+=padding;
  if(!log&&low>=0)lo=Math.max(0,lo);
  const px=x=>L+(x-xmin)/(xmax-xmin)*pw,py=y=>U+(hi-transform(y))/(hi-lo)*ph;
  let svg=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${title} 训练与验证随步数变化"><title>${esc(title)} · ${unit} · ${displayNote} · 训练 ${trainCount} / 验证 ${valCount} 个模型</title><rect width="${W}" height="${H}" fill="white"/>`;
  for(let i=0;i<=5;i++){
   const y=U+ph*i/5,v=hi-(hi-lo)*i/5,x=L+pw*i/5;
   svg+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="#e3ebea"/><text x="${L-9}" y="${y+4}" text-anchor="end" font-size="11" fill="#61747c">${lossNumber(log?10**v:v)}</text><text x="${x}" y="${H-23}" text-anchor="middle" font-size="11" fill="#61747c">${lossNumber(xmin+(xmax-xmin)*i/5)}</text>`;
  }
  svg+=`<line x1="${L}" y1="${U}" x2="${L}" y2="${H-B}" stroke="#a9bbb8"/><text x="${W/2}" y="${H-3}" text-anchor="middle" font-size="11" fill="#61747c">${axis==='step'?'Optimizer step':'各自目标步数完成比例 (%)'}</text><defs><clipPath id="loss-clip-${key}"><rect x="${L}" y="${U}" width="${pw}" height="${ph}"/></clipPath></defs><g clip-path="url(#loss-clip-${key})">`;
  for(const s of series){
   svg+=`<g class="loss-series loss-${s.split}" data-model="${esc(s.run.id)}">`;
   for(const segment of lossSegments(s.points)){
    if(segment.length>1)svg+=`<path d="${segment.map((p,i)=>(i?'L':'M')+px(p.x).toFixed(2)+','+py(p.value).toFixed(2)).join(' ')}" fill="none" stroke="${s.run.color}" stroke-width="${s.split==='validation'?1.8:1.5}" ${s.split==='validation'?`stroke-dasharray="${isUnifiedLoss(segment[0].detail)?'6 4':'2 4'}"`:''} stroke-linejoin="round"/>`;
    if(s.split==='validation'||segment.length===1)for(const p of segment)svg+=`<circle cx="${px(p.x)}" cy="${py(p.value)}" r="${s.split==='validation'?2.4:3}" fill="${s.split==='validation'?'white':s.run.color}" stroke="${s.run.color}" stroke-width="1.4"/>`;
   }
   svg+='</g>';
  }
  svg+='</g><line class="loss-crosshair" y1="'+U+'" y2="'+(H-B)+'" stroke="#61747c" stroke-dasharray="3 3" visibility="hidden"/></svg>';
  area.innerHTML=svg;
  const element=area.querySelector('svg'),tooltip=chart.querySelector('.loss-tooltip'),cross=element.querySelector('.loss-crosshair');
  element.addEventListener('pointermove',event=>{
   const rect=element.getBoundingClientRect(),xsvg=(event.clientX-rect.left)/rect.width*W;
   if(xsvg<L||xsvg>W-R){tooltip.hidden=true;cross.setAttribute('visibility','hidden');return}
   const x=xmin+(xsvg-L)/pw*(xmax-xmin),items=[];
   for(const s of series){
    const points=s.points;if(!points.length)continue;
    let a=0,b=points.length;while(a<b){const mid=(a+b)>>1;if(points[mid].x<x)a=mid+1;else b=mid}
    const options=[points[a],points[a-1]].filter(Boolean),p=options.sort((a,b)=>Math.abs(a.x-x)-Math.abs(b.x-x))[0];
    const tolerance=axis==='step'?s.interval*.6:100*s.interval*.6/(s.run.target_steps||1);
    if(p?.value!==null&&p&&Math.abs(p.x-x)<=tolerance)items.push({run:s.run,split:s.split,p});
   }
   cross.setAttribute('x1',xsvg);cross.setAttribute('x2',xsvg);cross.setAttribute('visibility','visible');
   const smoothed=split!=='validation'&&windowSize>1;
   tooltip.hidden=false;tooltip.innerHTML=`<strong>${title}</strong><table><thead><tr><td>模型 · 类型 · 实际步数</td><td>原始</td>${smoothed?'<td>训练平滑</td>':''}</tr></thead><tbody>`+items.slice(0,12).map(({run,split:kind,p})=>`<tr><td><span class="loss-dot" style="background:${run.color}"></span> ${esc(run.label)} · ${lossSplitLabel[kind]} · ${p.step.toLocaleString()}${kind==='validation'?`<span class="status">${lossProtocolLabel(p.detail)}</span>`:''}${kind==='validation'&&key==='climbmix'?`<span class="status">${esc(lossIndependence[p.detail?.pure_text_independence]??'未注明独立性')}</span>`:''}</td><td>${lossValue(p.raw)}</td>${smoothed?`<td>${kind==='train'?lossValue(p.value):'—'}</td>`:''}</tr>`).join('')+`</tbody></table>${items.length>12?`另有 ${items.length-12} 条曲线；筛选后可查看各自数值。`:items.length?'':'该位置无邻近日志点'}`;
   tooltip.style.left='12px';tooltip.style.top='66px';
  });
  element.addEventListener('pointerleave',()=>{tooltip.hidden=true;cross.setAttribute('visibility','hidden')});
  chart.querySelector('button').addEventListener('click',()=>{
   const clone=element.cloneNode(true),ns='http://www.w3.org/2000/svg',legend=series.flatMap(s=>s.split==='train'?[s]:[true,false].map(unified=>({...s,unified,points:s.points.filter(p=>isUnifiedLoss(p.detail)===unified)}))).filter(s=>s.points.some(p=>p.value!==null));
   clone.querySelector('.loss-crosshair').remove();clone.setAttribute('viewBox',`0 0 ${W} ${H+45+legend.length*20}`);
   const add=(name,attrs,text)=>{const el=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(text)el.textContent=text;clone.appendChild(el)};
   add('text',{x:L,y:H+20,'font-size':11,fill:'#172c35'},`${title} · ${displayNote} · ${T.updated_at}`);
   legend.forEach((s,i)=>{
    add('line',{x1:L,y1:H+40+i*20,x2:L+20,y2:H+40+i*20,stroke:s.run.color,'stroke-width':2,'stroke-dasharray':s.split==='validation'?(s.unified?'6 4':'2 4'):'none'});
    if(s.split==='validation')add('circle',{cx:L+10,cy:H+40+i*20,r:2.4,fill:'white',stroke:s.run.color});
    const note=key==='climbmix'&&s.split==='validation'&&s.points.some(p=>p.detail?.pure_text_independence==='may_have_been_seen_in_training')?'（可能见过训练文本）':'';
    add('text',{x:L+28,y:H+44+i*20,'font-size':12,fill:'#172c35'},`${s.run.label} · ${s.split==='train'?'训练':s.unified?'验证 · 与训练同口径':'验证 · 历史协议'}${note}`);
   });
   const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)],{type:'image/svg+xml'})),a=document.createElement('a');a.href=url;a.download=`loss-${split}-${key}.svg`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
 }
}

function initializeLoss(){
 $('loss-presets').innerHTML=[['all','全部模型'],...Object.entries(T.groups),['none','清空']].map(([group,label])=>`<button data-group="${group}">${esc(label)}</button>`).join('');
 $('loss-presets').querySelectorAll('button').forEach(el=>el.addEventListener('click',()=>{lossState.selected=new Set(T.runs.filter(r=>inModelGroup(r,el.dataset.group)).map(r=>r.id));$('loss-min').value='';$('loss-max').value='';renderLossSelectors();renderLossCharts();renderLossInventory()}));
 for(const id of ['loss-split','loss-mode','loss-smooth','loss-axis','loss-log','loss-min','loss-max','loss-val-protocol'])$(id).addEventListener('change',()=>{if(id==='loss-axis'){$('loss-min').value='';$('loss-max').value=''}renderLossCharts()});
 $('loss-early').addEventListener('click',()=>{$('loss-axis').value='step';$('loss-min').value='0';$('loss-max').value='1000';renderLossCharts()});
 $('loss-reset').addEventListener('click',()=>{$('loss-min').value='';$('loss-max').value='';renderLossCharts()});
 $('loss-downloads').innerHTML=`<a href="${esc(T.csv)}" download>训练原始记录 CSV</a><a href="${esc(T.validation_csv)}" download>验证原始记录 CSV</a><a href="${esc(T.json)}" download>训练 + 验证完整 JSON</a>`+(T.plots?.artifacts??[]).map(p=>`<a href="${esc(p.path)}" download>${esc(p.label)} ${p.format.toUpperCase()}</a>`).join('');
 $('loss-plot-time').textContent=T.plots?`独立 PNG / SVG 导出快照：${T.plots.updated_at}。实线为训练 20 点均值，虚线与圆点为验证原始值；CSV / JSON 保留每个原始点。静态图用 --plots 刷新。`:'CSV / JSON 保留训练与验证的每个原始点。可用 --plots 生成独立 PNG / SVG 图。';
 renderLossSelectors();renderLossInventory();renderLossCharts();
}
