/* Fixed-protocol final-EMA results across head capacities. Null scores stay blank. */
function flowHeadScaleSummary(study){
 const b=study.rows.filter(r=>r.family==='b').sort((a,b)=>a.depth-b.depth);
 const cfg2=b.map(r=>r.results.find(s=>s.cfg===2));
 return b.length>1&&cfg2.every(s=>s?.source)?
  `B · CFG 2.0：FID <strong>${cfg2[0].fid.toFixed(4)} → ${cfg2.at(-1).fid.toFixed(4)}</strong><br><span class="note">depth ${b[0].depth} → ${b.at(-1).depth}；F 扩展档 ${study.rows.filter(r=>r.family==='f'&&r.depth>8).reduce((n,r)=>n+r.completed,0)} 组已评测</span>`:
  '各规模按相同 CFG 比较；缺失分数留空';
}

function renderFlowHeadCfgSweep(){
 const study=D.flow_head_scale,sweep=study?.cfg_sweep;
 if(!sweep)return;
 const family=$('flow-scale-family').value,rows=study.rows.filter(row=>row.family===family);
 const cfgs=study.cfg_values,version='?v='+encodeURIComponent(study.updated_at);
 $('flow-scale-cfg-protocol').textContent=`CFG ${cfgs[0].toFixed(1)}–${cfgs.at(-1).toFixed(1)}，共 ${cfgs.length} 档；与 B 基准扫描网格对齐。ImageNet-val 50K、final EMA、Heun=10、Halton、seed 42。${rows.map(row=>`depth ${row.depth}：${row.completed}/${cfgs.length} 档`).join('；')}。`;
 const plots=sweep.plots[family]??{};
 $('flow-scale-cfg-plots').innerHTML=['fid','is'].map(key=>plots[`cfg-${key}_svg`]?`<figure><img src="${esc(plots[`cfg-${key}_svg`]+version)}" alt="${family.toUpperCase()} 各 head 规模的 ${key.toUpperCase()} 随 CFG 变化曲线"><figcaption class="note">${key==='fid'?'FID 越低越好':'IS 越高越好；阴影为 split 标准差'}</figcaption></figure>`:'').join('')||'<p class="note">曲线待评测结果更新；已完成分数见下表。</p>';
 $('flow-scale-cfg-downloads').innerHTML=[['cfg-sweep_png','PNG'],['cfg-sweep_pdf','PDF'],['cfg-sweep_svg','SVG']].filter(([key])=>plots[key]).map(([key,label])=>`<a href="${esc(plots[key]+version)}" download>下载折线图 ${label} →</a>`).join('')+`<a href="${esc(sweep.csv)}" download>下载逐点 CSV →</a>`;
 $('flow-scale-cfg-best').innerHTML='<table aria-label="各 head 规模在已测 CFG 中的最优结果"><thead><tr><th>模型</th><th>已测档位</th><th>最低 FID ↓</th><th>对应 CFG</th><th>同档 IS ↑</th><th>最高 IS ↑</th><th>对应 CFG</th></tr></thead><tbody>'+sweep.best.filter(row=>row.family===family).map(row=>{
  const label=`${family.toUpperCase()} depth ${row.depth}`,fid=row.best_fid,is=row.best_is;
  if(!fid)return `<tr><td>${label}</td><td>0 / ${row.total}</td><td colspan="5">待评测</td></tr>`;
  return `<tr><td>${label}</td><td>${row.completed} / ${row.total}</td><td class="metric"><a href="${esc(fid.source)}">${fid.fid.toFixed(4)}</a></td><td>${fid.cfg.toFixed(1)}</td><td class="metric">${fid.is.toFixed(2)}</td><td class="metric"><a href="${esc(is.source)}">${is.is.toFixed(2)}</a></td><td>${is.cfg.toFixed(1)}</td></tr>`;
 }).join('')+'</tbody></table>';
}

function renderFlowHeadScale(){
 const study=D.flow_head_scale;
 if(!study){$('flow-scale-conclusion').textContent='尚未收录 flow head scale 评测。';return}
 const rows=study.rows,cfgs=study.cfg_values;
 $('flow-scale-family').onchange=renderFlowHeadCfgSweep;
 renderFlowHeadCfgSweep();
 $('flow-scale-coverage').textContent=`已收录 ${study.completed} / ${study.total} 组评测（${rows.length} 个模型 × ${cfgs.length} 档 CFG）。参数量单位为 M（百万），上行为 head，下行为全模型；悬停可查看精确数量。`;
 $('flow-scale-pending').textContent=rows.filter(row=>row.completed<cfgs.length).map(row=>
  `${row.family.toUpperCase()} depth ${row.depth}：${row.training_status==='running'?'训练中，评测待完成':`尚无 CFG=${row.results.filter(r=>!r.source).map(r=>r.cfg.toFixed(1)).join(' / ')} 的结果`}`
 ).join('；');
 const best=(row,cfg,key)=>{
  const values=rows.filter(r=>r.family===row.family).map(r=>r.results.find(s=>s.cfg===cfg)).filter(s=>s?.source).map(s=>s[key]);
  return values.length>1?(key==='fid'?Math.min(...values):Math.max(...values)):null;
 };
 const cells=(row,result)=>['fid','is'].map(key=>{
  const attrs=`data-cfg="${result.cfg.toFixed(1)}" data-metric="${key}"`;
  if(!result.source)return `<td class="metric" ${attrs} aria-label="${esc(row.label)} · CFG ${result.cfg.toFixed(1)} · ${key.toUpperCase()} 待评测"></td>`;
  const value=key==='fid'?result.fid.toFixed(4):`${result.is.toFixed(2)} ± ${result.is_std.toFixed(2)}`;
  return `<td class="metric${result[key]===best(row,result.cfg,key)?' best':''}" ${attrs}><a href="${esc(result.source)}">${value}</a></td>`;
 }).join('');
 $('flow-scale-table').innerHTML='<table aria-label="Flow head scale：同类 head、相同 CFG 下比较规模"><thead><tr><th class="model-name" rowspan="2" scope="col">模型 / Head 类型</th><th rowspan="2" scope="col">深度 × 宽度</th><th rowspan="2" scope="col">参数量 · Head / 总量 (M)</th>'+cfgs.map(cfg=>`<th colspan="2" scope="colgroup">CFG ${cfg.toFixed(1)}</th>`).join('')+'</tr><tr>'+cfgs.map(()=>'<th scope="col">FID ↓</th><th scope="col">IS ↑</th>').join('')+'</tr></thead><tbody>'+rows.map((row,index)=>{
  const status=row.training_status==='running'?'训练中 · 待评测':`训练完成 · 已评测 ${row.completed} / ${cfgs.length}`;
  const family=row.family==='b'?'Contextual · 跨 token attention':'Position-wise · AdaLN MLP';
  const counts=`Head：${row.head_parameters.toLocaleString()}；全模型：${row.total_parameters.toLocaleString()}`;
  return `<tr data-scale-model="${esc(row.id)}"${index&&row.family!==rows[index-1].family?' class="scale-family-start"':''}><td class="model-name"><strong>${esc(row.label)}</strong><div class="note">${family}</div><span class="status">${status}</span></td><td class="scale-shape">${row.depth} × ${row.width}</td><td class="scale-parameters" title="${counts}"><strong>${(row.head_parameters/1e6).toFixed(3)}</strong><br><small>${(row.total_parameters/1e6).toFixed(3)}</small></td>${row.results.map(result=>cells(row,result)).join('')}</tr>`;
 }).join('')+'</tbody></table>';
 const b=rows.filter(r=>r.family==='b').sort((a,b)=>a.depth-b.depth);
 const trajectories=cfgs.map(cfg=>{
  const results=b.map(r=>r.results.find(s=>s.cfg===cfg));
  if(b.length<2||!results.every(r=>r?.source))return '';
  const change=(results.at(-1).fid/results[0].fid-1)*100;
  return `<p><strong>CFG=${cfg.toFixed(1)}</strong>：B 的 FID 为 ${results.map(r=>r.fid.toFixed(4)).join(' → ')}；depth ${b.at(-1).depth} 较 depth ${b[0].depth} ${change<=0?'降低':'升高'} ${Math.abs(change).toFixed(2)}%。</p>`;
 }).join('');
 const f=rows.filter(r=>r.family==='f'&&r.depth>8),pending=f.some(r=>r.completed<cfgs.length);
 $('flow-scale-conclusion').innerHTML=`<h3>B · depth ${b.map(r=>r.depth).join(' → ')}，固定 CFG 比较 FID</h3>${trajectories}<p class="note">${pending?'F 扩展档尚无完整评测，暂不判断 MLP head 的 scale 效应。':'F 各扩展档已收齐评测，可在相同 CFG 下比较规模效应。'}</p>`;
}
