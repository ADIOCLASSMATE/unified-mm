function renderStyleInstruction(data, root, prefix='') {
 const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const path=p=>esc(prefix+p), signed=n=>(n>=0?'+':'')+n.toFixed(4);
 if(!data?.complete){root.textContent='风格实验尚未完成。';return}
 const models=data.models, labels=Object.fromEntries(models.map(m=>[m.id,m.id==='b_t2i_only_matched'?'B · T2I-only':m.label]));
 const assessment=data.assessment??{}, name=key=>data.conditions[key]??key;
 const intermediate=Boolean(data.contract.intermediate_checkpoint), textCount=data.text_probes.rows.length;
 const link=(p,label)=>`<a href="${path(p)}">${esc(label)}</a>`;
 const findings=(assessment.findings??[]).map(f=>`<article class="style-finding"><h3>${esc(f.title)}</h3><p>${esc(f.body)}</p><p class="style-note">${esc(f.evidence)}</p></article>`).join('');
 root.innerHTML=`<div class="style-report-lead"><strong>${esc(assessment.headline??'固定噪声的 instruction / 文本 ICL 对照')}</strong><p>${esc(assessment.summary??'图像、文本探针与训练数据审计已完成；下面保留全部输出。')}</p></div>
 <div class="style-facts"><span>${models.length} 个${intermediate?'中期 BF16 EMA':'final EMA'}</span><span>${data.verified_images} 张图像 · 全部保留</span><span>4 个主体 × 4 个主种子</span><span>共同 CFG 2.0 · Heun 10</span><span>${textCount} 条文本探针 · ${textCount/7} 个模型</span></div>
 <div class="style-findings">${findings}</div>
 <p class="style-note">${esc(assessment.limitations??'这是最终权重与提示干预的诊断。T2I-only 匹配 B 的 T2I 数据曝光和更新次数，但没有匹配总计算量与任务梯度占比；不能独立归因到 ClimbMix、I2T 或某一结构。')}</p>
 <div class="style-links">${link('manifest.json','完整提示与生成协议')}${link('results.jsonl','全部结果 JSONL')}${link('aggregates.csv','汇总 CSV')}${link('assessment.json','可视分析记录')}${link('COMPLETED.json','完整性校验')}</div>
 <section class="style-section"><h3>如何区分关键词响应与上下文规则</h3><p>先比较风格名称、视觉特征描述、新名称 dax 的定义、两条文本示例；再用 dax / wug 同时定义两种风格。交换两者的定义时，提示中的词及频数保持相同，目标仍请求 dax。颜色规则使用同样的红 / 蓝绑定对照。没有输入示例图片，也没有更新模型参数。</p><p class="style-note">同一主体与种子的所有条件、模型和 CFG 共享 CPU 生成的初始噪声。主实验 ${data.primary_images} 张；CFG 1.0 / 3.5 敏感性实验 ${data.sensitivity_images} 张，仅覆盖照片、名称、定义、两例提示及种子 42 / 43。最长序列 377 / 512 token，无截断。</p></section>
 <section class="style-section"><h3>全部图像 · 同输入与噪声横向比较</h3><div class="style-controls">
 <label>任务 <select data-style-control="style"><option value="pixel">Pixel art</option><option value="watercolor">Watercolor</option><option value="color">红 / 蓝规则</option></select></label>
 <label>主体 <select data-style-control="subject">${data.subjects.map(s=>`<option value="${esc(s[0])}">${esc(s[0])}</option>`).join('')}</select></label>
 <label>Seed <select data-style-control="seed">${[42,43,44,45].map(s=>`<option>${s}</option>`).join('')}</select></label>
 <label>CFG <select data-style-control="cfg"><option value="2">2.0 · 主实验</option><option value="1">1.0</option><option value="3.5">3.5</option></select></label>
 <label>条件 <select data-style-control="condition"><option value="key">主要对照</option><option value="balanced">词频平衡的规则绑定</option><option value="all">全部条件</option></select></label>
 </div><p class="style-note" data-style-count></p><div class="style-scroll" data-style-gallery></div></section>
 <section class="style-section"><h3>独立视觉模型的辅助诊断</h3><p>下图为 SigLIP 的风格描述相似度减去照片描述相似度，均值覆盖 4 主体 × 4 种子。正值只表示该视觉模型更偏向所给风格文本；它不是人评通过率，也不证明规则学习。主体丢失、网格伪影和颜色变化需结合上面的原图判断。</p><figure class="style-figure"><a href="${path('style-margin.png')}"><img loading="lazy" src="${path('style-margin.png')}" alt="三个模型的风格相对照片相似度热力图"></a><figcaption class="style-note">固定 CFG 2.0；每格 16 张，全部保留。</figcaption></figure><div class="style-links">${link('style-margin.png','PNG')}${link('style-margin.svg','SVG')}${link('style-margin.pdf','PDF')}</div>
 <details><summary>展开配对规则切换的全部数值</summary><p class="style-note">风格差值 = [pixel − watercolor]在 dax 定义为 pixel 时 − [pixel − watercolor]在 dax 定义为 watercolor 时。颜色为对应的 red − blue 差。方向为正表明提示变化影响了相似度；“两端均偏向目标”要求两张图分别更像所请求目标，仍是代理指标。</p><div class="style-scroll"><table class="style-contrast-table"><thead><tr><th>模型</th><th>对照</th><th>配对数</th><th>平均方向差</th><th>正方向</th><th>两端均偏向目标</th></tr></thead><tbody>${data.contrasts.map(r=>`<tr><td>${esc(labels[r.model])}</td><td>${esc(r.contrast)}</td><td>${r.n}</td><td>${signed(r.mean_delta)}</td><td>${r.positive}/${r.n}</td><td>${r.both_proxy_targets}/${r.n}</td></tr>`).join('')}</tbody></table></div></details></section>
 <section class="style-section"><h3>文本中知道风格，是否就能画出来？</h3><p>${esc(assessment.text_analysis??'同时检查原始 Qwen 基座、B、S2-single、T2I-only、I2T-only、text-only 的相同文本续写。这里只保留生成证据，不把少量文本探针当作正式语言基准。')}</p><p class="style-note">每条最多 80 token，greedy 基座续写，不使用聊天模板。only 模型未训练的任务属于跨任务诊断。</p><details><summary>查看全部 ${textCount} 条文本探针与原始输出</summary><div class="style-scroll"><table class="style-text-table"><thead><tr><th>模型</th><th>探针 / 输入</th><th>输出</th></tr></thead><tbody>${data.text_probes.rows.map(r=>`<tr><td>${esc(labels[r.model]??r.model)}</td><td>${esc(r.id)}<pre class="style-text-output">${esc(r.prompt)}</pre></td><td><pre class="style-text-output">${esc(r.text)}</pre></td></tr>`).join('')}</tbody></table></div></details></section>
 <section class="style-section"><h3>训练数据中的风格监督</h3><p>${esc(assessment.training_analysis??'训练索引固定随机抽样 1,024 张图，每张都包含 12 条提示，其中包括 watercolor、graphic illustration 与 3D render。实际数据加载器轮换这些文本，但均读取该图的同一个 posterior 缓存；没有按风格替换目标图像。')}</p><p class="style-note">这是数据约束的直接证据；是否因此抑制风格控制仍需用修正后的训练数据做干预实验。ImageNet 也不能被简单视为绝无插画。本次未审计 Qwen 原始预训练全量语料，不能声称 pixel art 从未被见过。</p><details><summary>查看 8 条固定抽样的完整训练提示</summary>${data.training_audit.examples.map(e=>`<details><summary>${esc(e.image_id)} · index ${e.manifest_index}</summary>${e.image?`<img loading="lazy" style="max-width:256px" src="${path(e.image)}" alt="训练目标原图">`:''}${e.prompts.map(p=>`<p><strong>${esc(p.style)}</strong> ${esc(p.prompt)}</p>`).join('')}</details>`).join('')}</details><div class="style-links">${link('training_audit.json','抽样数据审计')}${link('text_probes.json','文本探针原始记录')}</div></section>
 <section class="style-section"><h3>原网页固定样例评测与本实验的关系</h3><p>${esc(assessment.only_analysis??'补齐 B 的 T2I-only 与 I2T-only：每模型 128 张 T2I、64 条 I2T、64 条文本续写。沿用原页面固定输入、种子与 CFG 3.5；B text-only 的既有结果继续保留。')}</p>${(assessment.only_examples??[]).map(t=>`<p class="style-note">${esc(t)}</p>`).join('')}<p><a href="${path('../../index.html#qualitative/t2i')}">回到主页面的 B / only 固定样例对照 →</a></p></section>
 <details><summary>权重与复现来源</summary>${models.map(m=>`<p><strong>${esc(labels[m.id])}</strong> · step ${m.source.global_step}<br><code>${esc(m.checkpoint)}</code></p>`).join('')}<p class="style-note">${esc(data.scorer)}</p></details>`;
 const controls=Object.fromEntries([...root.querySelectorAll('[data-style-control]')].map(e=>[e.dataset.styleControl,e]));
 function gallery(){
  const style=controls.style.value, color=style==='color',subject=color?'color_teapot':controls.subject.value,seed=+controls.seed.value,cfg=+controls.cfg.value,condition=controls.condition.value;
  controls.subject.disabled=color;controls.condition.disabled=color;
  const order=color?['red','blue','nonce','definition_red','definition_blue','examples_red','examples_blue']:condition==='key'?['photo','name','description','definition','examples','scrambled','binding_definition','binding_examples']:condition==='balanced'?['photo','name','binding_definition','binding_definition_swapped','binding_examples','binding_examples_swapped']:['photo',...Object.keys(data.conditions).filter(x=>x!=='photo')];
  const rows=data.records.filter(r=>r.subject_id===subject&&r.seed===seed&&r.cfg===cfg&&(r.style===style||(!color&&r.condition==='photo')));
  const available=order.filter(c=>rows.some(r=>r.condition===c));
  root.querySelector('[data-style-count]').textContent=`当前显示 ${available.length*models.length} 张；未运行的 CFG / seed 组合留空。点击图片查看原始 256×256 PNG。${cfg===2?'':'敏感性实验仅含照片、名称、单定义、单风格两例。'}`;
  root.querySelector('[data-style-gallery]').innerHTML=`<table class="style-gallery"><thead><tr><th>提示条件</th>${models.map(m=>`<th>${esc(labels[m.id])}</th>`).join('')}</tr></thead><tbody>${available.map(c=>{
   const sample=rows.find(r=>r.condition===c);
   return `<tr><td><strong>${esc(name(c))}</strong><span class="style-badge">目标 ${esc(sample.expected_style??'未指定')} · ${sample.text_tokens} 文本 token</span><details><summary>完整提示</summary><pre>Generate an image matching this description: ${esc(sample.prompt)}</pre></details></td>${models.map(m=>{
    const r=rows.find(r=>r.condition===c&&r.model===m.id);if(!r)return '<td>未运行</td>';
    const value=color?r.red_minus_blue:style==='pixel'?r.pixel_minus_photo:r.watercolor_minus_photo;
    return `<td><a href="${path(r.image)}"><img loading="lazy" src="${path(r.image)}" alt="${esc(labels[m.id]+' / '+name(c)+' / '+subject+' / seed '+seed)}"></a><span class="style-badge">${color?'red − blue':'style − photo'} ${signed(value)}</span></td>`;
   }).join('')}</tr>`;
  }).join('')}</tbody></table>`;
 }
 for(const el of Object.values(controls))el.addEventListener('change',gallery);
 gallery();
}
