'use strict';
const r = window.REPORT;
const $ = id => document.getElementById(id);
const fmt = v => Number(v).toFixed(1);
const avg = values => values.reduce((a,b)=>a+b,0)/values.length;
const err = (name,i) => avg(r.actual[i].map((v,j)=>Math.abs(v-r.predictions[name][i][j])));
const winner = r.metrics.find(m=>m.model===r.winner);
$('winner').textContent=r.winner;
$('skill').textContent=`${fmt(winner.skill_pct)}%`;
$('days').textContent=r.audit.test_days;
$('period').textContent=`${r.dates[0]} → ${r.dates.at(-1)}`;
$('audit').textContent=`${r.audit.rows.toLocaleString()} consecutive hourly records. No missing values or duplicate timestamps. Supplied file: quito_2017_2018 (1).csv.`;
$('hours').textContent=r.hours.map(h=>`${String(h).padStart(2,'0')}:00`).join(', ');
Object.keys(r.predictions).forEach(name=>$('model').add(new Option(name,name)));
r.dates.forEach((day,i)=>$('day').add(new Option(day,i)));
$('model').value=r.winner;
const ranked=[...r.metrics].sort((a,b)=>a.mae-b.mae);
const max=Math.max(...ranked.map(m=>m.mae));
ranked.forEach(m=>{
  const row=document.createElement('div');row.className='rank-row';
  const name=document.createElement('span');name.textContent=m.model;
  const track=document.createElement('div');const bar=document.createElement('div');
  bar.className='bar'+(m.model===r.winner?' selected':'');bar.style.width=`${100*m.mae/max}%`;track.append(bar);
  const value=document.createElement('b');value.textContent=fmt(m.mae);row.append(name,track,value);$('ranking').append(row);
  const duration=s=>s>0&&s<.01?'<0.01 s':`${s.toFixed(2)} s`;
  const tr=document.createElement('tr');[m.model,fmt(m.mae),fmt(m.rmse),fmt(m.p95_daily_mae),duration(m.fit_seconds),duration(m.total_seconds)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});$('metrics').append(tr);
});
const worstIndex=r.dates.reduce((best,_,i)=>err(r.winner,i)>err(r.winner,best)?i:best,0);
$('risk').textContent=`${fmt(err(r.winner,worstIndex))} RS`;
$('risk-copy').textContent=`Daylight MAE on ${r.dates[worstIndex]}. Compare this stress case with the model’s average test MAE of ${fmt(winner.mae)} RS units.`;
function chooseDay(worst){const order=r.dates.map((_,i)=>i).sort((a,b)=>err($('model').value,a)-err($('model').value,b));$('day').value=order[worst?order.length-1:Math.floor(order.length/2)];render()}
function render(){
 const name=$('model').value,i=Number($('day').value),actual=r.actual[i],pred=r.predictions[name][i],baseline=r.predictions['Repeat yesterday'][i];
 const width=1000,height=340,left=56,right=20,top=30,bottom=45;
 const limit=Math.ceil(Math.max(...actual,...pred,...baseline)/200)*200 || 200;
 const x=j=>left+j*(width-left-right)/(r.hours.length-1),y=v=>height-bottom-v/limit*(height-top-bottom);
 const path=values=>values.map((v,j)=>`${j?'L':'M'}${x(j)},${y(v)}`).join(' ');
 let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Observed and forecast daylight radiation for ${r.dates[i]}"><title>Daylight radiation comparison</title><text x="${left}" y="15" font-size="11" fill="#5c706b">RS units</text>`;
 for(let tick=0;tick<=limit;tick+=limit/4){svg+=`<line x1="${left}" y1="${y(tick)}" x2="${width-right}" y2="${y(tick)}" stroke="#e5eae3"/><text x="${left-9}" y="${y(tick)+4}" text-anchor="end" font-size="11" fill="#5c706b">${tick}</text>`}
 r.hours.forEach((h,j)=>{svg+=`<text x="${x(j)}" y="${height-18}" text-anchor="middle" font-size="11" fill="#5c706b">${h}:00</text>`});
 svg+=`<path d="${path(baseline)}" fill="none" stroke="#879c95" stroke-width="2" stroke-dasharray="6 6"/><path d="${path(actual)}" fill="none" stroke="#18332f" stroke-width="3"/><path d="${path(pred)}" fill="none" stroke="#d9a230" stroke-width="3"/>`;
 actual.forEach((v,j)=>{svg+=`<circle cx="${x(j)}" cy="${y(v)}" r="4" fill="#18332f"><title>${r.hours[j]}:00 · Observed ${fmt(v)} · Forecast ${fmt(pred[j])}</title></circle>`});
 $('forecast-chart').innerHTML=svg+'</svg>';
 $('day-summary').textContent=`${r.dates[i]} · ${name}: MAE ${fmt(err(name,i))} RS units. Repeat yesterday: ${fmt(err('Repeat yesterday',i))}. Forecast issued at 00:00 using only the preceding calendar day.`;
 $('hourly-table').innerHTML='<table><thead><tr><th>Hour</th><th>Observed</th><th>Forecast</th><th>Repeat yesterday</th></tr></thead><tbody>'+r.hours.map((h,j)=>`<tr><td>${h}:00</td><td>${fmt(actual[j])}</td><td>${fmt(pred[j])}</td><td>${fmt(baseline[j])}</td></tr>`).join('')+'</tbody></table>';
}
$('model').addEventListener('change',render);$('day').addEventListener('change',render);
$('typical').addEventListener('click',()=>chooseDay(false));$('worst').addEventListener('click',()=>chooseDay(true));
chooseDay(false);
const foldNotice=document.createElement('p');
foldNotice.textContent='Short-window caution: each validation block contains only 51–52 forecast days. Weather mix and serially correlated errors can inflate apparent variation; the effective sample size may be smaller. Four folds over this dataset cannot establish multi-year stability.';
$('fold-dates').append(foldNotice);
r.audit.folds.forEach((f,i)=>{const p=document.createElement('p');p.textContent=`Fold ${i+1}: ${f.first_date} → ${f.last_date} · ${f.train_days} training days · ${f.validation_days} validation days`; $('fold-dates').append(p)});
const stable=document.createElement('table');
stable.innerHTML='<caption>Daylight MAE by validation period · source RS units</caption><thead><tr><th>Model</th><th>Fold 1</th><th>Fold 2</th><th>Fold 3</th><th>Fold 4</th><th>Fold SD</th></tr></thead>';
const body=document.createElement('tbody');
r.metrics.forEach(m=>{const tr=document.createElement('tr');[m.model,...m.fold_maes.map(fmt),fmt(m.cv_std)].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});body.append(tr)});
stable.append(body);$('stability-table').append(stable);
const neuralNames=['RNN','GRU','LSTM'];
const neural=r.metrics.filter(m=>neuralNames.includes(m.model));
const classical=r.metrics.filter(m=>!neuralNames.includes(m.model)&&m.model!=='Repeat yesterday');
const bestNeural=[...neural].sort((a,b)=>a.validation_mae-b.validation_mae)[0];
const bestClassical=[...classical].sort((a,b)=>a.validation_mae-b.validation_mae)[0];
const gain=100*(1-bestNeural.mae/bestClassical.mae);
$('complexity-verdict').textContent=gain<=0
 ? `In this benchmark, extra recurrent complexity does not earn its place. ${bestNeural.model}, selected among the recurrent models by validation, has ${fmt(-gain)}% higher test error than ${bestClassical.model}, the validation-selected classical model. Prefer the simpler model for this task unless new evidence changes the trade-off.`
 : `${bestNeural.model}, selected among the recurrent models by validation, reduces test error by ${fmt(gain)}% versus ${bestClassical.model}, the validation-selected classical model. This is an observed gain, not proof of statistical or business significance; judge it alongside absolute training cost, seed variation, and maintenance needs.`;
neural.forEach(m=>{
 const article=document.createElement('article');const title=document.createElement('h3');title.textContent=m.model;
 const p=document.createElement('p');p.textContent=`Test MAE ${fmt(m.mae)} · ${fmt(100*(m.mae/bestClassical.mae-1))}% error change vs. ${bestClassical.model} (positive is worse). Search + final fit: ${m.total_seconds.toFixed(2)} s versus ${bestClassical.total_seconds.toFixed(3)} s. Final refit: ${m.epochs} epochs.`;
 article.append(title,p);$('complexity-evidence').append(article);
});
const seedTable=document.createElement('table');seedTable.innerHTML='<thead><tr><th>Model</th><th>Seed 42 MAE</th><th>Seed 7 MAE</th><th>Seed 123 MAE</th><th>Observed range</th></tr></thead>';
const seedBody=document.createElement('tbody');
neural.forEach(m=>{const entries=r.seed_sensitivity.filter(s=>s.model===m.model);const values=entries.map(s=>s.mae);const tr=document.createElement('tr');[m.model,...[42,7,123].map(seed=>fmt(entries.find(s=>s.seed===seed).mae)),`${fmt(Math.min(...values))}–${fmt(Math.max(...values))}`].forEach(v=>{const td=document.createElement('td');td.textContent=v;tr.append(td)});seedBody.append(tr)});
seedTable.append(seedBody);$('seed-results').append(seedTable);
Object.entries(r.architectures).forEach(([name,info])=>{const card=document.createElement('details');const heading=document.createElement('summary');heading.textContent=name;const structure=document.createElement('p');structure.className='architecture-flow';structure.textContent=info.structure;const settings=document.createElement('p');settings.className='note';settings.textContent=info.settings;card.append(heading,structure,settings);$('architecture-cards').append(card)});
