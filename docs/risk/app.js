'use strict';
const data=window.RISK_DATA,E=window.RiskEngine,$=id=>document.getElementById(id);
const f=v=>v.toLocaleString(undefined,{maximumFractionDigits:2});
const names=Object.keys(data.runs[0].predictions);
names.forEach(n=>$('model').add(new Option(n,n)));
for(let i=0;i<150;i++)$('day').add(new Option(`Day ${i+1}${i===75?' · midpoint':''}`,i));
const descriptions={light:'Control: unchanged daily/weekly structure with light-tailed Gamma innovations. There is no regime change; models have a learnable signal and unavoidable random error.',pareto:'Unchanged seasonal structure with Pareto innovations from the start. This is a stable heavy-tailed regime—not distribution drift. Training also contains heavy tails, so they are not hidden from the models.',shift:'Identical training history to the light-tailed control. Test days 1–75 remain light-tailed; days 76–150 use Pareto innovations with three times the conditional mean. The shift is predeclared and disclosed, not chosen after seeing errors.'};
function run(){return data.runs.find(r=>r.seed===Number($('seed').value)&&r.scenario===$('scenario').value)}
function table(target,headers,rows){const t=document.createElement('table'),head=document.createElement('thead'),h=document.createElement('tr');headers.forEach(v=>{const c=document.createElement('th');c.textContent=v;h.append(c)});head.append(h);t.append(head);const b=document.createElement('tbody');rows.forEach(row=>{const tr=document.createElement('tr');row.forEach(v=>{const c=document.createElement('td');c.textContent=v;tr.append(c)});b.append(tr)});t.append(b);$(target).replaceChildren(t)}
function peak(r){let day=0,hour=0;for(let d=0;d<r.actual.length;d++)for(let h=0;h<24;h++)if(r.actual[d][h]>r.actual[day][hour]){day=d;hour=h}return {day,hour,value:r.actual[day][hour]}}
function chooseInitial(){const r=run();$('model').value=r.selected;$('day').value=peak(r).day;render()}
function render(){
 const r=run(),name=$('model').value,actual=r.actual.flat(),pred=r.predictions[name].flat(),m=r.metrics.find(v=>v.model===name),e=Number($('exposure').value),c=Number($('capacity').value);
 $('scenario-description').textContent=descriptions[r.scenario];$('mae').textContent=f(m.mae);$('worst-error').textContent=f(m.worst_underforecast);
 const sorted=[...actual].sort((a,b)=>b-a);$('tail-share').textContent=f(100*sorted.slice(0,36).reduce((s,v)=>s+v,0)/actual.reduce((s,v)=>s+v,0))+'%';
 $('exposure-value').textContent=e.toFixed(1)+'×';$('capacity-value').textContent=c+' units';
 const largest=peak(r),forecast=r.predictions[name][largest.day][largest.hour],accepted=forecast*e<=c;
 $('peak-value').textContent=f(largest.value);
 $('peak-location').textContent=`Day ${largest.day+1} · ${String(largest.hour).padStart(2,'0')}:00 · largest of 3,600 test observations`;
 $('peak-forecast').textContent=f(forecast);
 $('peak-gap').textContent=f(Math.max(largest.value-forecast,0));
 $('peak-impact').textContent=accepted?(largest.value*e>c?`${f(largest.value*e-c)} units above capacity`:'Accepted · within capacity'):'Declined by the forecast gate';
 $('largest').setAttribute('aria-pressed',String(Number($('day').value)===largest.day));
 const d=E.decisions(actual,pred,e,c);$('accepted').textContent=f(d.accepted);$('declined').textContent=`${d.declined} of 3,600 hours declined`;
 $('breaches').textContent=f(d.breaches);$('breach-rate').textContent=d.rate===null?'No accepted work':f(100*d.rate)+'% of accepted hours';$('overload').textContent=f(d.overload);
 $('decision-reading').textContent=d.breaches?`${name} accepted ${d.accepted} hours of work and exceeded capacity in ${d.breaches}. Its average forecast error is ${f(m.mae)}, but the largest accepted overload is ${f(d.overload)} units. Adjust exposure and capacity to see how the same predictions lead to different consequences.`:`${name} caused no capacity breaches under this decision rule in this particular run. That is a successful observed outcome—not proof of safety outside this sample. Adjust exposure or capacity to explore its limits.`;
 table('score-table',['Model','MAE','RMSE','Top-1% mean shortfall','Accepted breaches','Declined hours'],r.metrics.map(v=>{const q=E.decisions(actual,r.predictions[v.model].flat(),e,c);return [v.model,f(v.mae),f(v.rmse),f(v.top_one_percent_underforecast),q.breaches,q.declined]}));
 $('selection-note').textContent=`Validation-selected model: ${r.selected}. This selection uses four historical folds, not test errors or breach counts. No retuning occurs when you move the decision sliders.`;
 table('period-table',['Model','First 75 days MAE','Last 75 days MAE'],names.map(n=>[n,f(E.metrics(actual.slice(0,1800),r.predictions[n].flat().slice(0,1800)).mae),f(E.metrics(actual.slice(1800),r.predictions[n].flat().slice(1800)).mae)]));
 table('seed-table',['Data seed',`${name} MAE`,'Accepted breaches','Declined hours'],data.runs.filter(v=>v.scenario===r.scenario).map(v=>{const q=E.decisions(v.actual.flat(),v.predictions[name].flat(),e,c);return [v.seed,f(v.metrics.find(m=>m.model===name).mae),q.breaches,q.declined]}));
 drawDay(r,name,e,c);
}
function drawDay(r,name,e,c){
 const day=Number($('day').value),a=r.actual[day],p=r.predictions[name][day],cap=c/e;
 const W=1000,H=320,L=55,R=30,T=25,B=40;
 // Do not stretch a quiet day to a distant capacity; indicate an off-chart limit.
 const maxValue=Math.max(...a,...p)*1.1,limitVisible=cap<=maxValue;
 const y=v=>H-B-v/maxValue*(H-T-B),x=i=>L+i*(W-L-R)/23;
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Actual and forecast load for test day ${day+1}"><title>Observed and forecast loads; original load units before exposure</title>`;
 for(let i=0;i<=4;i++){const v=i*maxValue/4;s+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#e4e5e9"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end" font-size="13" fill="#606875">${f(v)}</text>`}
 [0,4,8,12,16,20,23].forEach(i=>s+=`<text x="${x(i)}" y="${H-15}" text-anchor="middle" font-size="13" fill="#606875">${i}:00</text>`);
 const path=values=>values.map((v,i)=>`${i?'L':'M'}${x(i)},${y(v)}`).join(' ');
 s+=`<path d="${path(a)}" fill="none" stroke="#242b38" stroke-width="3"/><path d="${path(p)}" fill="none" stroke="#ca9029" stroke-width="3"/>`;
 if(limitVisible)s+=`<line x1="${L}" x2="${W-R}" y1="${y(cap)}" y2="${y(cap)}" stroke="#bc5160" stroke-width="2" stroke-dasharray="6 5"/>`;
 a.forEach((v,i)=>s+=`<circle cx="${x(i)}" cy="${y(v)}" r="3" fill="#242b38"><title>Hour ${i}: actual ${f(v)}, forecast ${f(p[i])}</title></circle>`);
 const largest=peak(r);
 if(day===largest.day){const px=x(largest.hour),py=y(largest.value),anchor=largest.hour>17?'end':'start',tx=largest.hour>17?px-16:px+16;
  s+=`<line x1="${px}" x2="${px}" y1="${py}" y2="${H-B}" stroke="#ca9029" stroke-dasharray="4 5" opacity="0.65"/><circle cx="${px}" cy="${py}" r="15" fill="#f0ba45" opacity="0.24"/><circle cx="${px}" cy="${py}" r="7" fill="#f0ba45" stroke="#785218" stroke-width="2"/><text x="${tx}" y="${py-12}" text-anchor="${anchor}" font-size="15" font-weight="700" fill="#785218">LARGEST · ${f(largest.value)}</text>`;
 }
 $('forecast-chart').innerHTML=s+'</svg>';
 $('day-note').textContent=`Day ${day+1} · ${name} · MAE ${f(E.metrics(a,p).mae)}. Capacity/exposure = ${f(cap)} load units${limitVisible?'':'; above this chart’s range'}. The vertical scale adapts per day; inspect the labeled values, not only line shapes.`;
 table('hour-values',['Hour','Actual load','Forecast','Decision','Accepted overload'],a.map((v,i)=>[i,f(v),f(p[i]),p[i]*e<=c?'Accept':'Decline',f(p[i]*e<=c?Math.max(v*e-c,0):0)]));
}
function drawTails(){
 const W=1000,H=340,L=65,R=25,T=25,B=50,x=z=>L+(Math.log10(z)+1)/3*(W-L-R),y=p=>T-Math.log10(p)/8*(H-T-B);
 let s=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Theoretical Gamma and Pareto survival curves on logarithmic axes"><title>Same mean, different tail probabilities</title>`;
 for(let k=0;k<=8;k+=2){const yy=y(10**-k);s+=`<line x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}" stroke="#e4e5e9"/><text x="${L-10}" y="${yy+4}" text-anchor="end" font-size="13">10^${-k}</text>`}
 [.1,1,10,100].forEach(z=>s+=`<text x="${x(z)}" y="${H-25}" text-anchor="middle" font-size="13">${z}×</text>`);
 [[E.gammaTail,'#505ca0'],[E.paretoTail,'#ca9029']].forEach(([fn,color])=>{let path='';for(let i=0;i<=300;i++){const z=10**(-1+3*i/300),p=fn(z);if(p<1e-8)break;path+=(path?' L':'M')+x(z)+','+y(p)}s+=`<path d="${path}" fill="none" stroke="${color}" stroke-width="3"/>`});
 s+='<text x="90" y="30" font-size="14" fill="#505ca0">Gamma · light tail</text><text x="600" y="30" font-size="14" fill="#a57721">Pareto · power-law tail</text></svg>';
 $('tail-chart').innerHTML=s;
}
['scenario','seed'].forEach(id=>$(id).addEventListener('change',chooseInitial));
['model','day'].forEach(id=>$(id).addEventListener('change',render));
['exposure','capacity'].forEach(id=>$(id).addEventListener('input',render));
$('first').addEventListener('click',()=>{$('day').value='0';render()});
$('largest').addEventListener('click',()=>{$('day').value=peak(run()).day;render()});
$('median').addEventListener('click',()=>{const r=run(),p=r.predictions[$('model').value];const order=r.actual.map((a,i)=>({i,error:E.metrics(a,p[i]).mae})).sort((a,b)=>a.error-b.error);$('day').value=order[Math.floor(order.length/2)].i;render()});
Object.entries(data.protocol.architecture).forEach(([name,description])=>{const h=document.createElement('h3'),p=document.createElement('p');h.textContent=name;p.textContent=description;$('architectures').append(h,p)});
$('runtime').textContent=`Recorded computation: ${f(data.elapsed_seconds)} seconds for all nine scenario/seed runs. The light and shift scenarios share identical pre-change fits. Timings exclude imports and installation. All numerical observations and predictions are retained in the downloadable data.`;
drawTails();chooseInitial();

// Fixed retrospective illustration; no fitting uses this event.
const exampleRun=data.runs.find(r=>r.seed===42&&r.scenario==='pareto');
const examplePeak=peak(exampleRun),exampleCapacity=10;
const exampleModels=names.filter(n=>n!=='Repeat yesterday');
const exampleForecasts=exampleModels.map(n=>exampleRun.predictions[n][examplePeak.day][examplePeak.hour]);
const safeExposure=.1,severeExposure=3;
const safeLoad=examplePeak.value*safeExposure,severeLoad=examplePeak.value*severeExposure;
$('example-context').textContent=`One observed load of ${f(examplePeak.value)} units. The same four forecasts. Capacity fixed at ${exampleCapacity} units. Only exposure changes. Worked example: power-law regime, seed 42, day ${examplePeak.day+1} at ${String(examplePeak.hour).padStart(2,'0')}:00.`;
$('safe-loss').textContent=f(Math.max(safeLoad-exampleCapacity,0))+' units';
$('severe-loss').textContent=f(Math.max(severeLoad-exampleCapacity,0))+' units';
$('safe-detail').textContent=`Exposure ${safeExposure}× → actual workload ${f(safeLoad)} / ${exampleCapacity} capacity. ${f(exampleCapacity-safeLoad)} units of capacity remain. All four forecasts permit the commitment.`;
$('severe-detail').textContent=`Exposure ${severeExposure}× → actual workload ${f(severeLoad)} / ${exampleCapacity} capacity: ${f(severeLoad/exampleCapacity)}× the available capacity. Predicted workloads are only ${f(Math.min(...exampleForecasts)*severeExposure)}–${f(Math.max(...exampleForecasts)*severeExposure)} units.`;
$('all-models-reading').textContent=exampleForecasts.every(v=>v<examplePeak.value)
 ? `All four models underestimate this observation: forecasts range from ${f(Math.min(...exampleForecasts))} to ${f(Math.max(...exampleForecasts))}, versus ${f(examplePeak.value)} actual. At 3× exposure, every model accepts work and incurs the same ${f(severeLoad-exampleCapacity)}-unit overload. For this event, changing the model does not change the severe outcome; limiting exposure does.`
 : 'Inspect each model against the same observation. The decision outcome depends on both the forecast and the chosen exposure.';
table('event-models',['Model','Forecast','Actual','Underforecast','At 0.1× exposure','At 3× exposure'],exampleModels.map(n=>{
 const p=exampleRun.predictions[n][examplePeak.day][examplePeak.hour];
 const safe=E.decisions([examplePeak.value],[p],safeExposure,exampleCapacity),severe=E.decisions([examplePeak.value],[p],severeExposure,exampleCapacity);
 return [n,f(p),f(examplePeak.value),f(Math.max(examplePeak.value-p,0)),safe.accepted?f(safe.overload)+' unserved':'Declined',severe.accepted?f(severe.overload)+' unserved':'Declined'];
}));
function applyExample(exposure){$('scenario').value='pareto';$('seed').value='42';$('model').value=exampleRun.selected;$('day').value=examplePeak.day;$('exposure').value=exposure;$('capacity').value=exampleCapacity;render();document.getElementById('decisions').scrollIntoView({block:'start'});}
$('apply-safe').addEventListener('click',()=>applyExample(safeExposure));
$('apply-severe').addEventListener('click',()=>applyExample(severeExposure));
