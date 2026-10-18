(function(root){
 'use strict';
 function mean(a){return a.reduce((s,v)=>s+v,0)/a.length}
 function decisions(actual,pred,exposure,capacity){
  if(!(exposure>0&&capacity>0)||actual.length!==pred.length)throw new Error('Invalid decision inputs');
  let accepted=0,breaches=0,overload=0;
  actual.forEach((v,i)=>{if(pred[i]*exposure<=capacity){accepted++;const excess=v*exposure-capacity;if(excess>0){breaches++;overload=Math.max(overload,excess)}}});
  return {accepted,declined:actual.length-accepted,breaches,overload,rate:accepted?breaches/accepted:null};
 }
 function metrics(actual,pred){const errors=actual.map((v,i)=>Math.abs(v-pred[i]));return {mae:mean(errors),rmse:Math.sqrt(mean(errors.map(v=>v*v)))}}
 function gammaTail(z){const x=4*z;return Math.exp(-x)*(1+x+x*x/2+x*x*x/6)}
 function paretoTail(z,alpha=1.5){const xm=(alpha-1)/alpha;return z<xm?1:(xm/z)**alpha}
 const api={mean,decisions,metrics,gammaTail,paretoTail};
 if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.RiskEngine=api;
})(typeof window!=='undefined'?window:globalThis);
