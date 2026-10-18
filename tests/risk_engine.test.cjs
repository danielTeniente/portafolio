const assert=require('node:assert/strict');
const E=require('../docs/risk/engine.js');
// Decisions must depend on forecasts, never on the future actual value.
const a=E.decisions([20,2,10],[2,20,10],1,10);
assert.deepEqual(a,{accepted:2,declined:1,breaches:1,overload:10,rate:.5});
assert.equal(E.decisions([200,200,200],[2,20,10],1,10).accepted,2);
assert.deepEqual(E.decisions([100],[20],1,10),{accepted:0,declined:1,breaches:0,overload:0,rate:null});
assert.equal(E.decisions([10],[10],1,10).breaches,0);
assert.equal(E.decisions([20],[2],.5,10).breaches,0);
assert.throws(()=>E.decisions([1],[1],0,10));
assert.equal(E.gammaTail(0),1);
assert.equal(E.paretoTail(.1),1);
assert.ok(Math.abs(E.paretoTail(20)/E.paretoTail(10)-2**-1.5)<1e-12);
assert.ok(E.gammaTail(10)<E.paretoTail(10));
assert.deepEqual(E.metrics([1,3],[2,2]),{mae:1,rmse:1});
console.log('Decision boundary, forecast-only gating, exposure, metrics and theoretical tail checks passed.');
