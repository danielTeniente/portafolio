const {test} = require('node:test');
const assert = require('node:assert/strict');
const E = require('../docs/2048/engine.js');
const {plan} = require('../docs/2048/motion.js');

test('merge animation keeps both source tiles and merges each tile at most once', () => {
  const result = plan([1,1,2,0,...Array(12).fill(0)], 3);
  assert.deepEqual(result.moves, [{from:0,to:0,exponent:1},{from:1,to:0,exponent:1},{from:2,to:1,exponent:2}]);
  assert.deepEqual(result.merges, [{to:0,exponent:2}]);
  assert.deepEqual(result.after, [2,2,0,0,...Array(12).fill(0)]);
});

test('motion paths match all four engine directions on seeded game boards', () => {
  for (const seed of [0,7,42]) {
    const game = new E.Game(seed);
    for (let step=0;step<200;step++) {
      for (let action=0;action<4;action++) {
        const result = plan(game.board,action), slide = E.slide(game.board,action);
        assert.deepEqual(result.after,slide.board);
        assert.equal(result.merges.reduce((sum,m)=>sum+2**m.exponent,0),slide.reward);
        assert.deepEqual(result.moves.map(m=>m.from).sort((a,b)=>a-b),game.board.map((v,i)=>v?i:-1).filter(i=>i>=0));
        for (const m of result.moves) {
          assert.equal(m.exponent,game.board[m.from]);
          assert.equal(action%2 ? Math.floor(m.from/4) : m.from%4, action%2 ? Math.floor(m.to/4) : m.to%4);
        }
      }
      const actions=E.legal(game.board); if (!actions.length) break;
      game.step(actions[step%actions.length]);
    }
  }
});

test('motion plan rejects invalid directions', () => {
  assert.throws(()=>plan(Array(16).fill(0),4));
});
