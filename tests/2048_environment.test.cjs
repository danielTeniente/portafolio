const {test} = require('node:test');
const assert = require('node:assert/strict');
const E = require('../docs/2048/engine.js');
const board = row => [...row, ...Array(12).fill(0)];

test('left merges once per tile and sums points', () => {
  const original = board([1, 1, 1, 1]);
  const result = E.slide(original, 3);
  assert.deepEqual(result.board, board([2, 2, 0, 0]));
  assert.equal(result.reward, 8);
  assert.deepEqual(original, board([1, 1, 1, 1]));
  assert.deepEqual(E.slide(board([1, 1, 2, 0]), 3).board, board([2, 2, 0, 0]));
  assert.equal(E.slide(board([1, 1, 2, 0]), 3).reward, 4);
});
test('directional order, gaps, and vertical merges', () => {
  assert.deepEqual(E.slide(board([1, 1, 1, 0]), 1).board, board([0, 0, 1, 2]));
  assert.deepEqual(E.slide(board([1, 0, 1, 0]), 3).board, board([2, 0, 0, 0]));
  const vertical = [1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0];
  assert.deepEqual(E.slide(vertical, 0).board, [2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0]);
  assert.deepEqual(E.slide(vertical, 2).board, [0,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0]);
});
test('reset spawns exactly two tiles and seeded runs reproduce', () => {
  const a = new E.Game(42), b = new E.Game(42);
  assert.equal(a.board.filter(Boolean).length, 2);
  assert.ok(a.board.every(v => [0, 1, 2].includes(v)));
  for (let i = 0; i < 100; i++) { const actions = E.legal(a.board); if (!actions.length) break; assert.deepEqual(a.step(actions[i % actions.length]), b.step(actions[i % actions.length])); }
});
test('invalid moves consume no random draw; valid moves spawn exactly one tile', () => {
  const a = new E.Game(7), b = new E.Game(7);
  a.board = board([1,0,0,0]); b.board = a.board.slice();
  const noOp = a.step(3);
  assert.equal(noOp.changed, false); assert.equal(noOp.reward, 0); assert.equal(noOp.spawned, null); assert.equal(a.moves, 0);
  const next = a.step(1);
  assert.deepEqual(next, b.step(1));
  assert.equal(next.board.filter(Boolean).length, 2);
  assert.equal(a.moves, 1); assert.equal(a.score, 0);
  next.board[0] = 99; assert.notEqual(a.board[0], 99);
});
test('terminal requires no legal moves, not merely a full board', () => {
  const dead = [1,2,1,2,2,1,2,1,1,2,1,2,2,1,2,1];
  assert.deepEqual(E.legal(dead), []);
  const g = new E.Game(); g.board = dead.slice(); assert.equal(g.step(0).done, true);
  dead[0] = 2; assert.ok(E.legal(dead).length > 0);
});
test('2048 is a milestone and score is accumulated merge reward', () => {
  const g = new E.Game(); g.board = board([10,10,0,0]);
  const result = g.step(3);
  assert.equal(result.reached2048, true); assert.equal(result.done, false);
  assert.equal(result.reward, 2048); assert.equal(g.score, 2048);
});
test('spawn draws choose a position uniformly and respect the 90% threshold', () => {
  const g = new E.Game(); g.board = Array(16).fill(0);
  let draws = [.5, .899999]; g.random = () => draws.shift();
  assert.deepEqual(g.spawn(), {index: 8, value: 2});
  g.board.fill(0); draws = [.5, .9];
  assert.deepEqual(g.spawn(), {index: 8, value: 4});
});
test('lossless representations preserve positions and one-hot vocabulary is explicit', () => {
  const s = board([0,1,2,11]);
  assert.deepEqual(E.encode(s, 'values').slice(0,4), [0,2,4,2048]);
  const exponents = E.encode(s); exponents[0] = 9; assert.equal(s[0], 0);
  const hot = E.encode(s, 'onehot');
  assert.equal(hot.flat().length, 288);
  hot.forEach((row, i) => { assert.equal(row.reduce((a,b) => a+b, 0), 1); assert.equal(row.indexOf(1), s[i]); });
  assert.throws(() => E.encode(board([18,0,0,0]), 'onehot'));
  assert.throws(() => E.encode(s, 'unknown'));
  assert.throws(() => E.slide(s, 4));
});
test('summary aliases distinct states with different left-move rewards', () => {
  const a = [0,0,0,0,0,1,1,0,0,2,0,0,0,0,0,0];
  const b = [0,0,0,0,0,1,2,0,0,1,0,0,0,0,0,0];
  assert.deepEqual(E.encode(a, 'summary'), E.encode(b, 'summary'));
  assert.notEqual(E.slide(a, 3).reward, E.slide(b, 3).reward);
});
