const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const E = require('../docs/2048/engine.js');
for (const stage of [1,2,"afterstate","hybrid"]) {
const report = JSON.parse(fs.readFileSync(path.join(__dirname, stage === 1 ? '../results/2048/report.json' : stage === 2 ? '../results/2048-stage2/report.json' : stage === "hybrid" ? "../results/2048-hybrid/report.json" : stage === "afterstate" ? "../results/2048-afterstate/report.json" : `../results/2048-stage3-${stage}/report.json`), 'utf8'));
const script = stage === 1 ? '../docs/2048/data.js' : stage === 2 ? '../docs/2048/stage2-data.js' : stage === "hybrid" ? "../docs/2048/hybrid-data.js" : stage === "afterstate" ? "../docs/2048/afterstate-data.js" : `../docs/2048/stage3-${stage}-data.js`;

test(`Stage ${stage}: dashboard data matches the auditable saved report`, () => {
  const context = {window: {}};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, script), 'utf8'), context);
  assert.equal(JSON.stringify(context.window[stage === 1 ? 'LAB2048_RESULTS' : stage === 2 ? 'LAB2048_STAGE2' : stage === 'hybrid' ? 'LAB2048_HYBRID' : stage === 'afterstate' ? 'LAB2048_AFTERSTATE' : 'LAB2048_STAGE3_' + stage.toUpperCase()]), JSON.stringify(report));
});

test(`Stage ${stage}: all recorded trajectories replay exactly in the human-play environment`, () => {
  const groups = [report.random.checkpoint, report.random.final];
  const training = [];
  for (const a of Object.values(report.agents)) {
    groups.push(a.final);
    for (const s of a.snapshots) { groups.push(s.evaluation); if (s.training) training.push(s.training); }
  }
  const traces = [...groups.flatMap(g => g.traces), ...training];
  for (const trace of traces) {
    const game = new E.Game(trace.seed);
    assert.deepEqual(game.board, trace.frames[0].board);
    for (const f of trace.frames.slice(1)) {
      const actions = E.legal(game.board);
      assert.deepEqual(actions, f.legal);
      assert.ok(actions.includes(f.action));
      assert.deepEqual(f.key, [2 ** Math.max(...game.board), game.board.filter(v => !v).length]);
      const result = game.step(f.action);
      assert.deepEqual(result.board, f.board);
      assert.deepEqual(result.spawned, f.spawned);
      assert.equal(result.reward, f.reward);
      assert.equal(game.score, f.score);
      assert.equal(game.moves, f.moves);
      assert.ok(f.q.every(Number.isFinite));
    }
    assert.equal(game.score, trace.score);
    assert.equal(game.moves, trace.moves);
    assert.equal(!E.legal(game.board).length, trace.terminated);
    assert.equal(2 ** Math.max(...game.board), trace.maxTile);
  }
});

test(`Stage ${stage}: evaluation means and budgets agree with their source records`, () => {
  for (const agent of Object.values(report.agents)) {
    assert.equal(agent.history.length, report.episodes);
    assert.deepEqual(agent.snapshots.map(s => s.episode), report.checkpoints);
    const metrics = [agent.final, ...agent.snapshots.map(s => s.evaluation)];
    for (const m of metrics) assert.ok(Math.abs(m.mean - m.scores.reduce((a,b) => a+b, 0)/m.scores.length) < 1e-8);
    assert.equal(agent.final.scores.length, report.finalGames);
    assert.ok(agent.final.traces.every(t => t.seed >= report.evaluationSeeds.finalStart));
  }
  if (report.agents.q) assert.equal(report.agents.q.qEntries, report.agents.q.table.length * 4);
  if (report.agents.dqn) assert.equal(report.agents.dqn.parameters, 5508);
  assert.equal(report.agents.cnn.parameters, stage === "hybrid" ? 11633 : stage === "afterstate" ? 10481 : 10676);
});

}
