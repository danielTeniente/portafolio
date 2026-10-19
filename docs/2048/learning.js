/* Offline playback of real training and frozen-policy evaluations. No simulated learning. */
function initializeLearning(prefix, advanced) {
  'use strict';
  const $ = id => document.getElementById(prefix + id) || document.getElementById(id);
  const reports = [window.LAB2048_RESULTS, window.LAB2048_STAGE2, window.LAB2048_AFTERSTATE, window.LAB2048_HYBRID];
  let experiment = advanced ? 3 : 1, data = reports[experiment];
  const stageNames = ['Stage 1 · Original', 'Stage 2 · Snake', 'Stage 3 · Afterstate CNN', 'Stage 4 · CNN + features'];
  const experimentName = () => stageNames[experiment];
  if (!data) { $('learning-status').textContent = 'Results are unavailable. Run python -m src.train_2048 to generate data.js.'; return; }
  const keys = ['q', 'dqn', 'cnn'], names = keys.map(k => reports[0].agents[k].name);
  const colors = ['#286751', '#b47620', '#6963a8'];
  let stage = data.checkpoints.length - 1, cursor = 0, timer = null, journey = false;
  let traces = [], playing = false, playbackVersion = 0, renderVersion = 0, pendingMotion = Promise.resolve();
  const views = keys.map(k => new window.Motion2048.Board($(`replay-${k}`), 'replay-tile'));
  const comparisonCard = $('card-q').cloneNode(true);
  comparisonCard.id = prefix + 'card-original';
  comparisonCard.querySelectorAll('[id]').forEach(node => { node.id = node.id.replace(/-q$/, '-original'); });
  comparisonCard.querySelectorAll('.motion-caption').forEach(node => node.remove());
  comparisonCard.hidden = true;
  $('agent-boards').prepend(comparisonCard);
  const modelTags = keys.map(k => $('card-' + k).querySelector('.model-tag').innerHTML);
  const originalView = new window.Motion2048.Board($('replay-original'), 'replay-tile');
  const comparing = () => $('agent-focus').value !== 'all' && !!reports[0] && !!reports[1];
  const number = value => Number(value).toLocaleString(undefined, {maximumFractionDigits: 1});
  const mode = () => $('replay-mode').value;
  const activeKeys = () => experiment >= 2 ? ['cnn'] : keys;
  const visibleKeys = () => $('agent-focus').value === 'all' ? activeKeys() : [$('agent-focus').value];
  const lastMove = () => Math.max(...traces.map(t => t.trace.frames.length - 1));

  async function renderFrame(animate = false) {
    const version = ++renderVersion, motions = [];
    $('step-replay').disabled = animate || cursor >= lastMove();
    const training = mode() === 'training' && stage > 0;
    traces.forEach(({key, slot, view, reportIndex, trace}) => {
      const at = Math.min(cursor, trace.frames.length - 1), frame = trace.frames[at];
      const finished = cursor >= trace.frames.length - 1;
      if (animate && at > 0 && cursor === at) {
        motions.push(view.animate(trace.frames[at-1].board, frame.board, frame.action, frame.spawned, Number($('replay-speed').value)*.8));
      } else view.show(frame.board, frame.spawned);
      $(`score-${slot}`).textContent = number(frame.score);
      $(`move-${slot}`).textContent = `${frame.moves} moves`;
      $(`action-${slot}`).textContent = frame.action === undefined ? 'Initial board' : `${window.Lab2048.ACTIONS[frame.action]} · +${frame.reward} points${training ? frame.exploratory ? ' · exploration' : ' · greedy' : ''}`;
      const parts = frame.rewardParts;
      $(`reward-${slot}`).textContent = training && frame.action !== undefined ? `Learning reward: ${(parts?.baseReward ?? frame.reward/128).toFixed(3)} base ${parts && parts.shapingBonus < 0 ? '−' : '+'} ${Math.abs(parts?.shapingBonus ?? 0).toFixed(3)} shaping = ${(parts?.learningReward ?? frame.reward/128).toFixed(3)}` : reportIndex ? 'Game score excludes the strategy bonus.' : 'Game score uses original merge points.';
      $(`end-${slot}`).textContent = finished ? trace.truncated ? 'Move limit reached (truncated)' : 'Game over' : 'In progress';
      $(`values-${slot}`).replaceChildren();
      window.Lab2048.ACTIONS.forEach((action, a) => {
        const cell = document.createElement('div'); cell.className = 'q-value' + (frame.action === a ? ' chosen' : '') + (frame.legal && !frame.legal.includes(a) ? ' illegal' : '');
        const label = document.createElement('span'), value = document.createElement('strong');
        label.textContent = action; value.textContent = frame.q ? frame.q[a].toFixed(2) : '—'; cell.append(label, value); $(`values-${slot}`).append(cell);
      });
      if (key === 'q') {
        $(`detail-${slot}`).textContent = frame.key ? `Decision state: largest ${frame.key[0]} · ${frame.key[1]} empty cells` : 'State: largest tile + empty-cell count';
        const update = frame.update;
        $(`update-${slot}`).textContent = training ? update ? `Q ← ${update.before.toFixed(3)} + 0.1 × (${update.target.toFixed(3)} − ${update.before.toFixed(3)}) = ${update.after.toFixed(3)}` : 'No transition yet in this episode.' : 'Evaluation: table values remain frozen.';
      } else {
        $(`detail-${slot}`).textContent = key === 'dqn' ? 'Input: 16 tile exponents ÷ 16' : 'Input: 1 channel × 4 × 4 exponents ÷ 16';
        $(`update-${slot}`).textContent = training ? frame.update ? `Latest replay-batch Huber loss: ${frame.update.loss.toFixed(4)} · update ${frame.update.updates}` : 'No update displayed at this frame.' : 'Evaluation: network weights remain frozen.';
      }
      let preview = $('card-' + slot).querySelector('.afterstate-preview');
      if (!preview) {
        preview = document.createElement('details'); preview.className = 'afterstate-preview';
        $('card-' + slot).append(preview);
      }
      preview.hidden = reportIndex < 2 || key !== 'cnn';
      if (!preview.hidden) {
        $('detail-' + slot).textContent = (reportIndex === 3 ? 'Extra branch: 7 statistics → 16 features, joined with 128 CNN features. ' : '') + 'Input: each pre-spawn 4 × 4 afterstate, exponents ÷ 16. Output: one discounted continuation value.';
        preview.innerHTML = '<summary>Inspect the move previews</summary>' + (frame.decision ? frame.decision.filter(p => p.legal).map(p => {
          const a = frame.decision.indexOf(p);
          return `<p>${window.Lab2048.ACTIONS[a]} · merge ${number(p.immediate*128)} points · move value ${p.actionValue.toFixed(3)}</p>${p.features ? `<p class="feature-values">Scaled features: ${p.features.map((v,i) => ['Empty','Corner','Snake breaks','Merge ↑','Merge →','Merge ↓','Merge ←'][i]+': '+v.toFixed(3)).join(' · ')}</p>` : ''}<div class="preview-grid">${p.board.map(e => `<span>${e ? 2**e : '·'}</span>`).join('')}</div>`;
        }).join('') : '<p>Step forward to inspect the previews used for that move.</p>');
      }
    });
    $('move-scrubber').max = lastMove(); $('move-scrubber').value = cursor;
    $('move-label').textContent = `Move ${cursor} / ${lastMove()}`;
    pendingMotion = Promise.all(motions);
    await pendingMotion;
    if (version === renderVersion) $('step-replay').disabled = cursor >= lastMove();
  }
  function chart() {
    const width = 780, height = 250, left = 60, right = 25, top = 20, bottom = 40;
    const maximum = Math.max(data.random.checkpoint.mean, ...reports.filter(Boolean).flatMap(r => Object.keys(r.agents).flatMap(k => r.agents[k].snapshots.map(s => s.evaluation.mean)))) * 1.15;
    const x = ep => left + ep / Math.max(...reports.filter(Boolean).map(r => r.episodes)) * (width - left - right);
    const y = score => height - bottom - score / maximum * (height - top - bottom);
    let svg = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Frozen-policy mean score across training checkpoints; higher is better"><title>Mean score across ${data.evalGames} fixed evaluation games</title>`;
    for (let i = 0; i <= 4; i++) { const value = maximum * i / 4; svg += `<line x1="${left}" x2="${width-right}" y1="${y(value)}" y2="${y(value)}" stroke="#dce3d8"/><text x="${left-10}" y="${y(value)+4}" text-anchor="end">${Math.round(value)}</text>`; }
    const baseline = y(data.random.checkpoint.mean);
    svg += `<line x1="${left}" x2="${width-right}" y1="${baseline}" y2="${baseline}" stroke="#8b938c" stroke-dasharray="5 5"/>`;
    activeKeys().forEach((key) => {
      const i = keys.indexOf(key);
      if (experiment === 1 || experiment === 3) {
        const reference = reports[experiment >= 2 ? experiment-1 : 0].agents[key].snapshots.filter(p => p.episode <= data.checkpoints[stage]);
        svg += `<polyline points="${reference.map(p => `${x(p.episode)},${y(p.evaluation.mean)}`).join(' ')}" fill="none" stroke="${colors[i]}" stroke-width="1.8" stroke-dasharray="5 4" opacity=".55"/>`;
      }
      const points = data.agents[key].snapshots.slice(0, stage+1);
      svg += `<polyline points="${points.map(p => `${x(p.episode)},${y(p.evaluation.mean)}`).join(' ')}" fill="none" stroke="${colors[i]}" stroke-width="2.5"/>`;
      points.forEach(p => { svg += `<circle cx="${x(p.episode)}" cy="${y(p.evaluation.mean)}" r="4" fill="${colors[i]}"><title>${data.agents[key].name} · episode ${p.episode} · mean ${number(p.evaluation.mean)}</title></circle>`; });
    });
    data.checkpoints.forEach(ep => { svg += `<text x="${x(ep)}" y="${height-20}" text-anchor="middle">${ep}</text>`; });
    svg += `<text x="${width/2}" y="${height-2}" text-anchor="middle">Training episodes</text></svg>`;
    $('learning-chart').innerHTML = svg;
    $('checkpoint-scores').textContent = activeKeys().map(k => `${data.agents[k].name}: ${number(data.agents[k].snapshots[stage].evaluation.mean)}`).join(' · ');
  }
  function stop() { clearTimeout(timer); timer = null; playing = false; playbackVersion++; $('play-replay').textContent = 'Play replay'; $('play-replay').setAttribute('aria-pressed', 'false'); }
  function selectStage(next) {
    stage = next; cursor = 0; $('checkpoint').value = stage;
    const training = mode() === 'training' && stage > 0;
    $('eval-seed').disabled = training;
    const focus = $('agent-focus').value, paired = comparing();
    views.forEach(view => view.cancel()); originalView.cancel();
    const entries = paired
      ? advanced
        ? [{key:'cnn',slot:'original',view:originalView,reportIndex:2}, {key:'cnn',slot:'q',view:views[0],reportIndex:3}]
        : [{key:focus,slot:'original',view:originalView,reportIndex:0}, {key:focus,slot:'q',view:views[0],reportIndex:1}]
      : visibleKeys().map(key => ({key,slot:key,view:views[keys.indexOf(key)],reportIndex:experiment}));
    ['original', ...keys].forEach(slot => { $('card-' + slot).hidden = !entries.some(e => e.slot === slot); });
    $('agent-boards').classList.toggle('stage-pair', paired);
    $('agent-boards').classList.toggle('single-agent', entries.length === 1);
    $('agent-boards').classList.toggle('stage-triple', paired && entries.length === 3);
    entries.forEach(({key,slot,reportIndex}) => {
      const card = $('card-' + slot), i = keys.indexOf(key);
      card.className = 'agent-card' + (key === 'dqn' ? ' dense' : key === 'cnn' ? ' conv' : '');
      card.querySelector('h3').textContent = reports[reportIndex].agents[key].name + (paired ? ' · ' + stageNames[reportIndex] : reports[reportIndex].reusedAgents?.includes(key) ? ' · Stage 2 reference' : '');
      card.querySelector('.model-tag').innerHTML = reportIndex === 3 && key === 'cnn' ? 'Preview grid + seven statistics<br>Two branches → one continuation value' : reportIndex === 2 && key === 'cnn' ? 'One preview per legal move<br>Shared CNN → one continuation value' : modelTags[i];
      $('replay-' + slot).setAttribute('aria-label', reports[reportIndex].agents[key].name + ' ' + stageNames[reportIndex] + ' replay board');
    });
    $('comparison-note').textContent = advanced ? 'Stage 3 afterstate CNN versus Stage 4 hybrid CNN. Same checkpoint and seed, synchronized moves. These controls are independent of Stages 1–2.' : 'Compare the selected model in Stages 1 and 2 with the same checkpoint and seed. These controls are independent of Stages 3–4.';
    const samples = data.agents.cnn.snapshots[stage].evaluation.traces;
    let sampleIndex = Number($('eval-seed').value);
    if ($('eval-seed').value === 'best') {
      const scores = samples.map((sample, i) => Math.max(...entries.map(entry => {
        const snapshot = reports[entry.reportIndex].agents[entry.key].snapshots[stage];
        return snapshot.evaluation.traces[i].frames.at(-1).score;
      })));
      sampleIndex = scores.indexOf(Math.max(...scores));
    }
    traces = entries.map(entry => {
      const snapshot = reports[entry.reportIndex].agents[entry.key].snapshots[stage];
      return {...entry, trace: training ? snapshot.training : snapshot.evaluation.traces[sampleIndex]};
    });
    const epsilon = stage ? data.agents.cnn.history[data.checkpoints[stage]-1].epsilon : null;
    $('learning-status').textContent = stage === 0 ? 'Untrained policies · exploration off · no updates during this replay.' : training ? `Recorded training episode ${data.checkpoints[stage]} · ε = ${epsilon.toFixed(3)} · exploratory moves and real updates included.` : `After ${data.checkpoints[stage]} training episodes · frozen policies · exploration off. These games did not update the agents.`;
    $('learning-provenance').textContent = training ? `Training seed ${traces[0].trace.seed}. Each agent starts from the same board; actions lead to different subsequent spawns. Neural losses refer to the latest replay minibatch, not just the displayed move.` : `Evaluation seed ${traces[0].trace.seed}. All agents start from the same board. Matching random streams do not force matching later tile positions once their boards diverge. Q-values shown belong to the board before the displayed move.`;
    $('learning-status').textContent = (paired ? (advanced ? 'Afterstate CNN vs hybrid CNN · Stages 3–4' : names[keys.indexOf(focus)] + ' · Stage 1 vs Stage 2') : experimentName()) + ' / ' + $('learning-status').textContent;
    $('comparison-note').hidden = !paired;
    chart(); renderFrame();
  }
  function play() {
    if (playing) { stop(); return; }
    if (cursor >= lastMove()) { cursor = 0; renderFrame(); }
    playing = true; const version = ++playbackVersion;
    $('play-replay').textContent = 'Pause replay'; $('play-replay').setAttribute('aria-pressed', 'true');
    async function advance() {
      if (!playing || version !== playbackVersion) return;
      if (cursor < lastMove()) { cursor++; await renderFrame(true); }
      else if (journey && stage < data.checkpoints.length - 1) selectStage(stage+1);
      else { journey = false; stop(); return; }
      if (playing && version === playbackVersion) timer = setTimeout(advance, Number($('replay-speed').value)*.2);
    }
    // Pause/resume never cuts a tile's path short or starts an overlapping move.
    pendingMotion.then(advance);
  }
  function configureReplayOptions() {
    const selectedSeed = $('eval-seed').value || 'best';
    $('checkpoint').innerHTML = data.checkpoints.map((ep, i) => `<option value="${i}">${ep ? (ep === data.episodes ? 'Fully trained · ' : 'After ') + ep + ' episodes' : 'Untrained · 0 episodes'}</option>`).join('');
    const samples = data.agents.cnn.snapshots[0].evaluation.traces;
    $('eval-seed').innerHTML = '<option value="best">Best recorded score</option>' + samples.map((t, i) => `<option value="${i}">Seed ${t.seed}</option>`).join('');
    $('eval-seed').value = selectedSeed === 'best' ? 'best' : Math.min(Number(selectedSeed), samples.length - 1);
  }
  if (advanced) $('agent-focus').innerHTML = '<option value="cnn" selected>Compare Stage 3 vs Stage 4</option><option value="all">Selected CNN stage only</option>';
  configureReplayOptions();
  $('checkpoint').addEventListener('change', () => { stop(); journey = false; selectStage(Number($('checkpoint').value)); });
  ['replay-mode', 'eval-seed'].forEach(id => $(id).addEventListener('change', () => { stop(); journey = false; selectStage(stage); }));
  $('play-replay').addEventListener('click', play);
  $('step-replay').addEventListener('click', () => { stop(); cursor = Math.min(lastMove(), cursor+1); renderFrame(true); });
  $('restart-replay').addEventListener('click', () => { stop(); journey = false; selectStage(stage); });
  $('learning-journey').addEventListener('click', () => { stop(); journey = true; selectStage(0); play(); });
  $('replay-speed').addEventListener('change', () => { if (playing) { stop(); play(); } });
  $('move-scrubber').addEventListener('input', () => { stop(); cursor = Number($('move-scrubber').value); renderFrame(); });
  $('agent-focus').addEventListener('change', () => {
    stop();
    journey = false; selectStage(stage);
  });
  document.addEventListener('visibilitychange', () => { if (document.hidden) stop(); });
  window.addEventListener('pagehide', stop);
  function renderSummary() {
    $('curve-description').textContent = `Mean original score in ${data.evalGames} fixed evaluation games. Exploration and learning are off. Only checkpoints up to your selection are shown.`;
    if (!advanced) {
    const row = (report,index,key) => {
      const a = report.agents[key], f = a.final;
      return `<tr><th scope="row">${stageNames[index]} · ${a.name}</th><td>${number(f.mean)}</td><td>${number(f.sd)}</td><td>${f.wins}/${report.finalGames}</td><td>${number(f.maxTile)}</td><td>${key === 'q' ? a.table.length+' table rows' : number(a.parameters)+' parameters'}</td><td>${number(a.trainingSeconds)} s</td></tr>`;
    };
    $('final-results').innerHTML = reports.slice(0,2).flatMap((r,i) => keys.map(k => row(r,i,k))).join('');
    $('result-caption').textContent = 'Stages 1–2 · three models per stage · 100 matched final games per model';
    $('result-verdict').textContent = 'Original rewards versus snake rewards: all three models trained separately from scratch in each stage. This section always shows both stages, independent of replay selection.';
    $('cnn-final-results').innerHTML = reports.slice(2).flatMap((r,i) => r ? [row(r,i+2,'cnn')] : []).join('');
    if (reports[3]) {
      const a=reports[2].agents.cnn.final,b=reports[3].agents.cnn.final;
      $('cnn-verdict').textContent = `Stage 4 mean ${number(b.mean)} versus Stage 3 ${number(a.mean)} (${b.mean>=a.mean?'+':''}${number(100*(b.mean/a.mean-1))}%). Score SD ${number(a.sd)} → ${number(b.sd)}. Same reward and budget; one training seed does not establish a general winner.`;
    }
    $('run-details').textContent = 'All runs: seed 42, 300 training episodes, 20 games per checkpoint, 100 final games (seeds 9000000–9000099). Scores are original merge points. Random legal-move reference: '+number(reports[0].random.final.mean)+' mean points. Equal episodes do not mean equal compute.';
    const table = reports[experiment === 0 ? 0 : 1].agents.q.table;
    $('q-table').innerHTML = '<table><caption>Stage '+(experiment === 0 ? 1 : 2)+' learned Q-table</caption><thead><tr><th>Largest tile</th><th>Empty cells</th><th>Up</th><th>Right</th><th>Down</th><th>Left</th></tr></thead><tbody>'+table.map(r => `<tr><td>${r.largest}</td><td>${r.empty}</td>${r.q.map(v => `<td>${v.toFixed(3)}</td>`).join('')}</tr>`).join('')+'</tbody></table>';
    }
    $('stage-description').textContent = [
      'Stage 1: original merge reward; three independently trained agents.',
      'Stage 2: snake-shaped reward; three freshly trained agents, same seed and budget.',
      'Stage 3: one afterstate CNN, freshly trained with the snake reward. Compare the two CNN variants in this section.',
      'Stage 4: one hybrid afterstate CNN with a separate seven-statistic branch. Same snake reward and budget as Stage 3. Compare the two CNN variants in this section.'
    ][experiment];
    $('stage-curve-note').textContent = experiment === 1 || experiment === 3 ? `Solid: ${stageNames[experiment]}. Dashed: ${stageNames[experiment-1]}. Same checkpoint seeds and axes.` : advanced ? 'Stage 3 afterstate CNN learning curve.' : 'Stage 1 baseline learning curves.';
    $('chart-legend').textContent = experiment >= 2 ? 'Purple: selected CNN · dashed purple: previous CNN stage · gray dashed: random legal moves' : 'Green: Q-learning · gold: Dense DQN · purple: CNN-DQN · gray dashed: random legal moves';
    $('agent-focus').querySelector('[value="all"]').textContent = advanced ? 'Selected CNN stage only' : 'All three agents';
  }
  $('experiment-stage').innerHTML = reports.map((r,i) => r && (advanced ? i >= 2 : i < 2) ? `<option value="${i}">${stageNames[i]}</option>` : '').join('');
  $('experiment-stage').value = experiment;
  $('experiment-stage').addEventListener('change', () => {
    stop(); journey = false; experiment = Number($('experiment-stage').value); data = reports[experiment];
    stage = Math.min(stage, data.checkpoints.length - 1);
    configureReplayOptions(); renderSummary(); selectStage(stage);
  });
  if (!advanced && reports[1]) {
    $('stage-comparison').innerHTML = keys.map(k => {
      const a=reports[0].agents[k].final, b=reports[1].agents[k].final, delta=b.mean-a.mean;
      return `<tr><th scope="row">${reports[0].agents[k].name}</th><td>${number(a.mean)}</td><td>${number(b.mean)}</td><td class="${delta>=0?'gain':'decline'}">${delta>=0?'+':''}${number(delta)} (${delta>=0?'+':''}${number(100*delta/a.mean)}%)</td><td>${a.wins}/${reports[0].finalGames} → ${b.wins}/${reports[1].finalGames}</td><td>${a.maxTile} → ${b.maxTile}</td></tr>`;
    }).join('');
    $('stage-outcome').textContent = keys.map(k => {const a=reports[0].agents[k].final.mean,b=reports[1].agents[k].final.mean; return `${reports[0].agents[k].name}: ${b>=a?'higher':'lower'} mean score with shaping`;}).join(' · ') + '. These are single-seed, fixed-budget observations, not a guarantee that the strategy helps.';
  }
  function renderOverall() {
    const weight = Number($('stability-weight').value);
    const candidates = reports.flatMap((report, index) => report ? Object.keys(report.agents).filter(key => !report.reusedAgents?.includes(key)).map(key => {
      const agent = report.agents[key], f = agent.final;
      return {name:agent.name, strategy:stageNames[index], mean:f.mean, sd:f.sd, cv:f.sd/f.mean, balanced:f.mean-weight*f.sd};
    }) : []);
    candidates.sort((a,b) => b.balanced-a.balanced || b.mean-a.mean);
    const balanced = candidates[0];
    const highest = candidates.reduce((a,b) => b.mean > a.mean ? b : a);
    const consistent = candidates.reduce((a,b) => b.cv < a.cv ? b : a);
    $('stability-label').textContent = `${weight.toFixed(1)} × score variation`;
    $('overall-picks').innerHTML = [[`Balanced pick · weight ${weight.toFixed(1)}`,balanced],['Highest mean score',highest],['Most consistent relative to score',consistent]].map(([title,c]) => `<article><div class="eyebrow">${title}</div><h3>${c.name}</h3><p>${c.strategy}</p><strong>${number(c.mean)} mean points</strong><p>Score SD: ${number(c.sd)}<br>Relative variation: ${number(100*c.cv)}%<br>Balanced score: ${number(c.balanced)}</p></article>`).join('');
    $('overall-explanation').textContent = `At this setting, ${balanced.name} / ${balanced.strategy} leads on mean − ${weight.toFixed(1)} × SD. Set the slider to 0 to rank purely by average score; increase it to give more weight to consistency. This compares Stages 1–4 regardless of replay selection.`;
    $('overall-ranking').innerHTML = candidates.map((c,i) => `<tr><td>${i+1}</td><th>${c.name}</th><td>${c.strategy}</td><td>${number(c.mean)}</td><td>${number(c.sd)}</td><td>${number(100*c.cv)}%</td><td>${number(c.balanced)}</td></tr>`).join('');
  }
  if (!advanced) {
    $('stability-weight').addEventListener('input', renderOverall);
    renderOverall();
  }
  renderSummary(); selectStage(stage);
}
initializeLearning('', false);
initializeLearning('advanced-', true);
