(function () {
  'use strict';
  const E = window.Lab2048, $ = id => document.getElementById(id);
  let game = new E.Game(42), representation = 'exponents', busy = false, moveGeneration = 0;
  const gameView = new window.Motion2048.Board($('board'), 'tile');
  const descriptions = {
    values: ['A / Raw tile values', 'Empty cells are 0; other cells contain 2, 4, 8, 16 and so on. This is the most direct, lossless description of the board.', 'Useful for inspecting the rules. For later neural inputs, large tile values would need careful scaling.'],
    exponents: ['B / Tile exponents', 'Empty cells are 0. A tile of 2 is 1, 4 is 2, 8 is 3, and so on. Every cell keeps its position. Both neural agents receive these exponents divided by 16.', 'Dense DQN uses a 16-number vector. CNN-DQN uses a single-channel 4 × 4 grid. Neither uses one-hot encoding.'],
    compact: ['C / Largest tile + empty cells', 'The tabular agent uses exactly these two numbers as its state key. Boards with the same largest tile and empty-cell count share four Q-values, even if their tiles are arranged differently.', 'Small table, lossy observation. Legal-action masking comes from the environment; it is not part of the table key.'],
    summary: ['D / Board summary', 'Describe the board using a handful of statistics. This is readable and compact, but many different boards can share the same summary.', 'Lossy: these features alone cannot determine transition probabilities and rewards. Treat them as an observation or additional features, not an equivalent full state.']
  };
  function renderState() {
    const [title, copy, use] = descriptions[representation];
    $('state-title').textContent = title; $('state-copy').textContent = copy; $('state-use').textContent = use;
    const encoded = E.encode(game.board, representation);
    if (representation === 'summary' || representation === 'compact') $('state-output').textContent = JSON.stringify(encoded, null, 2);
    else $('state-output').textContent = '[\n' + Array.from({length: 4}, (_, row) => '  [' + encoded.slice(row * 4, row * 4 + 4).map(v => String(v).padStart(representation === 'values' ? 6 : 2)).join(', ') + ']').join(',\n') + '\n]';
  }
  function render(before, result, action, draw = true) {
    if (draw) gameView.show(game.board, result?.spawned);
    const actions = E.legal(game.board), won = game.board.some(v => v >= 11);
    $('score').textContent = game.score; $('moves').textContent = game.moves;
    $('action').textContent = action === undefined ? '—' : E.ACTIONS[action] + (result.changed ? '' : ' (no change)');
    $('reward').textContent = result?.reward || 0;
    $('spawn').textContent = result ? (result.spawned ? `${result.spawned.value} at row ${Math.floor(result.spawned.index / 4) + 1}, column ${result.spawned.index % 4 + 1}` : 'None') : 'Two tiles at reset';
    $('legal').textContent = actions.map(a => E.ACTIONS[a]).join(' · ') || 'None';
    $('terminal').textContent = actions.length ? 'No' : 'Yes — no legal moves';
    $('status').textContent = !actions.length ? `Game over. Final score: ${game.score}. Start a new game to play again.` : won ? '2048 reached! You can keep playing until no moves remain.' : result && !result.changed ? 'That direction cannot change the board. No tile added, no points earned.' : result ? `${E.ACTIONS[action]} · +${result.reward} points. The outlined tile is the new spawn.` : 'Ready. Merge equal tiles and try to reach 2048.';
    document.querySelectorAll('[data-action]').forEach(b => { b.disabled = !actions.includes(Number(b.dataset.action)); });
    $('transition-json').textContent = JSON.stringify({state: E.encode(before || game.board), action: action === undefined ? null : E.ACTIONS[action], nextState: E.encode(game.board), reward: result?.reward || 0, changed: result?.changed ?? false, spawned: result?.spawned || null, terminal: !actions.length, reached2048: won}, null, 2);
    renderState();
  }
  async function move(action) {
    if (busy) return;
    const before = game.board.slice(), result = game.step(action), generation = ++moveGeneration;
    if (!result.changed) { render(before, result, action); return; }
    busy = true;
    document.querySelectorAll('[data-action]').forEach(b => { b.disabled = true; });
    await gameView.animate(before, result.board, action, result.spawned, 480);
    if (generation !== moveGeneration) return;
    busy = false; render(before, result, action, false);
  }
  document.querySelectorAll('[data-action]').forEach(b => b.addEventListener('click', () => move(Number(b.dataset.action))));
  $('board').addEventListener('keydown', event => { const keys = {ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3, w: 0, d: 1, s: 2, a: 3}; const action = keys[event.key.length === 1 ? event.key.toLowerCase() : event.key]; if (action !== undefined) { event.preventDefault(); move(action); } });
  let touchStart = null;
  $('board').addEventListener('pointerdown', event => { touchStart = [event.clientX, event.clientY]; $('board').setPointerCapture(event.pointerId); $('board').focus(); });
  $('board').addEventListener('pointercancel', () => { touchStart = null; });
  $('board').addEventListener('pointerup', event => { if (!touchStart) return; const dx = event.clientX - touchStart[0], dy = event.clientY - touchStart[1]; touchStart = null; if (Math.max(Math.abs(dx), Math.abs(dy)) > 25) move(Math.abs(dx) > Math.abs(dy) ? dx > 0 ? 1 : 3 : dy > 0 ? 2 : 0); });
  $('reset-form').addEventListener('submit', event => { event.preventDefault(); if (!$('reset-form').reportValidity()) return; moveGeneration++; busy = false; game = new E.Game(Number($('seed').value)); render(); });
  document.querySelectorAll('[data-state]').forEach(b => b.addEventListener('click', () => { representation = b.dataset.state; document.querySelectorAll('[data-state]').forEach(other => other.setAttribute('aria-pressed', String(other === b))); renderState(); }));
  render();
})();
