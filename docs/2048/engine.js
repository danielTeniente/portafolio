/* Shared browser/Node 2048 environment and state encodings. No external dependencies. */
(function (root) {
  'use strict';
  const ACTIONS = ['Up', 'Right', 'Down', 'Left'];
  function rng(seed) {
    let s = seed >>> 0;
    return () => { s += 0x6D2B79F5; let t = s; t = Math.imul(t ^ t >>> 15, t | 1); t ^= t + Math.imul(t ^ t >>> 7, t | 61); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  }
  // Boards store tile exponents: 0 = empty, 1 = 2, 2 = 4, ...
  function slide(board, action) {
    if (!Number.isInteger(action) || action < 0 || action > 3) throw new Error('Invalid action');
    const next = board.slice(); let reward = 0;
    for (let line = 0; line < 4; line++) {
      const ids = Array.from({length: 4}, (_, i) => action === 0 ? i * 4 + line : action === 1 ? line * 4 + 3 - i : action === 2 ? (3 - i) * 4 + line : line * 4 + i);
      const values = ids.map(i => board[i]).filter(Boolean), merged = [];
      for (let i = 0; i < values.length; i++) {
        if (values[i] === values[i + 1]) { merged.push(values[i] + 1); reward += 2 ** (values[i] + 1); i++; }
        else merged.push(values[i]);
      }
      ids.forEach((id, i) => { next[id] = merged[i] || 0; });
    }
    return {board: next, reward, changed: next.some((v, i) => v !== board[i])};
  }
  const legal = board => [0, 1, 2, 3].filter(a => slide(board, a).changed);
  class Game {
    constructor(seed = 42) { this.random = rng(seed); this.board = Array(16).fill(0); this.score = 0; this.moves = 0; this.spawn(); this.spawn(); }
    spawn() {
      const empty = this.board.map((v, i) => v === 0 ? i : -1).filter(i => i >= 0);
      if (!empty.length) return null;
      const index = empty[Math.floor(this.random() * empty.length)], exponent = this.random() < .9 ? 1 : 2;
      this.board[index] = exponent;
      return {index, value: 2 ** exponent};
    }
    step(action) {
      const result = slide(this.board, action);
      let spawned = null;
      if (result.changed) { this.board = result.board; this.score += result.reward; this.moves++; spawned = this.spawn(); }
      return {...result, board: this.board.slice(), spawned, done: legal(this.board).length === 0, reached2048: this.board.some(v => v >= 11)};
    }
  }
  function encode(board, representation = 'exponents') {
    if (representation === 'compact') return {largestTile: Math.max(...board) ? 2 ** Math.max(...board) : 0, emptyCells: board.filter(v => !v).length};
    if (representation === 'values') return board.map(v => v ? 2 ** v : 0);
    if (representation === 'exponents') return board.slice();
    if (representation === 'onehot') {
      // Fixed vocabulary: empty plus exponents 1..17. Reject unsupported custom boards.
      if (board.some(v => v < 0 || v > 17 || !Number.isInteger(v))) throw new Error('One-hot encoding supports ranks 0 through 17');
      return board.map(v => Array.from({length: 18}, (_, i) => Number(i === v)));
    }
    if (representation === 'summary') {
      return {
        emptyCells: board.filter(v => v === 0).length,
        largestTile: Math.max(...board) ? 2 ** Math.max(...board) : 0,
        tileSum: board.reduce((sum, v) => sum + (v ? 2 ** v : 0), 0),
        legalActions: legal(board).map(a => ACTIONS[a])
      };
    }
    throw new Error('Unknown representation');
  }
  const api = {ACTIONS, rng, slide, legal, Game, encode};
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Lab2048 = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
