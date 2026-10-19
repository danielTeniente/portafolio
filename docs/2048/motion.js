/* Preserve each tile's origin: slide first, resolve merges, then reveal the spawn. */
(function (root) {
  'use strict';
  function plan(board, action) {
    if (!Number.isInteger(action) || action < 0 || action > 3) throw new Error('Invalid action');
    const moves = [], merges = [], after = Array(16).fill(0);
    for (let line = 0; line < 4; line++) {
      const ids = Array.from({length: 4}, (_, i) => action === 0 ? i * 4 + line : action === 1 ? line * 4 + 3 - i : action === 2 ? (3 - i) * 4 + line : line * 4 + i);
      const tiles = ids.filter(i => board[i]); let destination = 0;
      for (let i = 0; i < tiles.length; i++) {
        const from = tiles[i], to = ids[destination++], exponent = board[from];
        moves.push({from, to, exponent});
        if (i + 1 < tiles.length && board[tiles[i + 1]] === exponent) {
          moves.push({from: tiles[++i], to, exponent});
          after[to] = exponent + 1;
          merges.push({to, exponent: exponent + 1});
        } else after[to] = exponent;
      }
    }
    return {moves, merges, after};
  }
  const arrows = ['↑', '→', '↓', '←'], directions = ['Up', 'Right', 'Down', 'Left'];
  class Board {
    constructor(node, tileClass) {
      this.node = node; this.tileClass = tileClass; this.version = 0; this.animations = [];
      node.classList.add('motion-board');
      this.caption = document.createElement('p'); this.caption.className = 'motion-caption';
      node.after(this.caption);
    }
    cancel() {
      this.version++;
      this.animations.forEach(a => a.cancel()); this.animations = [];
      this.node.querySelectorAll('.moving-tile').forEach(n => n.remove());
    }
    tile(exponent, index) {
      const tile = document.createElement('div'); tile.className = this.tileClass + ' motion-cell';
      tile.setAttribute('aria-label', `Row ${Math.floor(index / 4) + 1}, column ${index % 4 + 1}: ${exponent ? 2 ** exponent : 'empty'}`);
      if (exponent) {
        tile.textContent = 2 ** exponent;
        tile.style.background = `hsl(${105 - Math.min(exponent, 11) * 6} 38% ${94 - Math.min(exponent, 11) * 5}%)`;
        tile.style.color = exponent > 7 ? '#fff' : '#293b2a';
        if (exponent > 9) tile.style.fontSize = '19px';
      }
      return tile;
    }
    paint(values, spawned) {
      this.node.replaceChildren(...values.map((v, i) => this.tile(v, i)));
      if (spawned) this.node.children[spawned.index].classList.add('new-tile');
    }
    show(values, spawned) {
      this.cancel(); this.paint(values, spawned); this.caption.textContent = 'Board snapshot';
      this.node.dataset.phase = 'idle';
    }
    async animate(before, after, action, spawned, duration = 600) {
      this.cancel(); const version = this.version;
      const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      const path = plan(before, action), direction = `${arrows[action]} ${directions[action]}`;
      if (reduced || !this.node.getBoundingClientRect().width) {
        this.paint(after, spawned); this.caption.textContent = `${direction} · ${path.merges.length} merges${spawned ? ` · new ${spawned.value}` : ''}`;
        this.node.dataset.phase = 'idle'; return;
      }
      const run = async (nodes, frames, ms) => {
        const animations = nodes.map(n => n.animate(frames, {duration: ms, easing: 'ease-in-out', fill: 'forwards'}));
        this.animations.push(...animations);
        await Promise.all(animations.map(a => a.finished.catch(() => {})));
        return version === this.version;
      };
      this.paint(Array(16).fill(0));
      const cells = Array.from(this.node.children), rect = this.node.getBoundingClientRect();
      const positions = cells.map(c => { const r = c.getBoundingClientRect(); return {x:r.left-rect.left, y:r.top-rect.top, width:r.width, height:r.height}; });
      this.node.dataset.phase = 'slide'; this.caption.textContent = `${direction} · tiles slide together`;
      const slides = path.moves.map(move => {
        const start = positions[move.from], end = positions[move.to], tile = this.tile(move.exponent, move.from);
        tile.classList.add('moving-tile'); tile.setAttribute('aria-hidden', 'true');
        Object.assign(tile.style, {left:`${start.x}px`, top:`${start.y}px`, width:`${start.width}px`, height:`${start.height}px`});
        this.node.append(tile);
        const animation = tile.animate([{transform:'translate(0,0)'}, {transform:`translate(${end.x-start.x}px,${end.y-start.y}px)`}], {duration:duration*.55, easing:'cubic-bezier(.25,.7,.3,1)', fill:'forwards'});
        this.animations.push(animation); return animation;
      });
      await Promise.all(slides.map(a => a.finished.catch(() => {})));
      if (version !== this.version) return;
      this.paint(path.after);
      this.node.dataset.phase = 'merge';
      this.caption.textContent = path.merges.length ? `${direction} · ${path.merges.map(m => `${2**(m.exponent-1)} + ${2**(m.exponent-1)} → ${2**m.exponent}`).join(' · ')}` : `${direction} · no merge`;
      // Animate the board invisibly when there is no merge, preserving the phase timing.
      const mergeNodes = path.merges.map(m => this.node.children[m.to]);
      if (!await run(mergeNodes.length ? mergeNodes : [this.node], mergeNodes.length ? [{transform:'scale(1)'}, {transform:'scale(1.13)', boxShadow:'0 0 0 3px #bd8d31'}, {transform:'scale(1)'}] : [{opacity:1},{opacity:1}], duration*.25)) return;
      this.paint(after, spawned); this.node.dataset.phase = 'spawn';
      this.caption.textContent = `${direction} · ${spawned ? `new ${spawned.value} appears` : 'no new tile'}`;
      if (spawned) {
        const tile = this.node.children[spawned.index], badge = document.createElement('small');
        badge.className = 'spawn-label'; badge.textContent = 'NEW'; tile.append(badge);
        if (!await run([tile], [{transform:'scale(.2)',opacity:0},{transform:'scale(1)',opacity:1}], duration*.2)) return;
      }
      this.animations.forEach(a => a.cancel()); this.animations = [];
      this.node.dataset.phase = 'idle';
      this.caption.textContent = `${direction} · ${path.merges.length ? '+'+path.merges.reduce((s,m)=>s+2**m.exponent,0)+' merge points' : 'no merge'}${spawned ? ` · new ${spawned.value}` : ''}`;
    }
  }
  const api = {plan, Board};
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Motion2048 = api;
})(globalThis);
