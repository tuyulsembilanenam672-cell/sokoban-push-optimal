// sokoban-push-optimal: a push-optimal Sokoban solver.
//
// Every level is a one-wide hallway with one to four rooms off it. A box
// pushed out of its room can never come back, so each room is solved on its
// own, push-optimally, and the room solutions are stitched together with
// hallway walks. The score counts pushes only, so the stitching order does
// not matter for the score, only that every room ends push-optimal.
//
// Per room the search is A* over pushes. A state is the sorted box cells
// plus the player cell; a move is one push, and the player's walk to the
// pushing square is a BFS over the room plus the hallway. The heuristic is
// the minimum-cost matching of boxes to goals under shortest-path distances
// (ignoring boxes), which never overestimates the pushes left. States with
// a box on a statically dead square (found by reverse reachability from
// the goals) or inside a 2x2 block of walls and boxes are pruned.
//
// Deterministic: no clock, no randomness, no environment input. All budgets
// are counted in search expansions, so two runs write the same bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// dx, dy, move letter for u, d, l, r.
const DIRS = [[0, -1, 'u'], [0, 1, 'd'], [-1, 0, 'l'], [1, 0, 'r']];
// Work budgets, counted in node expansions so the run is deterministic.
const ROOM_EXPANSION_CAP = 1000000;
const WHOLE_EXPANSION_CAP = 500000;
const TOTAL_EXPANSION_CAP = 8000000;
let expansions = 0;

function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const boxes = [];
  let player = -1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < rows[y].length; x++) {
      const c = rows[y][x];
      const i = y * w + x;
      if (c === '#') continue;
      wall[i] = 0;
      if (c === '.' || c === '*' || c === '+') goal[i] = 1;
      if (c === '$' || c === '*') boxes.push(i);
      if (c === '@' || c === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goal, boxes, player };
}

// Breadth-first search from src over cells where pass(i) is true.
// Returns distances (-1 = unreachable) and parent pointers for paths.
function bfs(src, pass, w, h) {
  const n = w * h;
  const dist = new Int32Array(n).fill(-1);
  const par = new Int32Array(n).fill(-1);
  const q = [src];
  dist[src] = 0;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi];
    const x = c % w;
    const y = (c / w) | 0;
    for (let k = 0; k < 4; k++) {
      const nx = x + DIRS[k][0];
      const ny = y + DIRS[k][1];
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const ni = ny * w + nx;
      if (dist[ni] !== -1 || !pass(ni)) continue;
      dist[ni] = dist[c] + 1;
      par[ni] = c;
      q.push(ni);
    }
  }
  return { dist, par };
}

// Move string from src to dst along BFS parents, or null if unreachable.
function pathTo(par, src, dst, w) {
  if (dst === src) return '';
  const cells = [];
  let c = dst;
  while (c !== -1 && c !== src) {
    cells.push(c);
    c = par[c];
  }
  if (c !== src) return null;
  cells.reverse();
  let out = '';
  let prev = src;
  for (const cell of cells) {
    const d = cell - prev;
    out += d === 1 ? 'r' : d === -1 ? 'l' : d === w ? 'd' : 'u';
    prev = cell;
  }
  return out;
}

// All permutations of [0..n-1], for the matching heuristic (n <= 4).
function permutations(n) {
  const out = [];
  const a = [];
  for (let i = 0; i < n; i++) a.push(i);
  const gen = (k) => {
    if (k === n) {
      out.push(a.slice());
      return;
    }
    for (let i = k; i < n; i++) {
      const t = a[k];
      a[k] = a[i];
      a[i] = t;
      gen(k + 1);
      a[i] = a[k];
      a[k] = t;
    }
  };
  gen(0);
  return out;
}

// Static dead squares for one room: a room cell is dead when no goal can be
// reached from it by pulling the box backwards. Reverse a push of a box at
// c in direction d: the box came from c-d, with the player standing at c-2d.
// A cell no reverse walk from a goal reaches can never hold a box that ends
// on a goal, so any state with a box there (off a goal) is pruned.
function deadSquares(lv, roomCells, roomGoals, walk) {
  const { w, h } = lv;
  const room = new Uint8Array(w * h);
  for (const c of roomCells) room[c] = 1;
  const reach = new Uint8Array(w * h);
  const q = [];
  for (const g of roomGoals) {
    reach[g] = 1;
    q.push(g);
  }
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi];
    const x = c % w;
    const y = (c / w) | 0;
    for (let k = 0; k < 4; k++) {
      const dx = DIRS[k][0];
      const dy = DIRS[k][1];
      const px = x - dx;
      const py = y - dy;
      const sx = x - 2 * dx;
      const sy = y - 2 * dy;
      if (px < 0 || py < 0 || px >= w || py >= h) continue;
      if (sx < 0 || sy < 0 || sx >= w || sy >= h) continue;
      const prev = py * w + px;
      const stand = sy * w + sx;
      if (!room[prev] || !walk[stand] || reach[prev]) continue;
      reach[prev] = 1;
      q.push(prev);
    }
  }
  const dead = new Uint8Array(w * h);
  for (const c of roomCells) if (!reach[c]) dead[c] = 1;
  return dead;
}

// Minimum-cost matching of boxes to goals under shortest-path distances.
// Each push moves one box one cell, so this never overestimates the pushes
// left, and it is consistent because the distances satisfy the triangle
// inequality. Infinity when no box can reach every goal it must cover.
function heuristic(boxes, goals, distMat, perms) {
  let best = Infinity;
  for (const p of perms) {
    let s = 0;
    for (let i = 0; i < boxes.length; i++) {
      const d = distMat[boxes[i]][goals[p[i]]];
      if (d < 0) {
        s = Infinity;
        break;
      }
      s += d;
    }
    if (s < best) best = s;
  }
  return best;
}

// True when a non-goal box sits in a 2x2 block of walls and boxes: the
// block can never be broken, so the state is dead.
function has22Block(boxes, boxSet, goal, wall, w, h) {
  for (const b of boxes) {
    if (goal[b]) continue;
    const x = b % w;
    const y = (b / w) | 0;
    for (let k = 0; k < 4; k++) {
      const ox = k === 1 || k === 3 ? -1 : 0;
      const oy = k === 2 || k === 3 ? -1 : 0;
      const x0 = x + ox;
      const y0 = y + oy;
      if (x0 < 0 || y0 < 0 || x0 + 1 >= w || y0 + 1 >= h) continue;
      const a = y0 * w + x0;
      const c1 = a + 1;
      const c2 = a + w;
      const c3 = a + w + 1;
      if ((wall[a] || boxSet[a]) && (wall[c1] || boxSet[c1]) && (wall[c2] || boxSet[c2]) && (wall[c3] || boxSet[c3])) return true;
    }
  }
  return false;
}

// A* over pushes inside one room (or the whole level, when cells hold every
// floor cell). boxes0/player0 are the start, walk marks every cell the
// player may walk on, and a push may only move a box onto a cell of cells.
// Returns { moves, boxes, player } or null when the cap runs out or the
// room cannot be solved.
function astar(lv, cells, goals, boxes0, player0, walk, cap) {
  const { w, h, wall, goal } = lv;
  const n = boxes0.length;
  const cellSet = new Uint8Array(w * h);
  for (const c of cells) cellSet[c] = 1;
  const goalSet = new Uint8Array(w * h);
  for (const g of goals) goalSet[g] = 1;
  const dead = deadSquares(lv, cells, goals, walk);
  for (const b of boxes0) if (dead[b] && !goalSet[b]) return null;
  // Shortest-path distances between room cells, ignoring boxes.
  const distMat = new Array(w * h);
  for (const c of cells) {
    const r = bfs(c, (i) => cellSet[i] === 1, w, h);
    distMat[c] = r.dist;
  }
  const perms = permutations(n);
  const keyOf = (boxes, player) => {
    let k = player;
    for (let i = n - 1; i >= 0; i--) k = k * 256 + boxes[i];
    return k;
  };
  const solved = (boxes) => {
    for (const b of boxes) if (!goalSet[b]) return false;
    return true;
  };

  // Node storage, kept in parallel arrays to stay small.
  const par = [];
  const seg = [];
  const gg = [];
  const ff = [];
  const sq = [];
  const boxesArr = [];
  const playerArr = [];
  const heap = [];
  let seqNo = 0;
  const hless = (a, b) => {
    if (ff[a] !== ff[b]) return ff[a] < ff[b];
    if (gg[a] !== gg[b]) return gg[a] > gg[b];
    return sq[a] < sq[b];
  };
  const hpush = (i) => {
    heap.push(i);
    let c = heap.length - 1;
    while (c > 0) {
      const p = (c - 1) >> 1;
      if (!hless(heap[c], heap[p])) break;
      const t = heap[c];
      heap[c] = heap[p];
      heap[p] = t;
      c = p;
    }
  };
  const hpop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      let p = 0;
      for (;;) {
        const l = p * 2 + 1;
        const r = l + 1;
        let m = p;
        if (l < heap.length && hless(heap[l], heap[m])) m = l;
        if (r < heap.length && hless(heap[r], heap[m])) m = r;
        if (m === p) break;
        const t = heap[p];
        heap[p] = heap[m];
        heap[m] = t;
        p = m;
      }
    }
    return top;
  };

  const closed = new Map();
  const h0 = heuristic(boxes0, goals, distMat, perms);
  if (h0 === Infinity) return null;
  let roomExpansions = 0;
  par.push(-1);
  seg.push('');
  gg.push(0);
  ff.push(h0);
  sq.push(seqNo++);
  boxesArr.push(boxes0.slice());
  playerArr.push(player0);
  hpush(0);
  closed.set(keyOf(boxes0, player0), 0);
  const boxSet = new Uint8Array(w * h);

  while (heap.length > 0) {
    const u = hpop();
    const boxes = boxesArr[u];
    const player = playerArr[u];
    if (closed.get(keyOf(boxes, player)) < gg[u]) continue;
    if (solved(boxes)) {
      let moves = '';
      let v = u;
      const parts = [];
      while (v !== -1) {
        parts.push(seg[v]);
        v = par[v];
      }
      parts.reverse();
      for (const p of parts) moves += p;
      return { moves, boxes: boxes.slice(), player };
    }
    if (expansions >= TOTAL_EXPANSION_CAP) return null;
    if (roomExpansions >= cap) return null;
    expansions++;
    roomExpansions++;
    // Where the player can walk right now (boxes are obstacles).
    for (const b of boxes) boxSet[b] = 1;
    const reach = bfs(player, (i) => walk[i] === 1 && boxSet[i] !== 1, w, h);
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi];
      const bx = b % w;
      const by = (b / w) | 0;
      for (let k = 0; k < 4; k++) {
        const dx = DIRS[k][0];
        const dy = DIRS[k][1];
        const tx = bx + dx;
        const ty = by + dy;
        const px = bx - dx;
        const py = by - dy;
        if (tx < 0 || ty < 0 || tx >= w || ty >= h) continue;
        if (px < 0 || py < 0 || px >= w || py >= h) continue;
        const target = ty * w + tx;
        const pusher = py * w + px;
        if (!cellSet[target] || boxSet[target]) continue;
        if (reach.dist[pusher] === -1) continue;
        const nb = boxes.slice();
        nb[bi] = target;
        nb.sort((a, c2) => a - c2);
        let bad = false;
        for (const nb2 of nb) if (dead[nb2] && !goalSet[nb2]) {
          bad = true;
          break;
        }
        if (bad) continue;
        boxSet[b] = 0;
        boxSet[target] = 1;
        if (has22Block(nb, boxSet, goal, wall, w, h)) bad = true;
        boxSet[b] = 1;
        boxSet[target] = 0;
        if (bad) continue;
        const hv = heuristic(nb, goals, distMat, perms);
        if (hv === Infinity) continue;
        const g2 = gg[u] + 1;
        const key = keyOf(nb, b);
        const seen = closed.get(key);
        if (seen !== undefined && seen <= g2) continue;
        closed.set(key, g2);
        const walkPath = pathTo(reach.par, player, pusher, w);
        if (walkPath === null) continue;
        const idx = par.length;
        par.push(u);
        seg.push(walkPath + DIRS[k][2]);
        gg.push(g2);
        ff.push(g2 + hv);
        sq.push(seqNo++);
        boxesArr.push(nb);
        playerArr.push(b);
        hpush(idx);
      }
    }
    for (const b of boxes) boxSet[b] = 0;
  }
  return null;
}

// Split a level into its rooms: the hallway is the player's row, and each
// room is one connected component of floor once that row is removed. A room
// holds as many boxes as goals and opens onto the hallway through exactly
// one door cell. Returns null when the level does not fit that shape.
function decompose(lv) {
  const { w, h, wall, goal, boxes, player } = lv;
  const hr = (player / w) | 0;
  const boxSet = new Set(boxes);
  const goalSet = new Set();
  for (let i = 0; i < w * h; i++) if (goal[i]) goalSet.add(i);
  const seen = new Uint8Array(w * h);
  const rooms = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (y === hr || wall[i] || seen[i]) continue;
      const comp = [];
      const q = [i];
      seen[i] = 1;
      while (q.length > 0) {
        const c = q.pop();
        comp.push(c);
        const cx = c % w;
        const cy = (c / w) | 0;
        for (let k = 0; k < 4; k++) {
          const nx = cx + DIRS[k][0];
          const ny = cy + DIRS[k][1];
          if (ny === hr || nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const ni = ny * w + nx;
          if (wall[ni] || seen[ni]) continue;
          seen[ni] = 1;
          q.push(ni);
        }
      }
      const rBoxes = [];
      const rGoals = [];
      for (const c of comp) {
        if (boxSet.has(c)) rBoxes.push(c);
        if (goalSet.has(c)) rGoals.push(c);
      }
      if (rBoxes.length === 0 && rGoals.length === 0) continue;
      if (rBoxes.length !== rGoals.length) return null;
      const doorCells = comp.filter((c) => {
        const cx = c % w;
        const cy = (c / w) | 0;
        return (cy === hr - 1 || cy === hr + 1) && !wall[hr * w + cx];
      });
      if (doorCells.length !== 1) return null;
      const door = doorCells[0];
      rBoxes.sort((a, b) => a - b);
      rooms.push({ cells: comp, boxes: rBoxes, goals: rGoals, door, entry: hr * w + (door % w) });
    }
  }
  const inRoom = new Set();
  for (const r of rooms) for (const c of r.cells) inRoom.add(c);
  if (!boxes.every((b) => inRoom.has(b))) return null;
  for (const g of goalSet) if (!inRoom.has(g)) return null;
  return rooms;
}

// Solve one room push-optimally. The player walks the room plus the whole
// hallway (always clear of boxes) and starts on the hallway cell by the
// door. Boxes may only be pushed onto room cells.
function solveRoom(lv, room) {
  const { w, h, wall } = lv;
  const hr = (lv.player / w) | 0;
  const walk = new Uint8Array(w * h);
  for (const c of room.cells) walk[c] = 1;
  for (let x = 0; x < w; x++) {
    const i = hr * w + x;
    if (!wall[i]) walk[i] = 1;
  }
  const s = astar(lv, room.cells, room.goals, room.boxes, room.entry, walk, ROOM_EXPANSION_CAP);
  if (s === null) return null;
  return { moves: s.moves, boxes: s.boxes, boxes0: room.boxes, player: s.player, door: room.door, entry: room.entry };
}

// Walk the player to a room's entry and play that room's solution, with the
// level's boxes as they stand. Returns the moves, or null when the walk
// cannot reach the entry (for example a solved room's box sits on its door).
function stitch(lv, order) {
  const { w, h, wall } = lv;
  let player = lv.player;
  let boxes = lv.boxes.slice();
  let moves = '';
  for (const s of order) {
    const boxSet = new Set(boxes);
    const reach = bfs(player, (i) => !wall[i] && !boxSet.has(i), w, h);
    const walk = pathTo(reach.par, player, s.entry, w);
    if (walk === null) return null;
    moves += walk;
    moves += s.moves;
    player = s.player;
    const oldSet = new Set(s.boxes0);
    boxes = boxes.filter((b) => !oldSet.has(b)).concat(s.boxes);
    boxes.sort((a, b) => a - b);
  }
  return moves;
}

// All orders of the rooms, rooms whose solution leaves a box on their door
// (trapping the player inside) last.
function orderings(sols) {
  const perms = [];
  const a = sols.slice();
  const gen = (k) => {
    if (k === a.length) {
      perms.push(a.slice());
      return;
    }
    for (let i = k; i < a.length; i++) {
      const t = a[k];
      a[k] = a[i];
      a[i] = t;
      gen(k + 1);
      a[i] = a[k];
      a[k] = t;
    }
  };
  gen(0);
  const blocksDoor = (s) => s.boxes.includes(s.door);
  perms.sort((p, q) => {
    const score = (r) => {
      let v = 0;
      for (let i = 0; i < r.length; i++) if (blocksDoor(r[i])) v += i;
      return v;
    };
    return score(q) - score(p);
  });
  return perms;
}

// Replay moves on the level: true only when every step is legal and every
// box ends on a goal.
function verify(lv, moves) {
  const { w, h, wall, goal } = lv;
  const box = new Uint8Array(w * h);
  for (const b of lv.boxes) box[b] = 1;
  let player = lv.player;
  for (let k = 0; k < moves.length; k++) {
    const m = moves[k];
    let dx = 0;
    let dy = 0;
    if (m === 'u') dy = -1;
    else if (m === 'd') dy = 1;
    else if (m === 'l') dx = -1;
    else if (m === 'r') dx = 1;
    else return false;
    const next = player + dy * w + dx;
    if (next < 0 || next >= w * h || wall[next]) return false;
    if (box[next]) {
      const beyond = next + dy * w + dx;
      if (beyond < 0 || beyond >= w * h || wall[beyond] || box[beyond]) return false;
      box[next] = 0;
      box[beyond] = 1;
    }
    player = next;
  }
  for (let i = 0; i < w * h; i++) if (box[i] && !goal[i]) return false;
  return true;
}

function solveLevel(lv) {
  const rooms = decompose(lv);
  if (rooms !== null) {
    const sols = [];
    for (const room of rooms) {
      const s = solveRoom(lv, room);
      if (s === null) return '';
      sols.push(s);
    }
    for (const order of orderings(sols)) {
      const moves = stitch(lv, order);
      if (moves !== null && verify(lv, moves)) return moves;
    }
    return '';
  }
  // The level does not fit the hallway-and-rooms shape: try the whole level
  // as one search instead.
  const { w, h, wall, goal } = lv;
  const cells = [];
  const goals = [];
  for (let i = 0; i < w * h; i++) {
    if (!wall[i]) cells.push(i);
    if (goal[i]) goals.push(i);
  }
  const walk = new Uint8Array(w * h);
  for (const c of cells) walk[c] = 1;
  const s = astar(lv, cells, goals, lv.boxes, lv.player, walk, WHOLE_EXPANSION_CAP);
  if (s === null) return '';
  return verify(lv, s.moves) ? s.moves : '';
}

for (const id of readdirSync(INPUT).sort()) {
  let moves = '';
  try {
    const level = parseLevel(readFileSync(join(INPUT, id, 'level.txt'), 'utf8'));
    moves = solveLevel(level);
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT, id), { recursive: true });
  writeFileSync(join(OUTPUT, id, 'solution.txt'), moves + '\n');
}
