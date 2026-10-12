// Adapted from the RoboLab dashboard (https://github.com/NVlabs/RoboLab, dashboard/static/app.js).
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const state = { runs: [], expanded: new Set(), route: {}, tab: 'Overview', showEmpty: false };

const $ = (sel) => document.querySelector(sel);
const fmtPct = (r) => (r == null || !Number.isFinite(r) ? '—' : `${(r * 100).toFixed(1)}%`);
const fmtNum = (v, d = 2) => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(d));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(node.style, v);
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return node;
}

const fetchJSON = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
};

const chip = (text, cls = '') => el('span', { class: `chip ${cls}` }, text);
const badge = (ok) => chip(ok ? 'success' : 'fail', ok ? 'success' : 'fail');
const runSystem = (run) => (run.system ? `${run.model} · ${run.system}` : run.model);
const bar = (rate) => el('div', {
  class: 'bar', style: { width: '110px' },
  title: fmtPct(rate),
}, el('div', { class: 'fill', style: { width: `${Math.max(0, Math.min(1, rate || 0)) * 100}%` } }));

function wilson(k, n) {
  if (!n) return [0, 1];
  const z = 1.96, p = k / n, d = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / d;
  const margin = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [Math.max(0, centre - margin), Math.min(1, centre + margin)];
}

function srCell(successes, episodes) {
  if (!episodes) return el('span', { style: { color: 'var(--text-2)' } }, '—');
  const [lo, hi] = wilson(successes, episodes);
  return el('span', {
    class: 'val-with-ci',
    title: `${successes}/${episodes} · 95% CI [${(lo * 100).toFixed(1)}–${(hi * 100).toFixed(1)}]`,
  },
    el('span', { class: 'val' }, fmtPct(successes / episodes)),
    el('span', { class: 'ci' }, `±${(((hi - lo) / 2) * 100).toFixed(1)}`));
}

const METRIC_GROUPS = [
  ['Outcome', ['success', 'success_terminal', 'any_goal', 'goal_progress', 'goal_progress_best',
    'goal_regression', 'order_violation', 'recovery_rate']],
  ['Failure mode', ['failure_', 'cause_']],
  ['Objects', ['grasp_', 'lost_', 'goal_knocked', 'object_fell', 'misplaced_']],
  ['Control', ['verifier_', 'verify_', 'premature_switch', 'abandoned_goal', 'switch_lag',
    'instruction_churn', 'advance_', 'verdict_']],
  ['Plan', ['plan_']],
  ['Timing', ['success_steps', 'first_grasp_steps', 'steps_per_goal', 'idle_tail', 'dead_time']],
];

function metricPanel(metrics) {
  const left = new Map(Object.entries(metrics || {}));
  const groups = [];
  for (const [title, keys] of METRIC_GROUPS) {
    const rows = [];
    for (const [name, value] of [...left]) {
      if (keys.some((k) => (k.endsWith('_') ? name.startsWith(k) : name === k))) {
        rows.push([name, value]);
        left.delete(name);
      }
    }
    if (rows.length) groups.push([title, rows.sort()]);
  }
  if (left.size) groups.push(['Other', [...left].sort()]);
  if (!groups.length) return el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'No metrics.');
  const host = el('div', { class: 'grid gap-4', style: { gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))' } });
  for (const [title, rows] of groups) {
    const panel = el('div', { class: 'panel' }, el('div', { class: 'lang-label mb-2' }, title));
    for (const [name, value] of rows) {
      const [n, d] = String(value).split('/').map(Number);
      panel.appendChild(el('div', { class: 'flex items-baseline justify-between gap-2 py-0.5' },
        el('span', { class: 'text-xs', style: { color: 'var(--text-1)' } }, name),
        el('span', { class: 'val-with-ci' },
          el('span', { class: 'val text-xs' }, fmtNum(d ? n / d : null)),
          el('span', { class: 'ci' }, value))));
    }
    host.appendChild(panel);
  }
  return host;
}

function campaigns() {
  const grouped = new Map();
  for (const run of state.runs) {
    if (!state.showEmpty && !run.scored) continue;
    if (!grouped.has(run.campaign)) grouped.set(run.campaign, []);
    grouped.get(run.campaign).push(run);
  }
  return [...grouped].sort((a, b) => (a[0] < b[0] ? -1 : 1));
}

const emptyToggle = (onchange) => {
  const host = el('label', { class: 'text-xs', style: { color: 'var(--text-2)', cursor: 'pointer' } });
  const box = el('input', { type: 'checkbox', title: 'also show campaigns without tallies' });
  box.checked = state.showEmpty;
  box.addEventListener('change', () => { state.showEmpty = box.checked; onchange(); });
  host.append(box, ' show campaigns without results');
  return host;
};

function renderSidebar() {
  const host = $('#sidebar');
  host.innerHTML = '';
  host.appendChild(el('div', { class: 'p-3' }, emptyToggle(() => { renderSidebar(); render(); })));
  if (!state.runs.length) {
    host.appendChild(el('div', { class: 'p-3 text-xs', style: { color: 'var(--text-2)' } }, 'No runs found.'));
    return;
  }
  for (const [name, runs] of campaigns()) {
    const open = state.expanded.has(name);
    host.appendChild(el('div', {
      class: `tree-row px-3 py-1 flex items-center gap-1 ${state.route.campaign === name ? 'selected' : ''}`,
      title: `${runs.length} runs · ${[...new Set(runs.map((r) => r.suite))].join(', ')}`,
      onclick: () => {
        if (open) state.expanded.delete(name); else state.expanded.add(name);
        go(`#/campaign/${encodeURIComponent(name)}`);
        renderSidebar();
      },
    },
      el('span', { style: { color: 'var(--text-2)', width: '10px' } }, open ? '▾' : '▸'),
      el('span', { class: 'flex-1 truncate' }, name),
      el('span', { class: 'text-xs', style: { color: 'var(--text-2)' } }, String(runs.length))));
    if (!open) continue;
    for (const run of runs) {
      host.appendChild(el('div', {
        class: `tree-row pl-6 pr-3 py-1 text-xs truncate ${state.route.runId === run.run_id ? 'selected' : ''}`,
        title: run.run_id,
        onclick: (e) => { e.stopPropagation(); go(`#/run/${encodeURIComponent(run.run_id)}`); },
      }, runSystem(run)));
    }
  }
}

function go(hash) {
  if (location.hash === hash) render();
  else location.hash = hash;
}

function parseRoute() {
  const [name, ...rest] = location.hash.replace(/^#\/?/, '').split('/').map(decodeURIComponent);
  const campaign = rest[0] ? rest[0].split('/')[0] : undefined;
  if (name === 'campaign' && rest[0]) return { name, campaign: rest[0] };
  if (name === 'run' && rest[0]) return { name, runId: rest[0], campaign };
  if (name === 'unit' && rest[1]) return { name, runId: rest[0], unit: rest[1], campaign };
  if (name === 'episode' && rest[2] != null) {
    return { name, runId: rest[0], unit: rest[1], index: Number(rest[2]), campaign };
  }
  return { name: 'home' };
}

const link = (text, hash) => el('a', { class: 'hover:underline cursor-pointer', onclick: () => go(hash) }, text);

function setBreadcrumb(...parts) {
  const host = $('#breadcrumb');
  host.innerHTML = '';
  parts.filter(Boolean).forEach((part, i) => {
    if (i) host.appendChild(el('span', { style: { color: 'var(--text-2)' } }, ' / '));
    host.appendChild(typeof part === 'string' ? el('span', {}, part) : part);
  });
}

function trail(runId, unit) {
  const [campaign] = runId.split('/');
  const tail = runId.split('/').slice(2).join(' · ');
  return [
    link('Campaigns', '#/'),
    link(campaign, `#/campaign/${encodeURIComponent(campaign)}`),
    link(tail, `#/run/${encodeURIComponent(runId)}`),
    unit ? link(unit, `#/unit/${encodeURIComponent(runId)}/${encodeURIComponent(unit)}`) : null,
  ];
}

async function render() {
  state.route = parseRoute();
  const pane = $('#pane');
  pane.innerHTML = '<div style="color: var(--text-2);">Loading…</div>';
  const r = state.route;
  try {
    if (r.name === 'campaign') await renderCampaign(r.campaign);
    else if (r.name === 'run') await renderRun(r.runId);
    else if (r.name === 'unit') await renderUnit(r.runId, r.unit);
    else if (r.name === 'episode') await renderEpisode(r.runId, r.unit, r.index);
    else renderHome();
  } catch (error) {
    pane.innerHTML = '';
    pane.appendChild(el('div', { style: { color: 'var(--fail)' } }, String(error.message || error)));
  }
}

function renderHome() {
  setBreadcrumb('Campaigns');
  const pane = $('#pane');
  pane.innerHTML = '';
  const empty = state.runs.filter((r) => !r.scored).length;
  pane.appendChild(el('div', {
    title: `${empty} run(s) without tallies are hidden behind the sidebar's toggle`,
    class: 'flex items-center gap-2 mb-4',
  }, el('h2', { class: 'text-lg font-semibold' }, 'Campaigns'), emptyToggle(() => render())));
  const table = el('table', { class: 'w-full text-sm' }, el('thead', {}, el('tr', { style: { color: 'var(--text-2)' } },
    el('th', { class: 'px-3 py-2 text-left' }, 'Campaign'),
    el('th', { class: 'px-3 py-2 text-left' }, 'Suites'),
    el('th', { class: 'px-3 py-2 text-right' }, 'Runs'),
    el('th', { class: 'px-3 py-2 text-left' }, 'Models'),
    el('th', { class: 'px-3 py-2 text-left' }, 'Systems'))));
  const body = el('tbody');
  for (const [name, runs] of campaigns()) {
    const uniq = (key) => [...new Set(runs.map((r) => r[key]))].sort();
    body.appendChild(el('tr', {
      class: 'tbl-row tbl-row-clickable cursor-pointer',
      onclick: () => go(`#/campaign/${encodeURIComponent(name)}`),
    },
      el('td', { class: 'px-3 py-2 font-mono text-xs' }, name),
      el('td', { class: 'px-3 py-2' }, ...uniq('suite').map((s) => chip(s, 'accent'))),
      el('td', { class: 'px-3 py-2 text-right' }, String(runs.length)),
      el('td', { class: 'px-3 py-2 text-xs', style: { color: 'var(--text-1)' } }, uniq('model').join(', ')),
      el('td', { class: 'px-3 py-2 text-xs', style: { color: 'var(--text-1)' } }, uniq('system').join(', '))));
  }
  table.appendChild(body);
  pane.appendChild(table);
}

async function renderCampaign(name) {
  const runs = state.runs.filter((r) => r.campaign === name);
  const loaded = await Promise.all(runs.map((r) => fetchJSON(`/api/units?run=${encodeURIComponent(r.run_id)}`)));
  setBreadcrumb(link('Campaigns', '#/'), name);
  const pane = $('#pane');
  pane.innerHTML = '';
  pane.appendChild(el('h2', { class: 'text-lg font-semibold mb-4' }, name));

  const table = el('table', { class: 'w-full text-sm mb-8' }, el('thead', {}, el('tr', { style: { color: 'var(--text-2)' } },
    el('th', { class: 'px-3 py-2 text-left' }, 'Benchmark'),
    el('th', { class: 'px-3 py-2 text-left' }, 'Model'),
    el('th', { class: 'px-3 py-2 text-left' }, 'System'),
    el('th', { class: 'px-3 py-2 text-right' }, 'Units'),
    el('th', { class: 'px-3 py-2 text-right' }, 'Episodes'),
    el('th', { class: 'px-3 py-2 text-right' }, 'Success'),
    el('th', { class: 'px-3 py-2 text-right' }, 'Goals'),
    el('th', { class: 'px-3 py-2' }))));
  const body = el('tbody');
  loaded.forEach((data) => {
    const run = data.run;
    body.appendChild(el('tr', {
      class: 'tbl-row tbl-row-clickable cursor-pointer',
      onclick: () => go(`#/run/${encodeURIComponent(run.run_id)}`),
    },
      el('td', { class: 'px-3 py-2 text-xs' }, run.benchmark),
      el('td', { class: 'px-3 py-2' }, run.model),
      el('td', { class: 'px-3 py-2' }, run.system),
      el('td', { class: 'px-3 py-2 text-right' }, String(data.units)),
      el('td', { class: 'px-3 py-2 text-right' }, String(data.episodes)),
      el('td', { class: 'px-3 py-2 text-right' }, srCell(data.successes, data.episodes)),
      el('td', { class: 'px-3 py-2 text-right' }, fmtNum(data.goal_progress)),
      el('td', { class: 'px-3 py-2' }, bar(data.success_rate))));
  });
  table.appendChild(body);
  pane.appendChild(table);

  const units = [...new Set(loaded.flatMap((d) => d.unit_rows.map((u) => u.unit)))].sort();
  if (!units.length) return;
  pane.appendChild(el('div', { class: 'lang-label mb-2' }, 'Success rate per unit'));
  const matrix = el('table', { class: 'matrix' }, el('thead', {}, el('tr', {},
    el('th', { class: 'label-col' }, 'Unit'),
    ...loaded.map((d) => el('th', { class: 'cell-num', title: d.run.run_id }, `${d.run.model}/${d.run.system}`)))));
  const rows = el('tbody');
  for (const unit of units) {
    rows.appendChild(el('tr', {},
      el('td', { class: 'label-col' }, el('span', { class: 'label-cell', title: unit }, unit)),
      ...loaded.map((d) => {
        const cell = d.unit_rows.find((u) => u.unit === unit);
        return el('td', {
          class: 'cell-num cursor-pointer',
          onclick: () => cell && go(`#/unit/${encodeURIComponent(d.run.run_id)}/${encodeURIComponent(unit)}`),
        }, cell ? srCell(cell.successes, cell.episodes) : '—');
      })));
  }
  matrix.appendChild(rows);
  pane.appendChild(el('div', { style: { overflowX: 'auto' } }, matrix));
}

async function renderRun(runId) {
  const data = await fetchJSON(`/api/units?run=${encodeURIComponent(runId)}`);
  const run = data.run;
  setBreadcrumb(...trail(runId).slice(0, 3));
  const pane = $('#pane');
  pane.innerHTML = '';
  pane.appendChild(el('div', { class: 'flex items-center gap-2 flex-wrap mb-1' },
    el('h2', { class: 'text-lg font-semibold' }, runSystem(run)),
    chip(run.benchmark, 'accent'), chip(`${data.episodes} episodes`), chip(`${data.units} units`)));
  pane.appendChild(el('div', { class: 'text-xs font-mono mb-4', style: { color: 'var(--text-2)' } },
    run.paths.join('  ')));

  if (data.provenance.length) {
    const panel = el('div', { class: 'panel mb-6 text-xs' }, el('div', { class: 'lang-label mb-2' }, 'Provenance'));
    for (const cell of data.provenance) {
      panel.appendChild(el('div', { class: 'flex flex-wrap gap-x-4 gap-y-1 py-1 font-mono', style: { color: 'var(--text-1)' } },
        ...Object.entries(cell).map(([k, v]) => el('span', {},
          el('span', { style: { color: 'var(--text-2)' } }, `${k} `), v))));
    }
    pane.appendChild(panel);
  }

  const table = el('table', { class: 'w-full text-sm mb-8' }, el('thead', {}, el('tr', { style: { color: 'var(--text-2)' } },
    el('th', { class: 'px-3 py-2 text-left' }, 'Unit'),
    el('th', { class: 'px-3 py-2 text-left' }, 'Env'),
    el('th', { class: 'px-3 py-2 text-right' }, 'Episodes'),
    el('th', { class: 'px-3 py-2 text-right' }, 'Success'),
    el('th', { class: 'px-3 py-2 text-right' }, 'Goals'),
    el('th', { class: 'px-3 py-2' }))));
  const body = el('tbody');
  for (const unit of data.unit_rows) {
    body.appendChild(el('tr', {
      class: 'tbl-row tbl-row-clickable cursor-pointer',
      onclick: () => go(`#/unit/${encodeURIComponent(runId)}/${encodeURIComponent(unit.unit)}`),
    },
      el('td', { class: 'px-3 py-2' }, unit.unit),
      el('td', { class: 'px-3 py-2 text-xs font-mono', style: { color: 'var(--text-2)' } }, unit.env_name),
      el('td', { class: 'px-3 py-2 text-right' }, unit.scored ? String(unit.episodes) : '—'),
      el('td', { class: 'px-3 py-2 text-right' }, unit.scored
        ? srCell(unit.successes, unit.episodes)
        : el('span', { style: { color: 'var(--text-2)' }, title: 'episodes recorded, no tally written' }, 'not scored')),
      el('td', { class: 'px-3 py-2 text-right' }, fmtNum(unit.goal_progress)),
      el('td', { class: 'px-3 py-2' }, unit.scored ? bar(unit.success_rate) : null)));
  }
  table.appendChild(body);
  pane.appendChild(table);

  if (data.unit_rows.length) {
    pane.appendChild(el('div', { class: 'lang-label mb-2' }, 'Metrics over the run'));
    pane.appendChild(metricPanel(data.metrics));
  } else {
    pane.appendChild(el('div', { class: 'panel text-sm', style: { color: 'var(--text-2)' } },
      'This run wrote no results; the logs below are what its cells left.'));
  }
  pane.appendChild(logPanel(data.logs, { run: runId }));
}

function logPanel(logs, params) {
  const host = el('div', {});
  if (!logs || !logs.length) return host;
  host.appendChild(el('div', { class: 'lang-label mb-2 mt-6' }, 'Logs'));
  const tabs = el('div', { class: 'flex gap-1 mb-2 flex-wrap', style: { borderBottom: '1px solid var(--border)' } });
  const body = el('pre', {
    class: 'panel text-xs overflow-auto font-mono',
    style: { maxHeight: '340px', whiteSpace: 'pre', color: 'var(--text-1)' },
  }, 'Pick a log.');
  for (const log of logs) {
    tabs.appendChild(el('div', {
      class: 'tab', title: log.path,
      onclick: async (event) => {
        for (const node of tabs.children) node.classList.remove('active');
        event.currentTarget.classList.add('active');
        body.textContent = 'Loading…';
        body.textContent = await (await fetch(`/api/log?${new URLSearchParams({ ...params, path: log.path })}`)).text();
        body.scrollTop = body.scrollHeight;
      },
    }, log.name));
  }
  host.append(tabs, body);
  return host;
}

const goalRibbon = (goals) => (!goals || !goals.length ? null : el('div', {
  class: 'flex gap-0.5',
  title: `${goals.filter(Boolean).length}/${goals.length} goals at the end`,
}, ...goals.map((done) => el('div', {
  style: { width: '10px', height: '5px', borderRadius: '1px', background: done ? 'var(--success)' : 'var(--border-strong)' },
}))));

async function renderUnit(runId, unit) {
  const data = await fetchJSON(`/api/episodes?run=${encodeURIComponent(runId)}&unit=${encodeURIComponent(unit)}`);
  setBreadcrumb(...trail(runId).slice(0, 3), unit);
  const pane = $('#pane');
  pane.innerHTML = '';
  const s = data.summary;
  pane.appendChild(el('div', { class: 'flex items-center gap-2 flex-wrap mb-4' },
    el('h2', { class: 'text-lg font-semibold' }, unit),
    s && s.env_name ? chip(s.env_name, 'accent') : null,
    s && s.robot ? chip(s.robot) : null,
    s && s.scored ? chip(`${s.successes}/${s.episodes} success`, s.successes ? 'success' : 'fail')
                  : chip(`${data.episodes.length} episodes, not scored`)));

  const first = data.episodes.find((e) => e.instruction);
  if (first) {
    pane.appendChild(el('div', { class: 'mb-4' },
      el('div', { class: 'lang-label mb-1' }, 'Instruction'),
      el('div', { class: 'text-base' }, first.instruction)));
  }

  const grid = el('div', { class: 'grid gap-3 mb-8', style: { gridTemplateColumns: 'repeat(auto-fill, minmax(230px, 1fr))' } });
  for (const e of data.episodes) {
    grid.appendChild(el('div', {
      class: 'ep-card panel cursor-pointer',
      onclick: () => go(`#/episode/${encodeURIComponent(runId)}/${encodeURIComponent(unit)}/${e.index}`),
    },
      el('div', { class: 'flex items-center justify-between mb-2' },
        el('span', { class: 'text-xs font-mono', style: { color: 'var(--text-1)' } },
          `#${e.index} · batch ${e.batch} · env ${e.env_index}`),
        badge(e.success)),
      goalRibbon(e.goals),
      el('div', { class: 'flex items-center gap-2 mt-2 text-xs', style: { color: 'var(--text-2)' } },
        el('span', {}, `${e.num_steps} steps`),
        el('span', {}, e.videos.length ? e.videos.map((v) => v.name).join(' + ') : 'no video'),
        e.events.length ? el('span', {}, `${e.events.length} events`) : null,
        e.events.some((x) => x.kind === 'code') ? el('span', {}, `${e.events.filter((x) => x.kind === 'code').length} programs`) : null),
      e.system ? el('div', { class: 'mt-2 text-xs', style: { color: 'var(--text-2)' } }, e.system) : null));
  }
  pane.appendChild(grid);

  if (s && s.scored) {
    pane.appendChild(el('div', { class: 'lang-label mb-2' }, 'Metrics over the unit'));
    pane.appendChild(metricPanel(s.metrics));
  }
  pane.appendChild(logPanel(data.logs, { run: runId, unit }));
}

function seekAll(videos, seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return;
  for (const v of videos) {
    try {
      v.currentTime = Math.max(0, seconds);
      if (v.paused) v.play().catch(() => {});
    } catch { /* not ready yet */ }
  }
}

async function renderEpisode(runId, unit, index) {
  const query = `run=${encodeURIComponent(runId)}&unit=${encodeURIComponent(unit)}&index=${index}`;
  const data = await fetchJSON(`/api/episode?${query}`);
  const e = data.episode;
  setBreadcrumb(...trail(runId, unit), `episode ${index}`);
  const pane = $('#pane');
  pane.innerHTML = '';
  const programs = (data.events || []).filter((x) => x.kind === 'code' && x.code).length;
  pane.appendChild(el('div', { class: 'flex items-center gap-2 flex-wrap mb-4' },
    chip(`#${index} · batch ${e.batch} · env ${e.env_index}`, 'accent'),
    badge(e.success),
    chip(`${e.num_steps} steps`),
    e.goals.length ? chip(`${e.goals.filter(Boolean).length}/${e.goals.length} goals`) : null,
    programs ? chip(`${programs} generated program${programs > 1 ? 's' : ''}`, 'accent') : null,
    e.system ? chip(e.system) : null,
    data.failure_cause ? chip(`cause: ${data.failure_cause.replace(/_/g, ' ')}`, 'fail') : null));
  pane.appendChild(el('div', { class: 'mb-5' },
    el('div', { class: 'lang-label mb-1' }, 'Task instruction'),
    el('div', { class: 'text-base' }, e.instruction || '—')));

  const videos = [];
  if (e.videos.length) {
    const grid = el('div', { class: 'cam-grid mb-5' });
    for (const camera of e.videos) {
      const host = el('div', { class: 'cam-tile-video-wrap video-host is-loading' });
      const video = el('video', {
        src: `/api/video?${query}&name=${encodeURIComponent(camera.name)}`,
        autoplay: '', muted: '', loop: '', playsinline: '', controls: '',
      });
      video.addEventListener('canplay', () => host.classList.remove('is-loading'), { once: true });
      video.addEventListener('error', () => host.classList.replace('is-loading', 'is-failed'), { once: true });
      videos.push(video);
      host.appendChild(video);
      grid.appendChild(el('div', { class: 'cam-tile' },
        el('div', { class: 'cam-tile-label' }, el('span', { class: 'dot' }), camera.name), host));
    }
    pane.appendChild(grid);
  } else {
    pane.appendChild(el('div', { class: 'panel mb-5 text-sm', style: { color: 'var(--text-2)' } },
      'This episode was run with video disabled.'));
  }

  const clock = { duration: null, steps: Math.max(1, e.num_steps), ready: [] };
  const toTime = (step) => (clock.duration == null ? null : (step / clock.steps) * clock.duration);
  if (videos.length) {
    videos[0].addEventListener('loadedmetadata', () => {
      if (!Number.isFinite(videos[0].duration)) return;
      clock.duration = videos[0].duration;
      clock.ready.forEach((fn) => fn());
    }, { once: true });
  }

  pane.appendChild(renderCommands(e, videos, clock, toTime));
  pane.appendChild(renderEvents(data.events, e, videos, clock, toTime));
  pane.appendChild(episodeTabs(e, data, { runId, unit, index, query, videos, clock }));
}

function episodeTabs(e, data, ctx) {
  const tabs = ['Overview', 'Metrics', 'Traces'];
  const active = tabs.includes(state.tab) ? state.tab : 'Overview';
  const host = el('div', {});
  const strip = el('div', { class: 'flex gap-1 mb-4 mt-6', style: { borderBottom: '1px solid var(--border)' } });
  const body = el('div', {});
  const show = () => {
    for (const node of strip.children) node.classList.toggle('active', node.textContent === state.tab);
    if (state.tab === 'Overview') renderOverview(body, e, data);
    else if (state.tab === 'Metrics') body.innerHTML = '', body.appendChild(metricPanel(e.metrics));
    else renderTraces(body, ctx.query, ctx.videos, ctx.clock);
  };
  for (const name of tabs) {
    strip.appendChild(el('div', {
      class: `tab ${name === active ? 'active' : ''}`,
      onclick: () => { state.tab = name; show(); },
    }, name));
  }
  host.append(strip, body);
  if (active === 'Overview') renderOverview(body, e, data);
  else if (active === 'Metrics') body.appendChild(metricPanel(e.metrics));
  else renderTraces(body, ctx.query, ctx.videos, ctx.clock);
  return host;
}

function renderCommands(episode, videos, clock, toTime) {
  const host = el('div', { class: 'mb-5' });
  const commands = episode.commands || [];
  if (!commands.length) return host;
  host.appendChild(el('div', { class: 'lang-label mb-1' }, `Commanded subtasks (${commands.length})`));
  const track = el('div', { class: 'flex gap-0.5 mb-2' });
  const segments = commands.map((command, i) => {
    const next = i + 1 < commands.length ? commands[i + 1].step : episode.num_steps;
    const width = Math.max(1, next - command.step) / Math.max(1, episode.num_steps);
    return el('div', {
      class: 'chip cursor-pointer truncate',
      style: { flex: `${width} 1 0`, minWidth: '0', justifyContent: 'flex-start' },
      title: `step ${command.step}: ${command.instruction}`,
      onclick: () => seekAll(videos, toTime(command.step)),
    }, command.instruction);
  });
  segments.forEach((s) => track.appendChild(s));
  host.appendChild(track);
  const highlight = () => {
    if (!videos.length || clock.duration == null) return;
    const step = (videos[0].currentTime / clock.duration) * episode.num_steps;
    let current = -1;
    commands.forEach((c, i) => { if (c.step <= step) current = i; });
    segments.forEach((s, i) => s.classList.toggle('accent', i === current));
  };
  if (videos.length) videos[0].addEventListener('timeupdate', highlight);
  clock.ready.push(highlight);
  return host;
}

function renderEvents(events, episode, videos, clock, toTime) {
  const host = el('div', { class: 'mb-2' });
  if (!events.length) {
    host.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'No events recorded.'));
    return host;
  }
  host.appendChild(el('div', { class: 'lang-label mb-1' }, `Events (${events.length})`));
  const strip = el('div', { class: 'events-strip mb-2' }, el('div', { class: 'events-strip-track' }));
  const markers = events.map((ev) => {
    const marker = el('div', {
      class: `events-strip-marker ${ev.severity}`,
      style: { left: `${Math.min(100, (ev.step / Math.max(1, episode.num_steps)) * 100)}%` },
      title: `step ${ev.step} · ${ev.name}${ev.info ? `\n${ev.info}` : ''}`,
      onclick: (e) => { e.stopPropagation(); seekAll(videos, toTime(ev.step)); },
    });
    strip.appendChild(marker);
    return marker;
  });
  const playhead = el('div', { class: 'events-strip-playhead', style: { left: '0%' } });
  strip.appendChild(playhead);
  strip.addEventListener('click', (e) => {
    const rect = strip.getBoundingClientRect();
    if (clock.duration != null) seekAll(videos, ((e.clientX - rect.left) / rect.width) * clock.duration);
  });
  host.appendChild(strip);

  const list = el('div', { class: 'events-list' });
  const rows = events.map((ev) => el('div', {
    class: `events-row ${ev.severity}`,
    onclick: () => seekAll(videos, toTime(ev.step)),
  },
    el('span', { class: 'ev-time' }, `step ${ev.step}`),
    el('span', { class: 'ev-info', title: ev.info },
      el('span', { class: 'ev-name' }, ev.name), ev.info ? ` · ${ev.info}` : ''),
    el('span', { class: 'ev-time' }, ev.kind)));
  rows.forEach((row) => list.appendChild(row));
  host.appendChild(list);

  let active = -1;
  const follow = () => {
    if (!videos.length || clock.duration == null) return;
    const time = videos[0].currentTime;
    playhead.style.left = `${Math.min(100, (time / clock.duration) * 100)}%`;
    const step = (time / clock.duration) * episode.num_steps;
    let index = -1;
    events.forEach((ev, i) => { if (ev.step <= step) index = i; });
    if (index === active) return;
    if (active >= 0) { rows[active].classList.remove('active'); markers[active].classList.remove('active'); }
    if (index >= 0) {
      rows[index].classList.add('active');
      markers[index].classList.add('active');
      rows[index].scrollIntoView({ block: 'nearest' });
    }
    active = index;
  };
  if (videos.length) videos[0].addEventListener('timeupdate', follow);
  clock.ready.push(follow);
  return host;
}

function renderOverview(host, episode, data) {
  host.innerHTML = '';
  const grid = el('div', { class: 'grid gap-4', style: { gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))' } });

  const reference = el('div', { class: 'panel' }, el('div', { class: 'lang-label mb-2' }, 'Reference subtasks'));
  if (episode.subtasks.length) {
    episode.subtasks.forEach((subtask, i) => reference.appendChild(
      el('div', { class: 'flex items-baseline gap-2 py-0.5 text-sm' },
        el('span', { style: { color: episode.goals[i] ? 'var(--success)' : 'var(--text-2)' } }, episode.goals[i] ? '✓' : '·'),
        el('span', {}, subtask))));
  } else {
    reference.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'Not a composition env.'));
  }
  grid.appendChild(reference);

  const plan = el('div', { class: 'panel' }, el('div', { class: 'lang-label mb-2' }, 'Controller plan'));
  if (episode.plan.length) {
    episode.plan.forEach((step, i) => plan.appendChild(el('div', { class: 'py-0.5 text-sm' }, `${i + 1}. ${step}`)));
  }
  // A plan-bearing orchestrator states its subtasks; a program-bearing one (cap-x, tiptop) logs its
  // generated code per turn. One panel holds both, in the order they were recorded.
  const programs = (data.events || []).filter((e) => e.kind === 'code' && e.code);
  if (programs.length) {
    plan.appendChild(el('div', { class: 'text-xs', style: { color: 'var(--text-2)' } },
      'programs the controller generated; each execute(...) line runs one subtask'));
  }
  for (const program of programs) {
    // A program that only wraps the plan's lines stays collapsed; anything else opens on load.
    const lines = program.code.split('\n').filter((l) => l.trim());
    const wrapping = lines.length > 0 && lines.every((l) => l.trim().startsWith('#') || /^execute\(/.test(l.trim()));
    const details = el('details', { style: { minWidth: 0 } }, el('summary', {
      class: 'text-xs cursor-pointer', style: { color: 'var(--text-2)' },
      title: `step ${program.step}${program.turn != null ? `, turn ${program.turn}` : ''}`,
    }, `program · step ${program.step}${program.turn != null ? ` · turn ${program.turn}` : ''} · ${lines.length} lines`));
    details.appendChild(el('pre', {
      class: 'text-xs font-mono overflow-auto mt-1', style: { maxHeight: '220px', whiteSpace: 'pre-wrap' },
    }, program.code));
    details.open = !wrapping;
    plan.appendChild(details);
  }
  if (!episode.plan.length && !programs.length) {
    plan.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'No plan recorded.'));
  }
  grid.appendChild(plan);

  if (data.goal_objects && data.goal_objects.length) {
    const objects = el('div', { class: 'panel' }, el('div', { class: 'lang-label mb-2' }, 'Goal objects'));
    for (const [object, destination, relation] of data.goal_objects) {
      objects.appendChild(el('div', { class: 'py-0.5 text-sm' },
        object, el('span', { style: { color: 'var(--text-2)' } }, ` ${relation || '→'} `), destination || '—'));
    }
    grid.appendChild(objects);
  }
  host.appendChild(grid);
  host.appendChild(el('div', { class: 'text-xs font-mono mt-4', style: { color: 'var(--text-2)' } }, episode.path));
}

async function renderTraces(host, query, videos, clock) {
  host.innerHTML = '';
  host.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'Loading traces…'));
  const traces = await fetchJSON(`/api/traces?${query}`);
  host.innerHTML = '';
  const arms = Object.entries(traces.motion || {}).filter(([, rows]) => rows.length);
  if (!arms.length && !(traces.goals || []).length) {
    host.appendChild(el('div', { class: 'text-sm', style: { color: 'var(--text-2)' } }, 'No traces logged.'));
    return;
  }
  const steps = Math.max(1, traces.num_steps || 1);
  const layout = (title, yTitle) => ({
    title: { text: title, font: { size: 13, color: '#a4a8b4' } },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: '#a4a8b4', size: 10 }, margin: { l: 48, r: 16, t: 34, b: 34 }, height: 240,
    xaxis: { title: 'step', gridcolor: '#20232c', range: [0, steps] },
    yaxis: { title: yTitle, gridcolor: '#20232c' },
    legend: { orientation: 'h', y: -0.25 },
  });
  const config = { displayModeBar: false, responsive: true };
  const plots = [];
  const add = (series, options) => {
    const node = el('div', { class: 'panel mb-4' });
    host.appendChild(node);
    Plotly.newPlot(node, series, options, config);
    plots.push(node);
  };

  for (const [arm, rows] of arms) {
    const xs = rows.map(([step]) => step);
    add(['x', 'y', 'z', 'gripper'].map((name, i) => ({
      x: xs, y: rows.map(([, v]) => v[i]), name, type: 'scatter', mode: 'lines',
      yaxis: i === 3 ? 'y2' : 'y', line: { width: 1.4 },
    })), {
      ...layout(`${arm} effector position and gripper`, 'metres'),
      yaxis2: { title: 'openness', overlaying: 'y', side: 'right', range: [0, 1.05], gridcolor: 'rgba(0,0,0,0)' },
    });
  }

  if ((traces.goals || []).length) {
    add(Array.from({ length: traces.goals[0][1].length }, (_, goal) => ({
      x: traces.goals.map(([step]) => step),
      y: traces.goals.map(([, achieved]) => (achieved[goal] ? 1 : 0) + goal * 1.2),
      name: (traces.subtasks || [])[goal] || `goal ${goal}`,
      type: 'scatter', mode: 'lines', line: { shape: 'hv', width: 1.6 },
    })), { ...layout('Per-goal state', 'achieved'), yaxis: { showticklabels: false, gridcolor: '#20232c' } });
  }

  const objects = Object.entries(traces.objects || {});
  if (objects.length) {
    add(objects.map(([name, rows]) => {
      const [, origin] = rows[0];
      return {
        x: rows.map(([step]) => step),
        y: rows.map(([, c]) => Math.hypot(c[0] - origin[0], c[1] - origin[1], c[2] - origin[2])),
        name, type: 'scatter', mode: 'lines', line: { width: 1.4 },
      };
    }), layout('Object travel from its start pose', 'metres'));
  }

  if (videos.length) {
    videos[0].addEventListener('timeupdate', () => {
      if (clock.duration == null) return;
      const step = (videos[0].currentTime / clock.duration) * steps;
      const shapes = [{ type: 'line', x0: step, x1: step, yref: 'paper', y0: 0, y1: 1, line: { color: '#4f46e5', width: 1 } }];
      for (const node of plots) Plotly.relayout(node, { shapes });
    });
  }
}

async function boot() {
  $('#brand').addEventListener('click', () => go('#/'));
  $('#refresh').addEventListener('click', async () => {
    await fetch('/api/refresh', { method: 'POST' });
    await load();
  });
  window.addEventListener('hashchange', () => { render(); renderSidebar(); });
  await load();
}

async function load() {
  const data = await fetchJSON('/api/runs');
  state.runs = data.runs || [];
  $('#roots').textContent = (data.roots || []).join('  ');
  $('#roots').title = (data.roots || []).join('\n');
  renderSidebar();
  render();
}

boot();
