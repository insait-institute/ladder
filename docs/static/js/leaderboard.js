// LADDER leaderboard: three views drawn from the JSON embedded in leaderboard.html (the paper's systems) and
// static/data/submissions.json (systems submitted by pull request, scored by scripts/leaderboard.py build).
document.addEventListener('DOMContentLoaded', function () {
  var DATA = document.getElementById('lb-data');
  if (!DATA) return;  // pages without the leaderboard (research.html)
  var D = JSON.parse(DATA.textContent);
  var FULL = D.systems;
  var SUITES = D.suites;
  var GROUP = {}; D.groups.forEach(function (g) { GROUP[g.id] = g.name; });
  var COLORS = { e2e: '#9aa0aa', orch: '#e08a1e', tool: '#e63e3e', ref: '#9aa0aa', sub: '#1a9e77' };
  var state = { view: 'overall', sort: 'mean', suite: 'mean', metric: 'success' };
  var $ = function (s) { return document.querySelector(s); };
  var fmt = function (v) { return v == null ? '&ndash;' : v.toFixed(1); };
  var rankable = function (v) { return v == null ? -1 : v; };
  var avg = function (xs) { xs = xs.filter(function (x) { return x != null; }); return xs.length ? xs.reduce(function (a, b) { return a + b; }, 0) / xs.length : null; };

  function seg(items, value, onpick) {
    var box = document.createElement('div');
    box.className = 'lb-seg';
    items.forEach(function (it) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = it[0] === value ? 'is-active' : '';
      b.innerHTML = '<span>' + it[1] + '</span>';
      b.setAttribute('data-label', it[1].replace(/<[^>]+>/g, '').replace(/&middot;/g, '\u00b7'));
      b.onclick = function () { onpick(it[0]); draw(); };
      box.appendChild(b);
    });
    return box;
  }
  // the model a visitor knows (GPT-6 Astra, Gemini ...) leads; the system it runs through follows
  // where the frontier model is the system (GPT-Policy, PSP-Agent and submissions built the same way), it leads and the
  // harness follows; for every other system the method leads and its VLM is a detail
  var MODEL_FIRST = ['GPT-Policy', 'PSP-Agent'];
  function who(s) {
    if (s.group === 'ref') return { main: 'Privileged verifier', sub: 'reference, not ranked' };
    if (s.group === 'e2e') return { main: 'Robotics-only model', sub: '&pi;<sub>0.5</sub> / Bridge policy' };
    if (s.vlm && MODEL_FIRST.indexOf(s.name) >= 0) return { main: s.vlm, sub: 'via ' + s.name };
    if (s.vlm) return { main: s.name, sub: 'with ' + s.vlm };
    if (s.group === 'orch') return { main: s.name, sub: 'robotics model, rule-based steps' };
    return { main: s.name, sub: '' };
  }
  // short chart labels: the name a reader looks for
  function label(s) { return who(s).main; }
  function name(s) {
    var w = who(s);
    return '<span class="lb-name"><i style="background:' + COLORS[s.group] + '"></i><span class="lb-who"><span class="lb-main">' + w.main +
      (s.liteOnly || state.bench === 'lite' ? ' <span class="lb-badge is-lite" title="' + (s.liteOnly ? LITE_TIP : LITE_ALL_TIP) + '">Lite</span>' : '') +
      (s.partial ? ' <span class="lb-badge" title="Scored on ' + s.partial + ' of LADDER-Lite; the last one is still running">' + s.partial + '</span>' : '') +
      (s.submission && s.submission !== 'Placeholder' ? ' <span class="lb-badge' + (s.verified ? ' is-verified' : '') + '" title="' +
        (s.verified ? 'Re-run by the maintainers' : 'Results as submitted; the episode logs are public') + '">' +
        (s.verified ? 'verified' : 'self-reported') + '</span>' : '') +
      '</span>' + (w.sub ? '<span class="lb-sub">' + w.sub + '</span>' : '') + '</span></span>';
  }
  // how closely LADDER-Lite follows the full benchmark, from the data: how many of the leaders keep their place, and
  // the largest gap between a system's Lite and full score
  var TOTAL_TASKS = 145;
  var LITE_FIT = (function () {
    if (!D.lite) return '';
    var both = D.lite.systems.filter(function (l) { return !l.liteOnly && l.group !== 'ref'; }).map(function (l) {
      return { lite: l.mean.v, full: FULL.filter(function (f) { return f.name === l.name; })[0].mean.v };
    });
    var byFull = both.slice().sort(function (a, b) { return b.full - a.full; });
    var byLite = both.slice().sort(function (a, b) { return b.lite - a.lite; });
    var top = 0; while (top < byFull.length && byFull[top] === byLite[top]) top++;
    var gap = Math.max.apply(null, both.map(function (x) { return Math.abs(x.lite - x.full); }));
    return 'On it, the ' + (top === 2 ? 'two' : top === 3 ? 'three' : top) + ' best systems keep their places, and every score is within ' +
      gap.toFixed(1) + '% of its score on all ' + TOTAL_TASKS + ' tasks.';
  })();
  var LITE_ALL_TIP = 'Scored here on the LADDER-Lite tasks only; its full-benchmark score is in the other views';
  var LITE_TIP = 'Scored on LADDER-Lite, the ' + (D.lite ? SUITES.reduce(function (a, s) { return a + D.lite.tasks[s.id].length; }, 0) : '') +
    '-task version of LADDER, not on all ' + TOTAL_TASKS + ' tasks';
  var KIND = { Policy: 'End-to-end policy', Orchestrator: 'Orchestrator + low-level policy', Agent: 'Tool-using agent', Reference: 'Privileged reference, not ranked' };
  var SUITE_ITEMS = [['mean', 'All simulators']].concat(SUITES.map(function (s) { return [s.id, s.name]; }));

  // ---------- overall: one ranked table, sortable by any column, filterable by system type
  var money = function (v) { return v == null || v === 0 ? '&ndash;' : '$' + v.toFixed(v < 0.1 ? 3 : 2); };
  var SCORES = {
    mean: { label: 'Success', get: function (s) { return s.mean.v; }, pm: function (s) { return s.mean.pm; } },
    se: { get: function (s) { return s.suites.se.v; } },
    ra: { get: function (s) { return s.suites.ra.v; } },
    rl: { get: function (s) { return s.suites.rl.v; } },
    progress: { get: function (s) { return s.progress.mean; } },
    cost: { get: function (s) { return s.cost_ep; }, asc: true },
    succ: { get: function (s) { return s.cost_succ || null; }, asc: true },
    date: { get: function (s) { return s.evaluated ? Date.parse(s.evaluated[1]) : null; } },
    scale: { get: function (s) { return s.scale ? s.scale.episodes : null; } }
  };
  // the scale of a system's evaluation: different tasks, and attempts in total
  function scale(s) {
    if (!s.scale) return '&ndash;';
    return '<span class="lb-scale-v" title="' + s.scale.tasks + ' different tasks, ' + s.scale.episodes.toLocaleString('en-US') + ' attempts in total">' +
      s.scale.tasks + ' &middot; ' + s.scale.episodes.toLocaleString('en-US') + '</span>';
  }
  // a system's videos and logs: its folder in the released evaluation dataset, which holds every simulator and real robot
  function files(s) {
    var url = s.rollouts && D.rollouts_url ? D.rollouts_url + '/tree/main/' + s.rollouts : s.results_url;
    return url ? '<a class="lb-view-link" href="' + url + '" target="_blank" rel="noopener" title="Video and log of every attempt, on Hugging Face (opens in a new tab)">View<span aria-hidden="true"> &#8599;</span></a>' : '&ndash;';
  }
  // when a system was evaluated: one day, or a range within one month / across months ("13 Sep 2026", "10-12 Sep 2026")
  var MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  function when(s) {
    if (!s.evaluated) return '&ndash;';
    var a = s.evaluated[0].split('-'), b = s.evaluated[1].split('-');
    var day = function (d) { return +d[2] + ' ' + MON[+d[1] - 1]; };
    if (a.join() === b.join()) return day(a) + ' ' + a[0];
    if (a[0] === b[0] && a[1] === b[1]) return +a[2] + '&ndash;' + day(b) + ' ' + a[0];
    return day(a) + ' &ndash; ' + day(b) + ' ' + b[0];
  }
  var TYPES = [['all', 'All systems'], ['Policy', 'Policies'], ['Orchestrator', 'Orchestrators'], ['Agent', 'Agents']];
  state.type = 'all';
  function overall(view, controls) { return overallBody(view, controls); }
  function overallBody(view, controls) {
    var sc = SCORES[state.sort] || SCORES.mean;
    var natural = sc.asc ? 'asc' : 'desc', dir = state.dir || natural;
    var up = dir === 'asc';
    var key = function (s) { var v = sc.get(s); return v == null ? (up ? 1e9 : -1) : v; };
    var shown = D.systems.filter(function (s) { return state.type === 'all' || s.type === state.type || (s.group === 'sub' && s.type === state.type); });
    var ranked = shown.filter(function (s) { return s.group !== 'ref' && s.ranked !== false; })
      .sort(function (a, b) { return up ? key(a) - key(b) : key(b) - key(a); });
    if (state.sort === 'mean' && !up) {
      var slotted = ranked.filter(function (s) { return s.slot; }).sort(function (a, b) { return a.slot - b.slot; });
      ranked = ranked.filter(function (s) { return !s.slot; });
      slotted.forEach(function (s) { ranked.splice(Math.min(s.slot - 1, ranked.length), 0, s); });
    }
    var rows = ranked.concat(shown.filter(function (s) { return s.group === 'ref' || s.ranked === false; }));
    var TIP = { mean: 'Share of tasks the robot finished completely, with every object in the right place',
      progress: 'Share of all individual moves done, including those in unfinished tasks',
      cost: 'What the AI model charges for one try at a task',
      succ: 'What the AI model charges per finished task',
      date: 'The days on which the system was tested',
      scale: 'How many different tasks the system was tested on, and how many attempts it made in total' };
    var head = function (k, label, cls) {
      return '<th class="lb-num lb-sort' + (state.sort === k ? ' is-on is-' + dir : '') + (cls ? ' ' + cls : '') + '" data-k="' + k + '"' + (TIP[k] ? ' title="' + TIP[k] + '"' : '') + '>' + label + '</th>';
    };
    var suites = state.view === 'suites', simple = state.view === 'overall';
    var html = '<table class="lb-table' + (simple ? ' lb-simple' : '') + '"><thead><tr><th class="lb-rank">#</th><th class="lb-left">AI model</th>' +
      head('mean', simple ? 'Tasks completed' : 'Success (%)', 'lb-score') +
      (simple ? head('succ', 'Cost per<br>completed task', 'lb-money') :
       suites ? SUITES.map(function (s) { return head(s.id, s.name.replace(/-/g, '&#8209;').replace('&#8209;Long', '<br>&#8209;Long')); }).join('') :
        head('progress', 'Subgoals done (%)') + head('cost', 'Cost per<br>attempt', 'lb-money') + head('succ', 'Cost per<br>completed task', 'lb-suite')) +
      (simple ? '' : head('scale', 'Tasks &middot; attempts', 'lb-scale') + head('date', 'Date tested', 'lb-date') + '<th class="lb-files">Videos &amp; logs</th>') + '</tr></thead><tbody>';
    var rank = 0, last = null, n = 0;
    rows.forEach(function (s) {
      var ref = s.group === 'ref' || s.ranked === false, v = sc.get(s), m = s.mean.v, pm = s.mean.pm;
      if (!ref) { n += 1; if (s.slot && state.sort === 'mean') { rank = n; last = null; } else if (v !== last) rank = n; last = s.slot ? null : v; }
      var on = function (k) { return state.sort === k ? ' is-on' : ''; };
      html += '<tr class="' + (ref ? 'lb-ref' : '') + (!ref && rank <= 3 && state.sort === 'mean' ? ' lb-top' : '') + '">' +
        '<td class="lb-rank">' + (ref || state.sort === 'date' || state.sort === 'scale' || (v == null && !(s.slot && state.sort === 'mean')) ? '' : rank) + '</td><td class="lb-left">' + name(s) + (simple ? '' : '<span class="lb-when">' + (s.scale ? s.scale.tasks + ' tasks &middot; ' + s.scale.episodes.toLocaleString('en-US') + ' attempts<br>' : '') +
          'Tested ' + when(s) + ' &middot; ' + files(s) + '</span>') + '</td>' +
        '<td class="lb-score' + on('mean') + '"><div class="lb-scorebox"><div class="lb-meter"><span style="width:' + (m || 0) + '%"></span></div><b>' + (simple && m != null ? fmt(m) + '%' : fmt(m)) + '</b>' +
        '<small>' + (pm != null && !simple ? '&plusmn;' + fmt(pm) : '') + '</small></div></td>' +
        (simple ? '<td class="lb-num lb-money' + on('succ') + '">' + (s.cost_succ ? money(s.cost_succ) : '&ndash;') + '</td>' :
         suites ? SUITES.map(function (x) { return '<td class="lb-num' + on(x.id) + '">' + fmt(s.suites[x.id].v) + '<small class="lb-pm">' + (s.suites[x.id].pm != null ? '&plusmn;' + fmt(s.suites[x.id].pm) : '') + '</small></td>'; }).join('') :
          '<td class="lb-num' + on('progress') + '">' + fmt(s.progress.mean) + '</td>' +
          '<td class="lb-num lb-money' + on('cost') + '">' + money(s.cost_ep) + '</td>' +
          '<td class="lb-num lb-money lb-suite' + on('succ') + '">' + money(s.cost_succ) + '</td>') +
        (simple ? '' : '<td class="lb-num lb-scale' + on('scale') + '">' + scale(s) + '</td><td class="lb-num lb-date' + on('date') + '">' + when(s) + '</td><td class="lb-files">' + files(s) + '</td>') + '</tr>';
    });
    view.innerHTML = '<div class="lb-wrap">' + html + '</tbody></table></div>';
    view.querySelectorAll('.lb-sort').forEach(function (th) {
      th.onclick = function () {
        // first click: best first; second: reversed; third: back to the default ranking (by success)
        var k = th.dataset.k, first = (SCORES[k] || {}).asc ? 'asc' : 'desc', cur = state.dir || 'desc';
        if (state.sort !== k) { state.sort = k; state.dir = first; }
        else if (cur === first) { state.dir = first === 'asc' ? 'desc' : 'asc'; }
        else { state.sort = 'mean'; state.dir = null; }
        draw();
      };
    });
    var dated = 'Tasks &middot; attempts: how many different tasks a system was tested on, and how many attempts it made in total. ' +
      'Date tested: hosted AI models change over time, so a score holds for the model as it was on those days. ' +
      'Videos &amp; logs: a recording of every attempt, on Hugging Face.';
    if (suites) return 'Tasks completed (%) in each of the 3 simulators, with its margin of error. Each simulator has its own robot and objects. ' + dated + ' Click a column to rank by it.';
    if (simple) return '';
    return 'Subgoals done: the share of individual subgoals completed, including in unfinished tasks. Cost per attempt: what the AI model charges for one attempt. ' + dated + ' Click a column to rank by it.';
  }

  // success against cost per episode (log scale); systems without a hosted VLM sit in a "$0" lane on the left
  function frontier(size) {
    size = size || {};
    // the privileged reference reads the simulator and is not a model anyone can use: left out of this chart
    var pts = D.systems.filter(function (s) { return s.mean.v != null && s.cost_ep != null && s.group !== 'ref'; });
    if (!pts.length) return '';
    var paid = pts.filter(function (s) { return s.cost_ep > 0; }).map(function (s) { return s.cost_ep; });
    if (!paid.length) return '';
    var W = size.W || 1100, H = size.H || 440, L = size.L || 72, R = 30, T = 16, B = 48, Z = size.Z || 200, x0 = L + Z;
    var phone = !!size.phone;
    var lo = Math.log10(Math.min.apply(null, paid) / 1.8), hi = Math.log10(Math.max.apply(null, paid) * 1.8);
    var top = Math.ceil((Math.max.apply(null, pts.map(function (s) { return s.mean.v; })) + 8) / 10) * 10;
    var x = function (c) { return c <= 0 ? L + 14 : x0 + (W - R - x0) * (Math.log10(c) - lo) / (hi - lo); };
    var y = function (v) { return T + (H - T - B) * (1 - v / top); };
    var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="lb-chart lb-front ' + (phone ? 'for-phone' : 'for-wide') + '">';
    for (var g = 0; g <= top; g += 10) if (g % 20 === 0) svg += '<line class="lb-gridline" x1="' + L + '" x2="' + (W - R) + '" y1="' + y(g) + '" y2="' + y(g) + '"/><text x="' + (L - 8) + '" y="' + (y(g) + 4) + '" text-anchor="end">' + g + '%</text>';
    for (var e = Math.ceil(lo); e <= Math.floor(hi); e++) {
      var c = Math.pow(10, e);
      svg += '<line class="lb-gridline" x1="' + x(c) + '" x2="' + x(c) + '" y1="' + T + '" y2="' + (H - B) + '"/><text x="' + x(c) + '" y="' + (H - B + 18) + '" text-anchor="middle">$' + c + '</text>';
    }
    svg += '<text x="' + (L + 14) + '" y="' + (H - B + 18) + '" text-anchor="middle">$0</text><line class="lb-guide" x1="' + x0 + '" x2="' + x0 + '" y1="' + T + '" y2="' + (H - B) + '"/>' +
      '<text x="' + ((x0 + W - R) / 2) + '" y="' + (H - 6) + '" text-anchor="middle" class="lb-axis-t">cost per attempt (US dollars) &rarr;</text>' +
      '<text transform="translate(18,' + ((T + H - B) / 2) + ') rotate(-90)" text-anchor="middle" class="lb-axis-t">tasks completed &rarr;</text>';
    // the line joins scores on the same tasks: systems scored on LADDER-Lite only are plotted but not part of it
    var front = pts.filter(function (s) { return s.group !== 'ref' && !s.liteOnly; }).sort(function (a, b) { return a.cost_ep - b.cost_ep || b.mean.v - a.mean.v; });
    var best = -1, line = [];
    front.forEach(function (s) { if (s.mean.v > best) { best = s.mean.v; line.push(s); } });
    svg += '<polyline class="lb-pareto" points="' + line.map(function (s) { return x(s.cost_ep) + ',' + y(s.mean.v); }).join(' ') + '"/>';
    // labels: spread those that would overlap
    var placed = [];
    // each label sits beside its own mark: to the right when it fits, else to the left. Overlap is checked on the
    // span the label actually covers, and a clashing label moves down until it clears
    pts.slice().sort(function (a, b) { return b.mean.v - a.mean.v; }).forEach(function (s) {
      var cx = x(s.cost_ep), cy = y(s.mean.v), ty = cy + 4;
      var w2 = who(s), sub = !phone && w2.sub.indexOf('via ') === 0 ? ' ' + w2.sub : '';
      var w = (w2.main + (s.liteOnly ? ' · LADDER-Lite' : '') + sub).length * 7.4 + 14;
      var right = cx + w < W - 8;
      var x1 = right ? cx + 11 : cx - 11 - w, x2 = x1 + w, moved = true;
      while (moved) {
        moved = false;
        placed.forEach(function (p) { if (x1 < p[1] && p[0] < x2 && Math.abs(p[2] - ty) < 16) { ty = p[2] + 16; moved = true; } });
      }
      placed.push([x1, x2, ty]);
      var mark = s.liteOnly
        ? '<path d="M' + cx + ' ' + (cy - 9) + 'L' + (cx + 9) + ' ' + cy + 'L' + cx + ' ' + (cy + 9) + 'L' + (cx - 9) + ' ' + cy + 'Z" class="lite-mark"/>'
        : '<circle cx="' + cx + '" cy="' + cy + '" r="7" style="fill:' + COLORS[s.group] + '"/>';
      // a label pushed away from its mark keeps a thin leader line back to it
      var lead = Math.abs(ty - 4 - cy) > 6 ? '<line class="pt-lead" x1="' + cx + '" y1="' + cy + '" x2="' + (right ? x1 - 2 : x2 + 2) + '" y2="' + (ty - 4) + '"/>' : '';
      var tipd = { h: w2.main + (w2.sub ? ' ' + w2.sub.replace(/<sub>(.*?)<\/sub>/g, '$1').replace(/&pi;/g, '\u03c0').replace(/<[^>]+>/g, '') : '') + (s.liteOnly ? ' (LADDER-Lite)' : ''), r: [
        ['', 'Tasks completed', s.mean.v.toFixed(1) + '%'], ['', 'Cost per attempt', s.cost_ep ? '$' + s.cost_ep.toFixed(s.cost_ep < 0.1 ? 3 : 2) : 'no API cost'],
        ['', 'Cost per completed task', s.cost_succ ? '$' + s.cost_succ.toFixed(2) : 'no API cost']] };
      svg += '<g class="lb-pt tip-target' + (s.group === 'ref' ? ' is-ref' : '') + (s.liteOnly ? ' is-lite' : '') + '" tabindex="0" data-tip="' +
        JSON.stringify(tipd).replace(/&/g, '&amp;').replace(/"/g, '&quot;') + '">' + lead + mark +
        '<text x="' + (right ? x1 : x2) + '" y="' + ty + '" text-anchor="' + (right ? 'start' : 'end') + '"><tspan class="pt-main">' + w2.main +
        (s.liteOnly ? '<tspan class="pt-lite"> &middot; LADDER-Lite</tspan>' : '') + '</tspan>' +
        (sub ? '<tspan class="pt-sub">' + sub + '</tspan>' : '') + '</text></g>';
    });
    var key = '<div class="lb-typekey front-key"><i style="background:' + COLORS.tool + '"></i>AI agent <i style="background:' + COLORS.orch +
      '"></i>AI model + robotics model <i style="background:' + COLORS.e2e + '"></i>robotics-only model</div>';
    return size.bare ? svg + '</svg>' : '<div class="lb-frontbox">' + svg + '</svg>' + key + '</div>';
  }

  // ---------- by horizon and by category: one row per system, one column per subgoal count or category
  function cell(table, row, sys) {
    if (state.suite !== 'mean') { var r = table[state.suite][row]; return r ? r[sys] : null; }
    // averaged over the suites only when the system has all three (a submission may cover fewer)
    var vs = SUITES.map(function (s) { var r = table[s.id][row]; return r ? r[sys] : null; });
    return vs.indexOf(null) >= 0 || vs.indexOf(undefined) >= 0 ? null : avg(vs);
  }
  function grid(table, rows, label) {
    var systems = D.systems.slice().sort(function (a, b) { return rankable(cell(table, 'All tasks', b.name)) - rankable(cell(table, 'All tasks', a.name)); });
    var html = '<table class="lb-table lb-grid"><thead><tr><th class="lb-left">Model</th>' +
      rows.map(function (r) { return '<th class="lb-num' + (r === 'All tasks' ? ' lb-all' : '') + '">' + label(r) + '</th>'; }).join('') + '</tr></thead><tbody>';
    systems.forEach(function (s) {
      html += '<tr class="' + (s.group === 'ref' ? 'lb-ref' : '') + '"><td class="lb-left">' + name(s) + '</td>' +
        rows.map(function (r) {
          var v = cell(table, r, s.name);
          var per = SUITES.map(function (su) { var x = table[su.id][r]; return ['', su.name.replace('&infin;', '\u221e'), x && x[s.name] != null ? x[s.name].toFixed(1) + '%' : '\u2013']; });
          var tipd = { h: who(s).main + ', ' + (r === 'All tasks' ? 'all tasks' : r), r: [['', 'Mean of the three suites', v == null ? '–' : v.toFixed(1) + '%']].concat(per) };
          return '<td class="lb-num lb-heat tip-target' + (r === 'All tasks' ? ' lb-all' : '') + '" tabindex="0" data-tip="' +
            JSON.stringify(tipd).replace(/&/g, '&amp;').replace(/"/g, '&quot;') + '" style="--v:' + (v == null ? 0 : v / 100).toFixed(3) + '">' + fmt(v) + '</td>';
        }).join('') + '</tr>';
    });
    return '<div class="lb-wrap">' + html + '</tbody></table></div>';
  }
  // one colour per system (Tableau 10); the privileged reference is dashed
  var PALETTE = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#7f7f7f'];
  var colour = function (s) { return PALETTE[D.systems.indexOf(s) % PALETTE.length]; };
  var hidden = {};
  // the length chart starts with a few lines a visitor can follow; every other system can be switched on
  var SHOWN_FIRST = ['GPT-Policy', 'PSP-Agent', 'Proprioceptive verifier', 'Low-level policy'];
  D.systems.forEach(function (s) { if (SHOWN_FIRST.indexOf(s.name) < 0) hidden[s.name] = true; });
  var W = 760, H = 300, L = 40, R = 16, T = 12, B = 34;

  function chart(table, xs) {
    var x = function (i) { return L + (W - L - R) * i / (xs.length - 1); };
    var y = function (v) { return T + (H - T - B) * (1 - v / 100); };
    var svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="lb-chart">';
    [0, 25, 50, 75, 100].forEach(function (g) { svg += '<line class="lb-gridline" x1="' + L + '" x2="' + (W - R) + '" y1="' + y(g) + '" y2="' + y(g) + '"/><text x="' + (L - 8) + '" y="' + (y(g) + 4) + '" text-anchor="end">' + g + '</text>'; });
    xs.forEach(function (r, i) { svg += '<text x="' + x(i) + '" y="' + (H - 12) + '" text-anchor="middle">' + parseInt(r, 10) + '</text>'; });
    svg += '<rect class="lb-hit" x="' + L + '" y="' + T + '" width="' + (W - L - R) + '" height="' + (H - T - B) + '"/>';
    svg += '<line class="lb-guide" x1="0" x2="0" y1="' + T + '" y2="' + (H - B) + '" style="display:none"/>';
    D.systems.forEach(function (s) {
      if (hidden[s.name]) return;
      var pts = xs.map(function (r, i) { var v = cell(table, r, s.name); return v == null ? null : [x(i), y(v)]; }).filter(Boolean);
      svg += '<g class="lb-series" data-sys="' + s.name + '" style="--c:' + colour(s) + '"><polyline points="' + pts.map(function (p) { return p.join(','); }).join(' ') + '"' +
        (s.group === 'ref' ? ' stroke-dasharray="6 5"' : '') + '/>' + pts.map(function (p) { return '<circle cx="' + p[0] + '" cy="' + p[1] + '" r="3.2"/>'; }).join('') + '</g>';
    });
    svg += '</svg>';
    var legend = D.systems.map(function (s) {
      return '<button type="button" class="lb-key' + (hidden[s.name] ? ' is-off' : '') + '" data-sys="' + s.name + '"><i style="background:' + colour(s) + '"></i>' + label(s) + '</button>';
    }).join('');
    return '<div class="lb-chartbox"><div class="lb-plot">' + svg + '<div class="lb-tip" hidden></div></div><p class="lb-axis">subgoals in the task</p>' +
      '<div class="lb-legend">' + legend + '<span class="lb-keys-all"><button type="button" data-all="1">Show all</button> &middot; <button type="button" data-all="0">Hide all</button></span></div></div>';
  }

  function wire(view, table, xs) {
    var box = view.querySelector('.lb-chartbox'), svg = box.querySelector('svg'), tip = box.querySelector('.lb-tip'), guide = box.querySelector('.lb-guide');
    var focus = function (name) {
      box.classList.toggle('is-focus', !!name);
      box.querySelectorAll('.lb-series').forEach(function (g) { g.classList.toggle('is-on', g.dataset.sys === name); });
      box.querySelectorAll('.lb-key').forEach(function (k) { k.classList.toggle('is-on', k.dataset.sys === name); });
    };
    box.querySelectorAll('.lb-key').forEach(function (k) {
      k.onmouseenter = function () { if (!hidden[k.dataset.sys]) focus(k.dataset.sys); };
      k.onmouseleave = function () { focus(null); };
      k.onclick = function () { hidden[k.dataset.sys] = !hidden[k.dataset.sys]; draw(); };
    });
    box.querySelectorAll('[data-all]').forEach(function (b) {
      b.onclick = function () { D.systems.forEach(function (s) { hidden[s.name] = b.dataset.all === '0'; }); draw(); };
    });
    box.querySelectorAll('.lb-series').forEach(function (g) {
      g.onmouseenter = function () { focus(g.dataset.sys); };
      g.onmouseleave = function () { focus(null); };
    });
    // hovering the plot reads out every shown system at the nearest number of subgoals
    var move = function (e) {
      var r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * W / r.width;
      var i = Math.max(0, Math.min(xs.length - 1, Math.round((px - L) / (W - L - R) * (xs.length - 1))));
      var gx = L + (W - L - R) * i / (xs.length - 1);
      guide.setAttribute('x1', gx); guide.setAttribute('x2', gx); guide.style.display = '';
      var items = D.systems.filter(function (s) { return !hidden[s.name]; }).map(function (s) { return [s, cell(table, xs[i], s.name)]; })
        .filter(function (p) { return p[1] != null; }).sort(function (a, b) { return b[1] - a[1]; });
      if (!items.length) { tip.hidden = true; return; }
      tip.innerHTML = '<b>' + parseInt(xs[i], 10) + (parseInt(xs[i], 10) === 1 ? ' subgoal' : ' subgoals') + '</b>' + items.map(function (p) {
        return '<span><i style="background:' + colour(p[0]) + '"></i>' + label(p[0]) + '<em>' + fmt(p[1]) + '</em></span>';
      }).join('');
      tip.hidden = false;
      var left = gx * r.width / W;
      tip.style.left = (left > r.width * 0.6 ? left - tip.offsetWidth - 14 : left + 14) + 'px';
    };
    svg.onmousemove = move;
    svg.onmouseleave = function () { tip.hidden = true; guide.style.display = 'none'; };
  }

  function horizon(view) {
    var table = D.horizon[state.metric];
    var rows = Object.keys(table.se).filter(function (k) { return k !== '_n'; });
    var xs = rows.filter(function (r) { return r !== 'All tasks'; });
    view.innerHTML = chart(table, xs) + grid(table, rows, function (r) { return r === 'All tasks' ? 'All' : parseInt(r, 10); });
    wire(view, table, xs);
    return (state.metric === 'success' ? 'Tasks completed' : 'Subgoals done') + ' (%) by the number of subgoals in the task' +
      (state.suite === 'mean' ? ', averaged over the 3 simulators.' : '.') +
      ' Hover the plot to read the values, click a name to hide or show its line.';
  }
  function category(view) {
    var table = D.category.success;
    var rows = Object.keys(table.se).filter(function (k) { return k !== '_n'; });
    view.innerHTML = grid(table, rows, function (r) { return r === 'All tasks' ? 'All' : r.replace(' destination', '<br>destination'); });
    return 'Tasks completed (%) per task category' + (state.suite === 'mean' ? ', averaged over the 3 simulators.' : '.') + ' The categories are described on the <a href="research#benchmark">Research page</a>.';
  }

  function cost(view) {
    view.innerHTML = frontier();
    return 'Each dot is one model. Higher is more successful, further right is more expensive. Models on the dashed line have no rival that is both cheaper and more successful.';
  }
  function details(view, controls) {
    var lite = state.bench === 'lite';
    // "All on LADDER-Lite" ranks every system on the same Lite tasks; the other choices are on all 145 tasks
    var items = [['all', 'All numbers'], ['suites', 'By simulator'], ['horizon', 'By number of subgoals'], ['category', 'By task category']];
    if (D.lite) items.push(['lite', 'Compare on LADDER-Lite']);
    var row1 = seg(items, lite ? 'lite' : state.detail, function (v) {
      if (v === 'lite') { state.bench = 'lite'; state.detail = 'all'; } else { state.bench = 'full'; state.detail = v; }
      state.sort = 'mean'; state.dir = null;
    });
    controls.appendChild(row1);
    // the second row always takes the same space; its options only act on the views that use them
    var row2 = document.createElement('div'); row2.className = 'lb-row2';
    var uses = state.detail === 'horizon' || state.detail === 'category';
    var m = seg([['success', 'Tasks completed'], ['progress', 'Subgoals done']], state.metric, function (v) { state.metric = v; });
    var su = seg(SUITE_ITEMS, state.suite, function (v) { state.suite = v; });
    if (state.detail !== 'horizon') m.classList.add('is-off');
    if (!uses) su.classList.add('is-off');
    row2.appendChild(m); row2.appendChild(su); controls.appendChild(row2);
    var note;
    if (state.detail === 'horizon') note = horizon(view);
    else if (state.detail === 'category') note = category(view);
    else { var keep = state.view; state.view = state.detail === 'suites' && !lite ? 'suites' : 'all'; note = overallBody(view, controls); state.view = keep; }
    return note + '<br>' + KEY;
  }
  var KEY = '<span class="lb-typekey"><i style="background:' + COLORS.e2e + '"></i>robotics-only model <i style="background:' + COLORS.orch +
    '"></i>AI model + robotics model <i style="background:' + COLORS.tool + '"></i>AI agent</span>';
  state.detail = 'all';
  var VIEWS = [['overall', 'Ranking', overall], ['details', 'Details', details]];
  // LADDER-Lite: the same systems on a fixed subset of tasks; only the ranked tables, since a breakdown by
  // horizon or category would rest on one or two tasks
  state.bench = 'full';
  function liteCounts() {
    var t = 0, e = 0;
    SUITES.forEach(function (s) { t += D.lite.tasks[s.id].length; e += D.lite.episodes[s.id]; });
    return { tasks: t, episodes: e };
  }
  function bench() {
    var c = liteCounts();
    var box = seg([['full', 'LADDER &middot; 145 tasks'], ['lite', 'LADDER-Lite &middot; ' + c.tasks + ' tasks']],
      state.bench, function (v) { state.bench = v; });
    box.classList.add('lb-bench');
    return box;
  }
  function liteNote() {
    var c = liteCounts(), full = SUITES.reduce(function (a, s) { return a + s.episodes; }, 0);
    var tags = SUITES.map(function (s) {
      return '<div class="lite-suite"><span class="lite-suite-name">' + s.name + '</span>' +
        D.lite.tasks[s.id].map(function (t) { return '<span class="lite-tag">' + t + '</span>'; }).join('') + '</div>';
    }).join('');
    return '<div class="lite-card">' +
      '<div class="lite-stats">' +
        '<div><b>' + c.tasks + '</b><span>tasks</span></div>' +
        '<div><b>' + c.episodes + '</b><span>episodes</span></div>' +
        '<div><b>' + Math.round(full / c.episodes) + '&times;</b><span>cheaper</span></div>' +
        '<div><b>' + D.lite.subgoals[0] + ' to ' + D.lite.subgoals[1] + '</b><span>steps per task</span></div>' +
      '</div>' +
      '<p class="lite-text">A smaller, fixed set of LADDER tasks for testing new models quickly and cheaply. ' +
      'It ranks the models in the same order as the full benchmark.</p>' +
      '<div class="lite-tasks">' + tags + '</div></div>';
  }
  function draw() {
    var views = $('.lb-views'), controls = $('.lb-controls'), view = $('.lb-view');
    views.innerHTML = ''; controls.innerHTML = '';
    var lite = state.bench === 'lite' && D.lite;
    var liteOnly = D.lite ? D.lite.systems.filter(function (s) { return s.liteOnly; }) : [];
    var ranked = state.view === 'overall' || (state.view === 'details' && (state.detail === 'all' || state.detail === 'suites'));
    D.systems = lite ? D.lite.systems : ranked ? FULL.concat(liteOnly) : FULL;
    var spacer = document.createElement('span'); views.appendChild(spacer);
    if (lite && state.view === 'details' && ['all', 'suites'].indexOf(state.detail) < 0) state.detail = 'all';
    var more = document.createElement('button');
    more.type = 'button'; more.className = 'lb-more';
    more.innerHTML = state.view === 'details' ? '&larr; Back to the simple view' : 'More details &rarr;';
    more.onclick = function () { state.view = state.view === 'details' ? 'overall' : 'details'; state.bench = 'full'; state.sort = 'mean'; state.dir = null; draw(); };
    views.appendChild(more);
    var note = VIEWS.filter(function (v) { return v[0] === state.view; })[0][2](view, controls);
    if (D.lite && ranked && !lite) note = '<span class="lb-badge is-lite">Lite</span> <b>LADDER-Lite</b> is a ' + liteCounts().tasks +
      '-task version of LADDER for testing new models quickly. ' + LITE_FIT + ' ' +
      'Rows marked Lite were scored on LADDER-Lite only.' + partialNote() + (note ? '<br>' + note : '');
    if (D.lite && !ranked && !lite && liteOnly.length) note = (note ? note + '<br>' : '') + liteOnly.map(function (s) { return s.vlm; }).join(' and ') +
      (liteOnly.length > 1 ? ' are' : ' is') + ' not shown here: scored on LADDER-Lite only, ' + liteCounts().tasks + ' tasks are too few to split this way.';
    if (lite) note = '<b>Every system on the same ' + liteCounts().tasks + ' LADDER-Lite tasks</b> (' + liteCounts().episodes +
      ' attempts each), for a direct comparison with the models scored on LADDER-Lite only. ' + LITE_FIT + partialNote() +
      ' With fewer tasks, the margins of error are wider than on the full benchmark.' + (note ? '<br>' + note : '');
    $('.lb-note').innerHTML = note;
  }
  // a Lite run still in progress is shown on the tasks it has finished, and says so
  function partialNote() {
    var p = D.lite ? D.lite.systems.filter(function (s) { return s.partial; }) : [];
    return p.map(function (s) { return ' ' + s.vlm + ' is scored on ' + s.partial + ' so far; the last one is still running.'; }).join('');
  }
  // what drives the system: the VLM, and whether it commands a learned low-level policy
  function modelOf(s) {
    var policy = s.group === 'e2e' || s.group === 'orch' || s.group === 'ref' || (s.group === 'sub' && s.policy);
    if (s.group === 'ref') return 'Simulator state + policy';
    if (s.vlm) return s.vlm + (policy ? ' + policy' : '');
    if (s.group === 'orch') return 'Fixed plan + policy';
    return 'Policy only';
  }
  // submitted systems join the paper's: one row each, and a column in every per-suite table
  function merge(entries) {
    if (!entries.length) return;
    D.groups.push({ id: 'sub', name: 'Submitted' });
    entries.forEach(function (e) {
      var into = e.benchmark === 'lite' && D.lite ? D.lite.systems : FULL;
      into.push({ liteOnly: e.benchmark === 'lite',
        name: e.name, group: e.privileged ? 'ref' : 'sub', type: { policy: 'Policy', orchestrator: 'Orchestrator', agent: 'Agent' }[e.kind],
        ranked: !e.privileged && ['se', 'ra', 'rl'].every(function (k) { return e.suites[k]; }), cost_ep: null, cost_succ: null, vlm: e.vlm === 'none' ? '' : e.vlm, policy: e.low_level_policy,
        inputs: e.inputs, submission: 'Submitted ' + e.kind, verified: e.verified, evaluated: e.date ? [e.date, e.date] : null, results_url: e.results_url, scale: e.scale,
        mean: e.mean || { v: null }, pooled: e.pooled || { v: null }, progress: e.progress,
        suites: { se: e.suites.se || { v: null }, ra: e.suites.ra || { v: null }, rl: e.suites.rl || { v: null } },
        cost: { se: null, ra: null, rl: null }
      });
      if (e.benchmark === 'lite') return;
      [['horizon', 'success', e.horizon], ['horizon', 'progress', e.horizon_progress], ['category', 'success', e.category]].forEach(function (t) {
        Object.keys(t[2] || {}).forEach(function (suite) {
          Object.keys(t[2][suite]).forEach(function (row) {
            var table = D[t[0]][t[1]][suite];
            (table[row] = table[row] || {})[e.name] = t[2][suite][row];
          });
        });
      });
    });
  }
  function drawCost() {
    var box = document.getElementById('cost-chart');
    if (!box) return;
    var keep = D.systems; D.systems = FULL.concat(D.lite ? D.lite.systems.filter(function (s) { return s.liteOnly && s.cost_ep != null; }) : []);
    var key = '<div class="lb-typekey front-key"><i style="background:' + COLORS.tool + '"></i>AI agent <i style="background:' + COLORS.orch +
      '"></i>AI model + robotics model <i style="background:' + COLORS.e2e + '"></i>robotics-only model' +
      (D.systems.some(function (s) { return s.liteOnly; }) ? '<i class="lite-key"></i>scored on LADDER-Lite only' : '') + '</div>';
    box.innerHTML = '<div class="lb-frontbox">' + frontier({ bare: true }) + frontier({ W: 520, H: 600, L: 56, Z: 110, phone: true, bare: true }) + key + '</div>';
    D.systems = keep;
  }
  // draw at once, so nothing below moves when the page loads; redraw only if submitted systems arrive
  draw(); drawCost();
  fetch('./static/data/submissions.json').then(function (r) { return r.ok ? r.json() : []; })
    .then(function (entries) { if (entries && entries.length) { merge(entries); draw(); drawCost(); } })
    .catch(function (e) { console.warn('submissions not shown:', e); });
});
