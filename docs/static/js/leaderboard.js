// LADDER leaderboard: three views drawn from the JSON embedded in leaderboard.html (all numbers are the paper's).
document.addEventListener('DOMContentLoaded', function () {
  var D = JSON.parse(document.getElementById('lb-data').textContent);
  var SUITES = D.suites;
  var GROUP = {}; D.groups.forEach(function (g) { GROUP[g.id] = g.name; });
  var COLORS = { e2e: '#9aa0aa', orch: '#e08a1e', tool: '#2f5bea', ref: '#9aa0aa' };
  var state = { view: 'overall', sort: 'mean', suite: 'mean', metric: 'success' };
  var $ = function (s) { return document.querySelector(s); };
  var fmt = function (v) { return v == null ? '&ndash;' : v.toFixed(1); };
  var avg = function (xs) { xs = xs.filter(function (x) { return x != null; }); return xs.length ? xs.reduce(function (a, b) { return a + b; }, 0) / xs.length : null; };

  function seg(items, value, onpick) {
    var box = document.createElement('div');
    box.className = 'lb-seg';
    items.forEach(function (it) {
      var b = document.createElement('button');
      b.type = 'button';
      b.className = it[0] === value ? 'is-active' : '';
      b.innerHTML = it[1];
      b.onclick = function () { onpick(it[0]); draw(); };
      box.appendChild(b);
    });
    return box;
  }
  function name(s) {
    return '<span class="lb-name"><i style="background:' + COLORS[s.group] + '"></i>' + s.name +
      '</span><span class="lb-kind">' + GROUP[s.group] + (s.vlm ? ' &middot; ' + s.vlm : '') + '</span>';
  }
  var SUITE_ITEMS = [['mean', 'All suites']].concat(SUITES.map(function (s) { return [s.id, s.name]; }));

  // ---------- overall: one ranked table
  var SCORES = {
    mean: { label: 'All suites', get: function (s) { return s.mean.v; }, pm: function (s) { return s.mean.pm; } },
    se: { label: SUITES[0].name, get: function (s) { return s.suites.se.v; }, pm: function (s) { return s.suites.se.pm; } },
    ra: { label: SUITES[1].name, get: function (s) { return s.suites.ra.v; }, pm: function (s) { return s.suites.ra.pm; } },
    rl: { label: SUITES[2].name, get: function (s) { return s.suites.rl.v; }, pm: function (s) { return s.suites.rl.pm; } },
    progress: { label: 'Subgoal progress', get: function (s) { return s.progress.mean; } }
  };
  function overall(view, controls) {
    controls.appendChild(seg([['mean', 'All suites'], ['se', SUITES[0].name], ['ra', SUITES[1].name], ['rl', SUITES[2].name], ['progress', 'Subgoal progress']],
      state.sort, function (v) { state.sort = v; }));
    var sc = SCORES[state.sort];
    var ranked = D.systems.filter(function (s) { return s.group !== 'ref'; }).sort(function (a, b) { return sc.get(b) - sc.get(a); });
    var rows = ranked.concat(D.systems.filter(function (s) { return s.group === 'ref'; }));
    var rank = 0, last = null;
    var html = '<table class="lb-table"><thead><tr><th class="lb-rank">#</th><th class="lb-left">System</th><th class="lb-score">' + sc.label + '</th>' +
      SUITES.map(function (s) { return '<th class="lb-num' + (state.sort === s.id ? ' is-on' : '') + '">' + s.name.replace(/-/g, '&#8209;') + '</th>'; }).join('') +
      '<th class="lb-num' + (state.sort === 'progress' ? ' is-on' : '') + '">Subgoal<br>progress</th><th class="lb-num lb-cost">Cost per<br>episode</th></tr></thead><tbody>';
    rows.forEach(function (s, i) {
      var ref = s.group === 'ref', v = sc.get(s), pm = sc.pm ? sc.pm(s) : null;
      if (!ref) { if (v !== last) rank = i + 1; last = v; }
      var cost = avg([s.cost.se, s.cost.ra, s.cost.rl]);
      html += '<tr class="' + (ref ? 'lb-ref' : '') + '">' +
        '<td class="lb-rank">' + (ref ? '' : rank) + '</td><td class="lb-left">' + name(s) + '</td>' +
        '<td class="lb-score"><div class="lb-scorebox"><div class="lb-meter"><span style="width:' + v + '%"></span></div><b>' + fmt(v) + '</b>' +
        '<small>' + (pm != null ? '&plusmn;' + fmt(pm) : '') + '</small></div></td>' +
        SUITES.map(function (x) { return '<td class="lb-num' + (state.sort === x.id ? ' is-on' : '') + '">' + fmt(s.suites[x.id].v) + '</td>'; }).join('') +
        '<td class="lb-num' + (state.sort === 'progress' ? ' is-on' : '') + '">' + fmt(s.progress.mean) + '</td>' +
        '<td class="lb-num lb-cost">' + (cost ? '$' + cost.toFixed(cost < 0.1 ? 3 : 2) : '&ndash;') + '</td></tr>';
    });
    view.innerHTML = '<div class="lb-wrap">' + html + '</tbody></table></div>';
    return 'Task success (%): share of episodes in which every subgoal is completed, with 95% intervals. "All suites" weighs the ' +
      'three suites equally (the paper\'s Overall column pools their episodes instead). Tool selection systems use ground-truth depth in simulation. The privileged verifier reads the simulator state, so it is a reference and is not ranked. Cost: hosted VLM only, at list prices.';
  }

  // ---------- by horizon and by category: one row per system, one column per subgoal count or category
  function cell(table, row, sys) {
    if (state.suite !== 'mean') { var r = table[state.suite][row]; return r ? r[sys] : null; }
    return avg(SUITES.map(function (s) { var r = table[s.id][row]; return r ? r[sys] : null; }));
  }
  function grid(table, rows, label) {
    var systems = D.systems.slice().sort(function (a, b) { return cell(table, 'All tasks', b.name) - cell(table, 'All tasks', a.name); });
    var html = '<table class="lb-table lb-grid"><thead><tr><th class="lb-left">System</th>' +
      rows.map(function (r) { return '<th class="lb-num' + (r === 'All tasks' ? ' lb-all' : '') + '">' + label(r) + '</th>'; }).join('') + '</tr></thead><tbody>';
    systems.forEach(function (s) {
      html += '<tr class="' + (s.group === 'ref' ? 'lb-ref' : '') + '"><td class="lb-left">' + name(s) + '</td>' +
        rows.map(function (r) {
          var v = cell(table, r, s.name);
          return '<td class="lb-num lb-heat' + (r === 'All tasks' ? ' lb-all' : '') + '" style="--v:' + (v == null ? 0 : v / 100).toFixed(3) + '">' + fmt(v) + '</td>';
        }).join('') + '</tr>';
    });
    return '<div class="lb-wrap">' + html + '</tbody></table></div>';
  }
  // one colour per system (Tableau 10); the privileged reference is dashed
  var PALETTE = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#7f7f7f'];
  var colour = function (s) { return PALETTE[D.systems.indexOf(s) % PALETTE.length]; };
  var hidden = {};
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
      return '<button type="button" class="lb-key' + (hidden[s.name] ? ' is-off' : '') + '" data-sys="' + s.name + '"><i style="background:' + colour(s) + '"></i>' + s.name + '</button>';
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
      tip.innerHTML = '<b>' + xs[i] + '</b>' + items.map(function (p) {
        return '<span><i style="background:' + colour(p[0]) + '"></i>' + p[0].name + '<em>' + fmt(p[1]) + '</em></span>';
      }).join('');
      tip.hidden = false;
      var left = gx * r.width / W;
      tip.style.left = (left > r.width * 0.6 ? left - tip.offsetWidth - 14 : left + 14) + 'px';
    };
    svg.onmousemove = move;
    svg.onmouseleave = function () { tip.hidden = true; guide.style.display = 'none'; };
  }

  function horizon(view, controls) {
    controls.appendChild(seg([['success', 'Task success'], ['progress', 'Subgoal progress']], state.metric, function (v) { state.metric = v; }));
    controls.appendChild(seg(SUITE_ITEMS, state.suite, function (v) { state.suite = v; }));
    var table = D.horizon[state.metric];
    var rows = Object.keys(table.se).filter(function (k) { return k !== '_n'; });
    var xs = rows.filter(function (r) { return r !== 'All tasks'; });
    view.innerHTML = chart(table, xs) + grid(table, rows, function (r) { return r === 'All tasks' ? 'All' : parseInt(r, 10); });
    wire(view, table, xs);
    return (state.metric === 'success' ? 'Task success' : 'Subgoal progress') + ' (%) by number of subgoals' +
      (state.suite === 'mean' ? ', averaged over the three suites.' : '.') +
      ' Hover the plot to read every system at one length, hover a name to pick out its line, click a name to hide or show it.';
  }
  function category(view, controls) {
    controls.appendChild(seg(SUITE_ITEMS, state.suite, function (v) { state.suite = v; }));
    var table = D.category.success;
    var rows = Object.keys(table.se).filter(function (k) { return k !== '_n'; });
    view.innerHTML = grid(table, rows, function (r) { return r === 'All tasks' ? 'All' : r.replace(' destination', '<br>destination'); });
    return 'Task success (%) per task category' + (state.suite === 'mean' ? ', averaged over the three suites.' : '.');
  }

  var VIEWS = [['overall', 'Overall', overall], ['horizon', 'By horizon', horizon], ['category', 'By task category', category]];
  function draw() {
    var views = $('.lb-views'), controls = $('.lb-controls'), view = $('.lb-view');
    views.innerHTML = ''; controls.innerHTML = '';
    VIEWS.forEach(function (v) {
      var b = document.createElement('button');
      b.type = 'button'; b.className = v[0] === state.view ? 'is-active' : ''; b.textContent = v[1];
      b.onclick = function () { state.view = v[0]; draw(); };
      views.appendChild(b);
    });
    $('.lb-note').innerHTML = VIEWS.filter(function (v) { return v[0] === state.view; })[0][2](view, controls);
  }
  draw();
});
