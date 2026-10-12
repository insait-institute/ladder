document.addEventListener('DOMContentLoaded', function () {
  // Tabs: show one pane of a group; pause the videos of the panes that get hidden.
  // Each tab is a link to its pane (works without JavaScript); with it, the click switches panes in place.
  document.querySelectorAll('a[data-tabs]').forEach(function (a) {
    a.addEventListener('keydown', function (e) {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      var all = Array.prototype.slice.call(document.querySelectorAll('a[data-tabs="' + a.dataset.tabs + '"]'));
      var next = all[(all.indexOf(a) + (e.key === 'ArrowRight' ? 1 : all.length - 1)) % all.length];
      e.preventDefault(); next.focus(); next.click();
    });
    a.addEventListener('click', function (e) {
      e.preventDefault();
      var group = a.dataset.tabs;
      document.querySelectorAll('a[data-tabs="' + group + '"]').forEach(function (b) {
        b.parentElement.classList.toggle('is-active', b === a);
        b.setAttribute('aria-selected', b === a ? 'true' : 'false');
        b.tabIndex = b === a ? 0 : -1;
      });
      document.querySelectorAll('.tab-pane[data-tabs="' + group + '"]').forEach(function (p) {
        var show = p.id === a.dataset.pane;
        p.classList.toggle('is-hidden', !show);
        if (!show) { p.querySelectorAll('video').forEach(function (v) { v.pause(); }); }
      });
      if (window.__watch) { window.__watch(); }
    });
  });

  document.querySelectorAll('[role="tablist"]').forEach(function (list) {
    list.querySelectorAll('a[data-tabs]').forEach(function (b) { b.tabIndex = b.getAttribute('aria-selected') === 'true' ? 0 : -1; });
  });

  // Dropdown pickers: the same panes as tabs, chosen from a list.
  document.querySelectorAll('select[data-tabs]').forEach(function (sel) {
    sel.addEventListener('change', function () {
      document.querySelectorAll('.tab-pane[data-tabs="' + sel.dataset.tabs + '"]').forEach(function (p) {
        var show = p.id === sel.value;
        p.classList.toggle('is-hidden', !show);
        if (!show) { p.querySelectorAll('video').forEach(function (v) { v.pause(); }); }
      });
      if (window.__watch) { window.__watch(); }
    });
  });

  // The LADDER video plays by itself, muted, once it scrolls into view, and pauses when it leaves. Browsers only allow
  // muted autoplay; the controls unmute it. Once the visitor pauses or seeks it, scrolling no longer restarts it.
  var promo = document.querySelector('video.promo-video');
  if (promo && 'IntersectionObserver' in window && !(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)) {
    var chosen = false;
    promo.addEventListener('pause', function () { if (promo.dataset.auto !== '1') chosen = true; promo.dataset.auto = ''; });
    new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (chosen) return;
        if (e.isIntersecting) { promo.play().catch(function () {}); }
        else if (!promo.paused) { promo.dataset.auto = '1'; promo.pause(); }
      });
    }, { threshold: 0.5 }).observe(promo);
  }



  // Length charts ("The longer the task..."): drawn from their JSON, every system can be switched on in the legend, and
  // hovering reads out each shown line at the nearest task length.
  document.querySelectorAll('.len-box').forEach(function (box) {
    var D = JSON.parse(box.querySelector('.len-data').textContent);
    var plot = box.querySelector('.len-plot'), tip = box.querySelector('.len-tip');
    var phone = window.matchMedia('(max-width: 768px)');
    var esc = function (t) { return String(t).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
    var dims, focus = null, at = -1;
    function geo() {
      dims = phone.matches ? { W: 400, H: 300, L: 44, R: 14, T: 14, B: 50, fs: 14 } : { W: 1100, H: 440, L: 72, R: 30, T: 16, B: 48, fs: 15 };
      dims.x = function (i) { return dims.L + (dims.W - dims.L - dims.R) * i / (D.groups.length - 1); };
      dims.y = function (v) { return dims.T + (dims.H - dims.T - dims.B) * (1 - v / 100); };
    }
    function draw() {
      geo();
      var g = dims, s = '<svg viewBox="0 0 ' + g.W + ' ' + g.H + '" class="len-chart" style="font-size:' + g.fs + 'px" role="img" aria-label="' +
        esc(box.querySelector('.len-title').textContent) + '">';
      [0, 25, 50, 75, 100].forEach(function (v) {
        s += '<line x1="' + g.L + '" x2="' + (g.W - g.R) + '" y1="' + g.y(v) + '" y2="' + g.y(v) + '" class="len-grid"/><text x="' + (g.L - 8) + '" y="' + (g.y(v) + g.fs * .35) + '" text-anchor="end">' + v + '%</text>';
      });
      D.groups.forEach(function (lab, i) { s += '<text x="' + g.x(i) + '" y="' + (g.H - g.B + g.fs * 1.5) + '" text-anchor="middle">' + lab + '</text>'; });
      s += '<text x="' + ((g.L + g.W - g.R) / 2) + '" y="' + (g.H - 4) + '" text-anchor="middle" class="len-axis">' + esc(D.axis) + '</text>';
      if (at >= 0) s += '<line class="len-guide" x1="' + g.x(at) + '" x2="' + g.x(at) + '" y1="' + g.T + '" y2="' + (g.H - g.B) + '"/>';
      // shown lines, the focused one last so it is drawn on top
      var on = D.series.filter(function (r) { return r.on; });
      on.sort(function (a, b) { return (a.id === focus) - (b.id === focus); });
      on.forEach(function (r) {
        var dim = focus && r.id !== focus;
        s += '<g class="len-line' + (dim ? ' is-dim' : '') + '" data-id="' + esc(r.id) + '"><polyline points="' + r.values.map(function (v, i) { return g.x(i) + ',' + g.y(v); }).join(' ') +
          '" fill="none" stroke="' + r.colour + '" stroke-width="' + (r.id === focus ? 5 : 3.5) + '" stroke-linejoin="round"' + (r.ref ? ' stroke-dasharray="9 7"' : '') + '/>' +
          r.values.map(function (v, i) { return '<circle cx="' + g.x(i) + '" cy="' + g.y(v) + '" r="' + (i === at ? 7 : 5) + '" fill="' + r.colour + '"/>'; }).join('') + '</g>';
      });
      s += '<rect class="len-hit" x="' + g.L + '" y="' + g.T + '" width="' + (g.W - g.L - g.R) + '" height="' + (g.H - g.T - g.B) + '" fill="transparent"/></svg>';
      plot.querySelector('svg').outerHTML = s;
      box.querySelectorAll('.len-key').forEach(function (k) {
        var r = D.series.filter(function (x) { return x.id === k.dataset.id; })[0];
        k.classList.toggle('is-off', !r.on); k.setAttribute('aria-pressed', r.on ? 'true' : 'false');
        k.classList.toggle('is-focus', r.id === focus);
      });
      wire();
    }
    function read(i) {
      var rows = D.series.filter(function (r) { return r.on; }).slice().sort(function (a, b) { return b.values[i] - a.values[i]; });
      if (!rows.length) { tip.hidden = true; return; }
      tip.innerHTML = '<b>' + D.groups[i] + ' ' + esc(D.axis.replace(/^objects to move in one task$/, 'objects')) + '</b>' + rows.map(function (r) {
        return '<span' + (r.id === focus ? ' class="is-focus"' : '') + '><i style="background:' + r.colour + '"></i>' + esc(r.name) + '<em>' + r.values[i].toFixed(1) + '%</em></span>';
      }).join('');
      tip.hidden = false;
      var svg = plot.querySelector('svg'), rect = svg.getBoundingClientRect(), left = dims.x(i) * rect.width / dims.W;
      var x = left > rect.width * .55 ? left - tip.offsetWidth - 16 : left + 16;
      tip.style.left = Math.max(0, Math.min(rect.width - tip.offsetWidth, x)) + 'px';
    }
    function wire() {
      var svg = plot.querySelector('svg');
      var pick = function (e) {
        var r = svg.getBoundingClientRect(), px = (e.clientX - r.left) * dims.W / r.width;
        var i = Math.max(0, Math.min(D.groups.length - 1, Math.round((px - dims.L) / (dims.W - dims.L - dims.R) * (D.groups.length - 1))));
        // the line nearest the pointer is focused, so a crowded point can still be told apart
        var py = (e.clientY - r.top) * dims.H / r.height, best = null, gap = 1e9;
        D.series.forEach(function (row) { if (row.on) { var d = Math.abs(dims.y(row.values[i]) - py); if (d < gap) { gap = d; best = row.id; } } });
        var f = gap < 40 ? best : null;
        if (i !== at || f !== focus) { at = i; focus = f; draw(); }
        read(i);
      };
      svg.addEventListener('mousemove', pick);
      svg.addEventListener('click', pick);
      svg.addEventListener('mouseleave', function () { at = -1; focus = null; tip.hidden = true; draw(); });
    }
    box.querySelectorAll('.len-key').forEach(function (k) {
      k.addEventListener('click', function () {
        D.series.forEach(function (r) { if (r.id === k.dataset.id) r.on = !r.on; });
        draw(); if (at >= 0) read(at);
      });
      k.addEventListener('mouseenter', function () { var r = D.series.filter(function (x) { return x.id === k.dataset.id; })[0]; if (r.on) { focus = r.id; draw(); } });
      k.addEventListener('mouseleave', function () { focus = null; draw(); });
    });
    (phone.addEventListener ? phone.addEventListener('change', draw) : phone.addListener(draw));
    draw();
  });

  // The systems explainer (details page): one video, and the chosen system's steps drawn over it in time with the video.
  var ex = document.getElementById('explainer');
  if (ex) {
    var D = JSON.parse(document.getElementById('explainer-data').textContent);
    var clip = ex.querySelector('.ex-clip'), feed = ex.querySelector('.ex-feed'), tags = ex.querySelector('.ex-tags');
    var steps = ex.querySelectorAll('.ex-steps li'), play = ex.querySelector('.ex-play');
    var icon = function (k) { var e = ex.querySelector('.ex-icons [data-k="' + k + '"]'); return e ? e.innerHTML : ''; };
    var FEED = { say: 1, grip: 1, timer: 1, check: 1, plan: 1, code: 1, goal: 1, path: 1, call: 1, ok: 1, truth: 1, boxes: 1, points: 1 };
    var LABEL = { boxes: 'Found the objects', points: 'Pointed at the objects' };
    var cur = null, shown = -1;
    var esc = function (t) { return String(t).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };

    function choose(id) {
      cur = D.systems[id]; shown = -1;
      ex.querySelectorAll('.ex-pick').forEach(function (b) {
        var on = b.dataset.sys === id; b.classList.toggle('is-active', on); b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      ex.dataset.sys = id;
      ex.querySelector('.ex-in').innerHTML = cur.inputs.map(function (i) {
        return '<span class="ex-chip">' + icon(i) + esc(i) + '</span>';
      }).join('');
      ex.querySelector('.ex-brain').textContent = cur.brain;
      ex.querySelector('.ex-out').textContent = cur.output;
      ex.querySelector('.ex-desc').textContent = cur.desc;
      var kinds = cur.beats.map(function (b) { return b[1]; });
      ex.classList.toggle('has-steps', kinds.indexOf('step') >= 0);
      if (!clip.paused || clip.currentTime > 0) { clip.currentTime = 0; }
      draw();
    }

    function draw() {
      if (!cur) return;
      var t = clip.currentTime || 0, past = cur.beats.filter(function (b) { return b[0] <= t; });
      // the step the robotics model is working on, and the steps already done
      var step = -1;
      past.forEach(function (b) { if (b[1] === 'step') step = b[2]; });
      steps.forEach(function (li, i) {
        li.classList.toggle('is-now', i === step && t < D.steps[i][1] + 0.2);
        li.classList.toggle('is-done', t >= D.steps[i][1] + 0.2);
      });
      // what the robot is told right now, as a caption on the video
      var said = '';
      past.forEach(function (b) {
        if (b[1] === 'say') said = 'Told: \u201c' + b[2] + '\u201d';
        if (b[1] === 'step') said = 'Told: \u201c' + steps[b[2]].textContent + '\u201d';
        if (b[1] === 'call') said = b[2];
      });
      tags.innerHTML = said ? '<span class="ex-caption">' + esc(said) + '</span>' : '';
      // boxes, points and the planned path are drawn on the still opening frame (the first 3 s), where the system looks
      // before the arm moves; once the arm moves the camera's view of the table changes, and they fade out
      var boxes = '', points = '', still = t < 3;
      past.forEach(function (b) {
        if (b[1] === 'boxes' && still) boxes = b[2];
        if (b[1] === 'points' && still) points = b[2];
      });
      ex.querySelectorAll('.ex-box').forEach(function (e) { e.classList.toggle('is-on', boxes.split(',').indexOf(e.dataset.obj) >= 0); });
      ex.querySelectorAll('.ex-point').forEach(function (e, i) {
        e.classList.toggle('is-on', points.split(',').indexOf(['bin', 'banana', 'orange'][i]) >= 0);
      });
      ex.classList.toggle('show-path', past.some(function (b) { return b[1] === 'path'; }) && t < 3);
      // the log of what the system did, newest first
      var items = past.filter(function (b) { return FEED[b[1]]; });
      if (items.length !== shown) {
        shown = items.length;
        feed.innerHTML = items.slice(-4).reverse().map(function (b, i) {
          var text = LABEL[b[1]] || b[2];
          var body = b[1] === 'code' ? '<pre>' + esc(text) + '</pre>' : '<span>' + esc(text) + '</span>';
          return '<div class="ex-ev ev-' + b[1] + (i === 0 ? ' is-new' : '') + '">' + icon(b[1]) + body + '</div>';
        }).join('');
      }
    }

    ex.querySelectorAll('.ex-pick').forEach(function (b) {
      b.addEventListener('click', function () { choose(b.dataset.sys); if (clip.paused && !ex.dataset.stopped) clip.play().catch(function () {}); });
    });
    var tick = function () { draw(); if (!clip.paused) requestAnimationFrame(tick); };
    clip.addEventListener('play', function () { play.innerHTML = icon('pause'); play.setAttribute('aria-label', 'Pause'); tick(); });
    clip.addEventListener('pause', function () { play.innerHTML = icon('play'); play.setAttribute('aria-label', 'Play'); });
    clip.addEventListener('seeked', draw);
    play.addEventListener('click', function () {
      if (clip.paused) { delete ex.dataset.stopped; clip.play().catch(function () {}); } else { ex.dataset.stopped = '1'; clip.pause(); }
    });
    play.innerHTML = icon('play'); play.setAttribute('aria-label', 'Play');
    choose('policy');
    // plays by itself while on screen (muted, looping), unless the visitor paused it or prefers less motion
    var still = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if ('IntersectionObserver' in window && !still) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting && !ex.dataset.stopped) { clip.preload = 'auto'; clip.play().catch(function () {}); }
          else if (!e.isIntersecting) { clip.pause(); }
        });
      }, { threshold: 0.4 }).observe(clip);
    }
  }

  // Posters: fetched only as their video nears the screen (data-poster), so the first screen loads alone.
  var posters = document.querySelectorAll('video[data-poster]:not(.rollout)');
  if ('IntersectionObserver' in window) {
    var posterIo = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.poster = e.target.dataset.poster; posterIo.unobserve(e.target); }
      });
    }, { rootMargin: '600px 0px' });
    posters.forEach(function (v) { posterIo.observe(v); });
  } else {
    posters.forEach(function (v) { v.poster = v.dataset.poster; });
  }

  // The home page's clips: no download until one nears the screen, then it plays while visible (muted, looping).
  var autos = document.querySelectorAll('video[data-autoplay]');
  if ('IntersectionObserver' in window) {
    var autoIo = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.preload = 'auto'; e.target.play().catch(function () {}); }
        else { e.target.pause(); }
      });
    }, { rootMargin: '200px 0px' });
    autos.forEach(function (v) { autoIo.observe(v); });
  } else {
    autos.forEach(function (v) { v.autoplay = true; v.preload = 'auto'; });
  }

  // Play a rollout only while it is on screen, so a page of many videos stays light.
  var videos = document.querySelectorAll('video.rollout');
  if (!('IntersectionObserver' in window)) {
    videos.forEach(function (v) { v.poster = v.dataset.poster; v.preload = 'metadata'; v.controls = true; });
    return;
  }
  // and fetch its poster only as it nears the screen (the page has well over a hundred)
  var near = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { e.target.poster = e.target.dataset.poster; near.unobserve(e.target); }
    });
  }, { rootMargin: '800px 0px' });
  videos.forEach(function (v) { near.observe(v); });
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting && e.target.offsetParent !== null) { e.target.play().catch(function () {}); }
      else { e.target.pause(); }
    });
  }, { threshold: 0.4 });
  window.__watch = function () { videos.forEach(function (v) { io.unobserve(v); io.observe(v); }); };
  window.__watch();

  // Chart tips: every element with data-tip (failure bars, real-robot bars, cost-chart points, heat-table cells) shows
  // its values on hover, keyboard focus and tap. One tooltip for the page; events are delegated, so charts the
  // leaderboard redraws later get tips too.
  (function () {
    var tip = document.createElement('div');
    tip.className = 'chart-tip'; tip.hidden = true; tip.setAttribute('role', 'tooltip');
    document.body.appendChild(tip);
    var esc = function (t) { return String(t).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
    var cur = null;
    var target = function (e) { return e.target.closest ? e.target.closest('[data-tip]') : null; };
    function show(el, x, y) {
      if (el !== cur) {
        var d = JSON.parse(el.dataset.tip);
        tip.innerHTML = '<b>' + esc(d.h) + '</b>' + d.r.map(function (r) {
          return '<span>' + (r[0] ? '<i style="background:' + r[0] + '"></i>' : '') + esc(r[1]) + '<em>' + esc(r[2]) + '</em></span>';
        }).join('');
        if (cur) cur.classList.remove('is-tip');
        cur = el; el.classList.add('is-tip');
      }
      tip.hidden = false;
      var w = tip.offsetWidth, h = tip.offsetHeight, vw = document.documentElement.clientWidth, vh = window.innerHeight;
      var left = x + 14 + w > vw - 8 ? x - 14 - w : x + 14, top = y + 14 + h > vh - 8 ? y - 14 - h : y + 14;
      tip.style.left = Math.max(8, left) + window.scrollX + 'px';
      tip.style.top = Math.max(8, top) + window.scrollY + 'px';
    }
    function hide() { tip.hidden = true; if (cur) cur.classList.remove('is-tip'); cur = null; }
    document.addEventListener('mousemove', function (e) { var el = target(e); if (el) show(el, e.clientX, e.clientY); else if (cur) hide(); });
    document.addEventListener('click', function (e) { var el = target(e); if (el) show(el, e.clientX, e.clientY); else hide(); });
    document.addEventListener('focusin', function (e) { var el = target(e); if (el) { var r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.top); } });
    document.addEventListener('focusout', function (e) { if (target(e)) hide(); });
    // a tip opened by tap or focus goes away when the page scrolls; one under the mouse follows the next move
    window.addEventListener('scroll', function () { if (cur && !cur.matches(':hover')) hide(); }, { passive: true });
  })();
  // Copy buttons (BibTeX): copy the code block named by data-copy, then confirm on the button for two seconds.
  document.querySelectorAll('.copy-btn').forEach(function (b) {
    b.addEventListener('click', function () {
      var src = document.getElementById(b.dataset.copy), label = b.querySelector('span');
      if (!src || !navigator.clipboard) return;
      navigator.clipboard.writeText(src.textContent).then(function () {
        b.classList.add('is-done'); label.textContent = 'Copied';
        setTimeout(function () { b.classList.remove('is-done'); label.textContent = 'Copy'; }, 2000);
      });
    });
  });
});
