// The LADDER logo, animated once on load as in the promo video (ladder_promo_video/src/components/Logo.tsx):
// the rails draw, the rungs follow, a small robot climbs one rung per hop, the top rung turns red, then the
// letters of the wordmark rise in. Without JavaScript, or with reduced motion, the final frame is shown.
(function () {
  var box = document.querySelector('.brand-hero');
  if (!box) return;
  var INK = '#111111', RED = '#e63e3e', L = 30, R = 90, TOP = 14, BOT = 146, N = 5;
  var rungY = function (i) { return BOT - 14 - i * ((BOT - TOP - 28) / (N - 1)); };
  var clamp = function (v) { return Math.max(0, Math.min(1, v)); };
  var ease = function (t) { return 1 - Math.pow(1 - t, 3); };
  var inOut = function (t) { return t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
  var lerp = function (f, a, b) { return clamp((f - a) / (b - a)); };
  var svg = box.querySelector('svg.mark'), letters = box.querySelectorAll('.wordmark span');
  var robot = '<rect x="-14" y="2" width="6" height="14" rx="3" fill="' + INK + '"/><rect x="8" y="2" width="6" height="14" rx="3" fill="' + INK + '"/>' +
    '<rect x="-17" y="-22" width="34" height="26" rx="10" fill="' + INK + '"/><circle cx="-6" cy="-10" r="3.2" fill="#fff"/><circle cx="6" cy="-10" r="3.2" fill="#fff"/>' +
    '<line x1="0" y1="-22" x2="0" y2="-30" stroke="' + INK + '" stroke-width="3" stroke-linecap="round"/><circle cx="0" cy="-32" r="3.5" fill="' + RED + '"/>';
  function frame(f) {  // f in video frames (30 per second)
    var rail = ease(lerp(f, 0, 18)), h = '';
    h += '<line x1="' + L + '" y1="' + BOT + '" x2="' + L + '" y2="' + (BOT - (BOT - TOP) * rail) + '" stroke="' + INK + '" stroke-width="7" stroke-linecap="round"/>';
    h += '<line x1="' + R + '" y1="' + BOT + '" x2="' + R + '" y2="' + (BOT - (BOT - TOP) * rail) + '" stroke="' + INK + '" stroke-width="7" stroke-linecap="round"/>';
    var hop = 11, t = Math.max(0, (f - 30) / hop), rung = Math.min(N - 1, Math.floor(t)), ph = Math.min(1, t - rung), atTop = t >= N - 1;
    for (var i = 0; i < N; i++) {
      var r = ease(lerp(f - 8 - i * 3, 0, 10));
      if (r > 0) h += '<line x1="' + L + '" y1="' + (rungY(i) + 2) + '" x2="' + (L + (R - L) * r) + '" y2="' + (rungY(i) + 2) + '" stroke="' + (atTop && i === N - 1 ? RED : INK) + '" stroke-width="5" stroke-linecap="round"/>';
    }
    var y0 = rungY(rung), y1 = rungY(Math.min(N - 1, rung + 1)), k = atTop ? 0 : inOut(ph);
    var y = y0 + (y1 - y0) * k - (atTop ? 0 : Math.sin(Math.PI * ph) * 8);
    var sq = atTop ? 0 : Math.max(0, 1 - ph * 5) * .6, after = f - 30 - hop * (N - 1);
    var bounce = atTop ? Math.max(0, Math.sin(after / 3)) * 3 * Math.exp(-after / 12) : 0;
    h += '<g opacity="' + lerp(f, 22, 30) + '" transform="translate(60,' + (y - bounce - 4) + ') scale(' + 1.15 * (1 + .08 * sq) + ',' + 1.15 * (1 - .1 * sq) + ')">' + robot + '</g>';
    svg.innerHTML = h;
    letters.forEach(function (s, j) {
      var u = ease(lerp(f - 52 - j * 2, 0, 16));
      s.style.opacity = u; s.style.transform = 'translateY(' + (1 - u) * 0.25 + 'em)';
    });
  }
  var END = 52 + 2 * 5 + 16 + 30;
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) { frame(END); return; }
  var t0 = null;
  function tick(now) {
    if (t0 === null) t0 = now;
    var f = (now - t0) / 1000 * 30;
    frame(Math.min(f, END));
    if (f < END) requestAnimationFrame(tick);
  }
  frame(0);
  requestAnimationFrame(tick);
})();
