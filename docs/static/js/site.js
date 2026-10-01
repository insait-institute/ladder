document.addEventListener('DOMContentLoaded', function () {
  // Tabs: show one pane of a group; pause the videos of the panes that get hidden.
  document.querySelectorAll('a[data-tabs]').forEach(function (a) {
    a.addEventListener('click', function () {
      var group = a.dataset.tabs;
      document.querySelectorAll('a[data-tabs="' + group + '"]').forEach(function (b) {
        b.parentElement.classList.toggle('is-active', b === a);
      });
      document.querySelectorAll('.tab-pane[data-tabs="' + group + '"]').forEach(function (p) {
        var show = p.id === a.dataset.pane;
        p.classList.toggle('is-hidden', !show);
        if (!show) { p.querySelectorAll('video').forEach(function (v) { v.pause(); }); }
      });
      if (window.__watch) { window.__watch(); }
    });
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

  // Play a rollout only while it is on screen, so a page of many videos stays light.
  var videos = document.querySelectorAll('video.rollout');
  if (!('IntersectionObserver' in window)) {
    videos.forEach(function (v) { v.preload = 'metadata'; v.controls = true; });
    return;
  }
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting && e.target.offsetParent !== null) { e.target.play().catch(function () {}); }
      else { e.target.pause(); }
    });
  }, { threshold: 0.4 });
  window.__watch = function () { videos.forEach(function (v) { io.unobserve(v); io.observe(v); }); };
  window.__watch();
});
