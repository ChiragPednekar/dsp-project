"""
The transport: Play / Pause / Stop plus an instant A/B switch.

Both the original and the processed audio are loaded into the component at
once, as two <audio> elements. Switching A/B copies the playhead from one to
the other and starts it, so the comparison is instantaneous and level-matched
in time — you hear the same moment of the track both ways rather than
restarting it. A plain Streamlit st.audio widget cannot do that, which is why
this is a custom component.

Playback position survives a rerun (moving a slider re-renders the component)
via sessionStorage, so tweaking the EQ mid-playback does not jump the track
back to zero.
"""

from __future__ import annotations

import base64

PLAYER_HEIGHT = 210

_TEMPLATE = """
<style>
  :root {
    --bg: #161b22;
    --panel: #0d1117;
    --border: #2a2f3a;
    --text: #c9d1d9;
    --muted: #8b949e;
    --accent: #00d4a0;
    --accent-dim: #0a7f62;
    --orig: #5b6b7f;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: transparent;
    color: var(--text);
  }
  .wrap {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px 16px;
  }
  .row { display: flex; align-items: center; gap: 8px; }
  .controls { margin-bottom: 12px; flex-wrap: wrap; }
  button {
    font: inherit;
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 7px;
    padding: 8px 15px;
    cursor: pointer;
    transition: background .12s, border-color .12s, color .12s;
    display: inline-flex;
    align-items: center;
    gap: 7px;
  }
  button:hover:not(:disabled) { background: #1c232d; border-color: #3d4450; }
  button:active:not(:disabled) { transform: translateY(1px); }
  button:disabled { opacity: .4; cursor: not-allowed; }
  button.primary { border-color: var(--accent-dim); color: var(--accent); }
  button.primary:hover:not(:disabled) { background: #0f2b25; border-color: var(--accent); }
  .spacer { flex: 1; }
  .ab {
    border-color: var(--accent-dim);
    background: #0f2b25;
    color: var(--accent);
    min-width: 168px;
    justify-content: center;
  }
  .ab.original {
    border-color: #3d4450;
    background: var(--panel);
    color: var(--orig);
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
  .seek { gap: 11px; }
  input[type=range] {
    -webkit-appearance: none;
    appearance: none;
    flex: 1;
    height: 5px;
    border-radius: 3px;
    background: var(--border);
    outline: none;
    cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 14px; height: 14px;
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--panel);
    cursor: pointer;
  }
  input[type=range]::-moz-range-thumb {
    width: 14px; height: 14px; border: 2px solid var(--panel);
    border-radius: 50%; background: var(--accent); cursor: pointer;
  }
  .time {
    font-variant-numeric: tabular-nums;
    font-size: 12px;
    color: var(--muted);
    min-width: 42px;
  }
  .status {
    margin-top: 11px;
    font-size: 11.5px;
    color: var(--muted);
    min-height: 15px;
    line-height: 1.5;
  }
  .status b { color: var(--accent); font-weight: 600; }
  .status b.orig { color: var(--orig); }
</style>

<div class="wrap">
  <div class="row controls">
    <button id="play" class="primary">&#9654;&nbsp; Play</button>
    <button id="pause">&#10073;&#10073;&nbsp; Pause</button>
    <button id="stop">&#9632;&nbsp; Stop</button>
    <div class="spacer"></div>
    <button id="ab" class="ab" title="Instantly compare original vs EQ'd">
      <span class="dot"></span><span id="ablabel">Hearing: EQ'd</span>
    </button>
  </div>

  <div class="row seek">
    <span class="time" id="cur">0:00</span>
    <input type="range" id="seek" min="0" max="1000" value="0" step="1">
    <span class="time" id="dur">0:00</span>
  </div>

  <div class="status" id="status"></div>

  <audio id="audioA" preload="auto" src="__SRC_A__"></audio>
  <audio id="audioB" preload="auto" src="__SRC_B__"></audio>
</div>

<script>
(function () {
  const A = document.getElementById('audioA');   // original
  const B = document.getElementById('audioB');   // processed
  const playBtn  = document.getElementById('play');
  const pauseBtn = document.getElementById('pause');
  const stopBtn  = document.getElementById('stop');
  const abBtn    = document.getElementById('ab');
  const abLabel  = document.getElementById('ablabel');
  const seek     = document.getElementById('seek');
  const curEl    = document.getElementById('cur');
  const durEl    = document.getElementById('dur');
  const statusEl = document.getElementById('status');

  // Bumped whenever the processed audio changes, so a stale saved position
  // from a different clip is never restored onto a new one.
  const TOKEN = "__TOKEN__";
  const STORE = 'eq_player_state';

  let active = 'B';        // 'A' = original, 'B' = EQ'd
  let scrubbing = false;

  // When Streamlit swaps in a new render, this document is torn down and its
  // <audio> elements emit a final timeupdate sitting at duration with
  // paused = true. Left unguarded that write lands in sessionStorage *after*
  // the real playhead and the next render restores the end of the clip
  // instead of where the user actually was. Two guards, because the teardown
  // event and pagehide do not have a guaranteed order:
  //   1. stop saving once the page is going away, and
  //   2. never let a timeupdate from a paused element persist a position.
  let unloading = false;
  const markUnloading = () => { unloading = true; };
  window.addEventListener('pagehide', markUnloading);
  window.addEventListener('beforeunload', markUnloading);
  window.addEventListener('unload', markUnloading);

  const cur = () => (active === 'A' ? A : B);
  const other = () => (active === 'A' ? B : A);

  function fmt(s) {
    if (!isFinite(s) || s < 0) s = 0;
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60);
    return m + ':' + String(r).padStart(2, '0');
  }

  function saveState() {
    if (unloading) return;
    try {
      sessionStorage.setItem(STORE, JSON.stringify({
        token: TOKEN,
        t: cur().currentTime,
        playing: !cur().paused,
        active: active
      }));
    } catch (e) { /* private mode — position just will not persist */ }
  }

  function setStatus(msg) { statusEl.innerHTML = msg; }

  function describe() {
    const which = active === 'A'
      ? '<b class="orig">ORIGINAL</b> (EQ bypassed)'
      : "<b>EQ'd</b> (filter chain applied)";
    setStatus('Now playing: ' + which +
      ' &nbsp;·&nbsp; press A/B to switch instantly at the same position.');
  }

  function paintAB() {
    if (active === 'A') {
      abBtn.classList.add('original');
      abLabel.textContent = 'Hearing: Original';
    } else {
      abBtn.classList.remove('original');
      abLabel.textContent = "Hearing: EQ'd";
    }
    describe();
  }

  function tick() {
    const a = cur();
    const d = a.duration;
    if (isFinite(d) && d > 0) {
      durEl.textContent = fmt(d);
      if (!scrubbing) seek.value = String(Math.round((a.currentTime / d) * 1000));
    }
    curEl.textContent = fmt(a.currentTime);
  }

  playBtn.addEventListener('click', () => {
    other().pause();
    cur().play().then(describe).catch(err => {
      setStatus('Playback was blocked by the browser: ' + err.message);
    });
    saveState();
  });

  pauseBtn.addEventListener('click', () => {
    A.pause(); B.pause();
    setStatus('Paused at ' + fmt(cur().currentTime) + '.');
    saveState();
  });

  stopBtn.addEventListener('click', () => {
    A.pause(); B.pause();
    A.currentTime = 0; B.currentTime = 0;
    seek.value = '0';
    tick();
    setStatus('Stopped — playhead reset to the start.');
    saveState();
  });

  // The A/B switch: hand the playhead across so the same instant of the
  // track is heard both ways. This is the whole point of the component.
  abBtn.addEventListener('click', () => {
    const from = cur();
    const wasPlaying = !from.paused;
    const t = from.currentTime;

    from.pause();
    active = (active === 'A') ? 'B' : 'A';

    const to = cur();
    try { to.currentTime = t; } catch (e) { /* not seekable yet */ }
    if (wasPlaying) {
      to.play().catch(() => {});
    }
    paintAB();
    tick();
    saveState();
  });

  seek.addEventListener('input', () => {
    scrubbing = true;
    const d = cur().duration;
    if (isFinite(d) && d > 0) curEl.textContent = fmt((seek.value / 1000) * d);
  });

  seek.addEventListener('change', () => {
    const d = cur().duration;
    if (isFinite(d) && d > 0) {
      const t = (seek.value / 1000) * d;
      // Keep both elements aligned so a later A/B switch lands in the same place.
      try { A.currentTime = t; B.currentTime = t; } catch (e) {}
    }
    scrubbing = false;
    tick();
    saveState();
  });

  [A, B].forEach(el => {
    el.addEventListener('timeupdate', () => {
      if (el !== cur()) return;
      tick();
      // Only a genuinely playing element may persist a position; see the
      // teardown note above. Pausing and stopping save from their handlers.
      if (!el.paused) saveState();
    });
    el.addEventListener('loadedmetadata', tick);
    el.addEventListener('ended', () => {
      setStatus('Reached the end of the clip.');
      tick();
    });
    el.addEventListener('error', () => {
      setStatus('This browser could not decode the rendered audio.');
    });
  });

  // Seeking is only trustworthy once an element reports HAVE_ENOUGH_DATA.
  // libsndfile writes MP3 without a Xing/LAME seek table, so setting
  // currentTime on a partly-buffered element makes the browser estimate the
  // byte offset and overshoot to the end of the clip. The audio is a data:
  // URI, so waiting for full buffering costs milliseconds and no network.
  function whenBuffered(el) {
    return new Promise(resolve => {
      if (el.readyState >= 4) return resolve();
      let settled = false;
      const finish = () => {
        if (settled) return;
        settled = true;
        el.removeEventListener('canplaythrough', finish);
        clearTimeout(timer);
        resolve();
      };
      el.addEventListener('canplaythrough', finish);
      // Never leave the transport dead if the event never arrives.
      const timer = setTimeout(finish, 8000);
    });
  }

  // Restore position across the rerun that a slider move causes.
  async function restore() {
    let s = null;
    try { s = JSON.parse(sessionStorage.getItem(STORE) || 'null'); } catch (e) {}

    paintAB();
    tick();

    if (!s || s.token !== TOKEN) { describe(); return; }
    if (s.active === 'A' || s.active === 'B') { active = s.active; paintAB(); }

    await Promise.all([whenBuffered(A), whenBuffered(B)]);

    let t = Number(s.t) || 0;
    const dur = cur().duration;
    // Resuming a hair before the end just plays silence and stops.
    if (!isFinite(t) || t < 0 || (isFinite(dur) && t >= dur - 0.25)) t = 0;

    try { A.currentTime = t; B.currentTime = t; } catch (e) {}
    tick();

    if (s.playing) {
      cur().play().then(() => {
        setStatus('Resumed at ' + fmt(t) + ' with the updated EQ.');
      }).catch(() => {
        setStatus('EQ updated — press Play to hear it from ' + fmt(t) + '.');
      });
    }
  }

  restore();

  setInterval(() => { if (!cur().paused) tick(); }, 200);
})();
</script>
"""


def _data_uri(payload: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def render_html(
    original_bytes: bytes,
    processed_bytes: bytes,
    mime: str,
    token: str,
) -> str:
    """
    Build the transport's HTML.

    `token` should change whenever the processed audio changes; it is what
    tells the restore logic that a saved playhead belongs to this render.
    """
    return (
        _TEMPLATE
        .replace("__SRC_A__", _data_uri(original_bytes, mime))
        .replace("__SRC_B__", _data_uri(processed_bytes, mime))
        .replace("__TOKEN__", token)
    )
