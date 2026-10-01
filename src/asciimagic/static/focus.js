"use strict";

// "Focus & background" on the Image tab: drag a focus box on a copy of the
// picture, show the detected subject (green) after each render, and make
// sure the subject model is downloaded before it's needed. Relies on app.js
// globals: $, state, render, autoRender, setStatus, readJson.

const stage = $("focus-stage");
const fimg = $("focus-img");
const frect = $("focus-rect");
const fmask = $("focus-mask");
let models = null; // /api/subject/models

// ---------- picture ----------

// app.js swaps #thumb's src on every upload; mirror it here.
new MutationObserver(() => {
  const src = $("thumb").getAttribute("src");
  if (!src) return;
  fimg.src = src;
  stage.hidden = false;
  $("focus-buttons").hidden = false;
  clearBox(false);
  clearMask();
  $("focus-hint").textContent = "Optional: draw a box to convert just part of the picture.";
}).observe($("thumb"), { attributes: true, attributeFilter: ["src"] });

// ---------- focus box ----------

function showBox(b) {
  frect.hidden = !b;
  if (!b) return;
  frect.style.left = `${b[0] * 100}%`;
  frect.style.top = `${b[1] * 100}%`;
  frect.style.width = `${b[2] * 100}%`;
  frect.style.height = `${b[3] * 100}%`;
}

function clearBox(rerender = true) {
  state.focusBox = null;
  showBox(null);
  $("focus-clear").disabled = true;
  if (rerender) render();
}

$("focus-clear").addEventListener("click", () => clearBox());
$("focus-draw").addEventListener("click", () => {
  stage.classList.add("drawing");
  $("focus-hint").textContent = "Drag across the picture to choose the focus area.";
});

function pointFrac(e) {
  const r = fimg.getBoundingClientRect();
  return [
    Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
    Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
  ];
}

stage.addEventListener("pointerdown", (e) => {
  // Dragging always draws; the button just makes that discoverable.
  e.preventDefault();
  stage.setPointerCapture(e.pointerId);
  const [x0, y0] = pointFrac(e);
  const move = (ev) => {
    const [x1, y1] = pointFrac(ev);
    showBox([Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)]);
  };
  const up = (ev) => {
    stage.removeEventListener("pointermove", move);
    stage.removeEventListener("pointerup", up);
    stage.removeEventListener("pointercancel", up);
    stage.classList.remove("drawing");
    const [x1, y1] = pointFrac(ev);
    const b = [Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)];
    if (b[2] < 0.03 || b[3] < 0.03) {
      // A click, not a drag: keep whatever box there was.
      showBox(state.focusBox);
      return;
    }
    state.focusBox = b.map((v) => Math.round(v * 1000) / 1000);
    showBox(state.focusBox);
    $("focus-clear").disabled = false;
    $("focus-hint").textContent = "Only the boxed area is converted. \"Whole picture\" undoes it.";
    render();
  };
  stage.addEventListener("pointermove", move);
  stage.addEventListener("pointerup", up);
  stage.addEventListener("pointercancel", up);
});

// ---------- subject overlay ----------

function clearMask() {
  fmask.getContext("2d").clearRect(0, 0, fmask.width, fmask.height);
  fmask.hidden = true;
}

function drawMask(b64) {
  if (!b64 || !$("show_mask").checked) { clearMask(); return; }
  const im = new Image();
  im.onload = () => {
    fmask.width = im.naturalWidth;
    fmask.height = im.naturalHeight;
    const ctx = fmask.getContext("2d");
    ctx.drawImage(im, 0, 0);
    const px = ctx.getImageData(0, 0, fmask.width, fmask.height);
    const d = px.data;
    for (let i = 0; i < d.length; i += 4) {
      const v = d[i]; // grayscale: subject = white
      d[i] = 61; d[i + 1] = 220; d[i + 2] = 132;
      d[i + 3] = Math.round(v * 0.5);
    }
    ctx.putImageData(px, 0, 0);
    fmask.hidden = false;
  };
  im.src = `data:image/png;base64,${b64}`;
}

// After each render (whatever started it), show what the server treated as
// the subject. Every render rewrites the preview's srcdoc.
new MutationObserver(() => {
  if (state.tab !== "image") return;
  const f = state.result && state.result.focus;
  drawMask(f && f.mask_png_b64);
}).observe($("preview"), { attributes: true, attributeFilter: ["srcdoc"] });
$("show_mask").addEventListener("change", () => {
  const f = state.result && state.result.focus;
  drawMask(f && f.mask_png_b64);
});

// ---------- background picture ----------

$("bg_file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  state.bgFile = f || null;
  $("bg-file-name").textContent = f ? f.name : "none chosen";
  if (f) render();
});

// ---------- models ----------

function needsSubject() {
  return $("background").value !== "keep" || $("zoom_subject").checked || $("enhance_subject").checked;
}

function describeModels() {
  const el = $("subject-status");
  if (!models) { el.textContent = ""; return; }
  if (!models.engine) {
    el.textContent = 'This server lacks the subject extra: pip install "ascii-magic-tools[subject]". The focus box still works.';
    return;
  }
  const m = models.models[$("subject_model").value];
  if (m.installed) el.textContent = "Ready. Runs on this server; nothing is uploaded anywhere.";
  else if (models.can_install) el.textContent = `Downloads once (${m.mb} MB) the first time it's used.`;
  else el.textContent = `Not installed, and downloads are off here. In a terminal: ascii-magic subject install ${$("subject_model").value}`;
}

async function loadModels() {
  try {
    models = await (await fetch("/api/subject/models")).json();
  } catch (_) {
    models = null;
  }
  describeModels();
}

// Download the model up front (with a clear message) rather than leaving a
// first render to sit on "Rendering…" while it downloads.
async function ensureModel() {
  if (!needsSubject() || !models || !models.engine) return;
  const name = $("subject_model").value;
  const m = models.models[name];
  if (m.installed || !models.can_install) return;
  setStatus(`Downloading the ${name} subject model (${m.mb} MB, once)…`, "busy");
  try {
    const res = await fetch("/api/subject/install", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: name }),
    });
    const body = await readJson(res);
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
    m.installed = true;
    describeModels();
    render();
  } catch (err) {
    setStatus(`Model download failed: ${err.message}`, "error");
  }
}

function syncFocusFields() {
  const bg = $("background").value;
  $("bg-color-field").hidden = bg !== "color";
  $("bg-image-field").hidden = bg !== "image";
  describeModels();
}
for (const id of ["background", "zoom_subject", "enhance_subject", "subject_model"]) {
  $(id).addEventListener("change", () => { syncFocusFields(); ensureModel(); });
}
syncFocusFields();
loadModels();
