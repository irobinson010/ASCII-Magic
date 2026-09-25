"use strict";

// Compose tab: a stack of image/text layers rendered by /api/compose.
// Relies on app.js globals: $, state, setStatus, readJson, autoRender,
// injectMeasure.

const ANCHORS = [
  "top-left", "top", "top-right",
  "left", "center", "right",
  "bottom-left", "bottom", "bottom-right",
];
const THEMES = ["green", "amber", "cyan", "crimson", "violet", "white"];
const PALETTES = ["rainbow", "sunset", "ocean", "fire", "forest", "neon", "matrix", "mono"];
const DIRECTIONS = ["horizontal", "vertical", "diagonal", "diagonal-up", "radial"];
const BLENDS = ["tint", "multiply", "screen", "overlay"];

const cmp = {
  layers: [],      // {type, file?, fileName?, ...layer fields}
  selected: -1,
  placed: [],      // last server layout: {index, x, y, w, h}
  canvas: null,    // last server canvas {cols, rows}
  rendering: false,
  queued: false,
};

// ---------- layer model ----------

function newLayer(type, extra) {
  const base = { type, at: "center", color: null };
  if (type === "text") Object.assign(base, { text: "Hello", style: "block", scale: 0.5, align: "left" });
  // No color by default, like the CLI: "From image" paints dark ink in its
  // own dark colors, which vanish on a black page for dark-on-light art.
  else Object.assign(base, { mode: "braille" });
  return Object.assign(base, extra || {});
}

function layerLabel(layer, i) {
  if (layer.type === "text") return `T  ${layer.text.split("\n")[0].slice(0, 24) || "(empty)"}`;
  return `I  ${layer.fileName || "image " + (i + 1)}`;
}

// Scene JSON for the server: uploads are referenced by position.
function sceneForServer() {
  const files = [];
  const layers = cmp.layers.map((layer) => {
    const out = {};
    for (const [k, v] of Object.entries(layer)) {
      if (k === "file" || k === "fileName") continue;
      if (v === null || v === undefined || v === "") continue;
      out[k] = v;
    }
    if (layer.type === "image") {
      let idx = files.indexOf(layer.file);
      if (idx < 0) { files.push(layer.file); idx = files.length - 1; }
      out.upload = idx;
    }
    return out;
  });
  const canvas = {};
  if (num("cmp_cols")) canvas.cols = num("cmp_cols");
  if (num("cmp_rows")) canvas.rows = num("cmp_rows");
  if ($("cmp_bg_mode").value === "custom") canvas.background = $("cmp_bg").value;
  if ($("cmp_overlay").value) {
    canvas.overlay = $("cmp_overlay").value;
    canvas.overlay_direction = $("cmp_overlay_direction").value;
  }
  return { scene: { canvas, layers }, files };
}

// ---------- rendering ----------

function composeCanRender() {
  return cmp.layers.length > 0 && cmp.layers.every((l) => l.type !== "image" || l.file);
}

async function composeRender() {
  if (!composeCanRender()) {
    setStatus(cmp.layers.length ? "Pick a file for every image layer." : "Add an image or text layer to begin.", "");
    return;
  }
  if (cmp.rendering) { cmp.queued = true; return; }
  cmp.rendering = true;
  setStatus("Composing…", "busy");
  const { scene, files } = sceneForServer();
  const form = new FormData();
  for (const f of files) form.append("images", f);
  form.append("scene", JSON.stringify(scene));
  try {
    const res = await fetch("/api/compose", { method: "POST", body: form });
    const body = await readJson(res);
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
    state.result = body;
    state.fileStem = "composition";
    cmp.placed = body.layers;
    cmp.canvas = body.canvas;
    $("cmp-overlay").hidden = true; // re-shown when the preview reports its box
    $("preview").srcdoc = injectMeasure(body.html);
    for (const b of ["dl-ans", "dl-html", "dl-txt"]) $(b).disabled = false;
    for (const b of ["dl-gif", "dl-frames", "dl-mp4"]) $(b).disabled = true;
    $("cmp_dl_scene").disabled = false;
    setStatus(`Composed ${body.canvas.cols} × ${body.canvas.rows}, ${body.layers.length} layers in ${body.elapsed_ms} ms`, "");
  } catch (err) {
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    cmp.rendering = false;
    if (cmp.queued) { cmp.queued = false; composeRender(); }
  }
}

// ---------- layer list ----------

function refreshList() {
  const ul = $("cmp_layers");
  ul.replaceChildren();
  cmp.layers.forEach((layer, i) => {
    const li = document.createElement("li");
    li.classList.toggle("selected", i === cmp.selected);
    const name = document.createElement("span");
    name.className = "layer-name";
    name.textContent = layerLabel(layer, i); // textContent: names are user data
    li.append(name);
    const btn = (label, title, fn, disabled) => {
      const b = document.createElement("button");
      b.textContent = label;
      b.title = title;
      b.setAttribute("aria-label", title);
      b.disabled = !!disabled;
      b.addEventListener("click", (e) => { e.stopPropagation(); fn(); });
      li.append(b);
    };
    btn("↓", "Move down (paint earlier)", () => moveLayer(i, -1), i === 0);
    btn("↑", "Move up (paint later)", () => moveLayer(i, 1), i === cmp.layers.length - 1);
    btn("×", "Delete layer", () => deleteLayer(i));
    li.addEventListener("click", () => selectLayer(i));
    ul.append(li);
  });
}

function selectLayer(i) {
  cmp.selected = i;
  refreshList();
  buildEditor();
  drawOverlay();
}

function moveLayer(i, delta) {
  const j = i + delta;
  if (j < 0 || j >= cmp.layers.length) return;
  [cmp.layers[i], cmp.layers[j]] = [cmp.layers[j], cmp.layers[i]];
  if (cmp.selected === i) cmp.selected = j;
  else if (cmp.selected === j) cmp.selected = i;
  refreshList();
  autoRender();
}

function deleteLayer(i) {
  cmp.layers.splice(i, 1);
  if (cmp.selected >= cmp.layers.length) cmp.selected = cmp.layers.length - 1;
  refreshList();
  buildEditor();
  if (cmp.layers.length) autoRender();
  else { $("cmp-overlay").hidden = true; setStatus("Add an image or text layer to begin.", ""); }
}

function addLayer(layer) {
  cmp.layers.push(layer);
  selectLayer(cmp.layers.length - 1);
  composeRender();
}

$("cmp_add_text").addEventListener("click", (e) => {
  e.preventDefault();
  addLayer(newLayer("text", cmp.layers.length ? { at: "top" } : {}));
});
$("cmp_add_image").addEventListener("click", (e) => {
  e.preventDefault();
  $("cmp_file").value = "";
  $("cmp_file").click();
});
$("cmp_file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  if (!f.type.startsWith("image/")) { setStatus("That doesn't look like an image file.", "error"); return; }
  addLayer(newLayer("image", { file: f, fileName: f.name, cols: 80 }));
});

// ---------- editor ----------

// [key, label, kind, extra]; kind: text | area | int | float | range | select | check | color
const COMMON_FIELDS = [
  ["at", "Anchor", "select", { options: ANCHORS }],
  ["x", "X (col)", "int", { placeholder: "anchor" }],
  ["y", "Y (row)", "int", { placeholder: "anchor" }],
  ["dx", "Nudge →", "int", { placeholder: "0" }],
  ["dy", "Nudge ↓", "int", { placeholder: "0" }],
  ["cols", "Width", "int", { placeholder: "auto", min: 1, max: 500 }],
  ["rows", "Height", "int", { placeholder: "auto", min: 1, max: 500 }],
  ["color", "Color", "color"],
  ["outline", "Outline", "int", { placeholder: "auto", min: 0, max: 10,
    title: "Clear this many cells around the ink so it reads over busy art (auto: 1 for text, 0 for images)" }],
  ["opaque", "Opaque box", "check"],
  ["overlay", "Overlay", "overlay"],
  ["overlay_direction", "Overlay direction", "select", { options: DIRECTIONS }],
  ["overlay_mode", "Overlay blend", "select", { options: BLENDS }],
  ["overlay_strength", "Overlay strength", "range", { min: 0, max: 1, step: 0.05, fallback: 1 }],
];
const TEXT_FIELDS = [
  ["text", "Text", "area"],
  ["_translate", "", "translate"],
  ["style", "Style", "select", { options: ["block", "small", "shadow", "box", "banner", "figlet"] }],
  ["scale", "Auto size (of canvas width)", "range", { min: 0.05, max: 1, step: 0.05 }],
  ["align", "Align lines", "select", { options: ["left", "center", "right"] }],
];
const IMAGE_FIELDS = [
  ["mode", "Mode", "select", { options: ["braille", "glyph"] }],
  ["threshold", "Threshold", "range", { min: 0, max: 1, step: 0.05, fallback: 0.5 }],
  ["gamma", "Gamma", "range", { min: 0.2, max: 3, step: 0.05, fallback: 1 }],
  ["dither", "Dither (braille)", "check"],
  ["invert", "Invert", "check"],
  ["autocontrast", "Autocontrast", "check"],
  ["rotate", "Rotate", "select", { options: ["0", "90", "180", "270"], number: true }],
];

function field(layer, [key, label, kind, extra]) {
  extra = extra || {};
  const id = `cmp_f_${key}`;
  const wrap = document.createElement(kind === "check" ? "label" : "div");
  const set = (v) => {
    if (v === "" || v === null || (typeof v === "number" && Number.isNaN(v))) delete layer[key];
    else layer[key] = v;
    refreshList();
    autoRender();
  };

  if (kind === "check") {
    wrap.className = "check";
    const input = document.createElement("input");
    input.type = "checkbox";
    input.id = id;
    input.checked = !!layer[key];
    input.addEventListener("change", () => set(input.checked || null));
    wrap.append(input, " " + label);
    return wrap;
  }

  wrap.className = "field";
  const lab = document.createElement("label");
  lab.htmlFor = id;
  lab.textContent = label;
  if (extra.title) lab.title = extra.title;
  wrap.append(lab);

  let input;
  if (kind === "select") {
    input = document.createElement("select");
    for (const o of extra.options) {
      const opt = document.createElement("option");
      opt.value = opt.textContent = o;
      input.append(opt);
    }
    input.value = String(layer[key] ?? extra.options[0]);
    input.addEventListener("change", () => set(extra.number ? Number(input.value) : input.value));
  } else if (kind === "area") {
    input = document.createElement("textarea");
    input.rows = 2;
    input.value = layer[key] || "";
    input.addEventListener("input", () => set(input.value));
  } else if (kind === "range") {
    input = document.createElement("input");
    input.type = "range";
    Object.assign(input, { min: extra.min, max: extra.max, step: extra.step });
    input.value = layer[key] ?? extra.fallback ?? extra.min;
    const out = document.createElement("output");
    out.textContent = input.value;
    lab.append(" ", out);
    input.addEventListener("input", () => { out.textContent = input.value; set(Number(input.value)); });
  } else if (kind === "color") {
    return colorField(layer, wrap, lab, id, set);
  } else if (kind === "overlay") {
    return overlayField(layer, wrap, id, set);
  } else if (kind === "translate") {
    // translate.js fills the layer's Text box; the edited text is what renders.
    wrap.replaceChildren();
    if (window.translateControl) wrap.append(window.translateControl("cmp_f_text"));
    return wrap;
  } else {
    input = document.createElement("input");
    input.type = "number";
    if (extra.min !== undefined) input.min = extra.min;
    if (extra.max !== undefined) input.max = extra.max;
    input.placeholder = extra.placeholder || "";
    input.value = layer[key] ?? "";
    input.addEventListener("change", () => set(input.value === "" ? null : Math.round(Number(input.value))));
  }
  input.id = id;
  wrap.append(input);
  return wrap;
}

function colorField(layer, wrap, lab, id, set) {
  const row = document.createElement("span");
  row.className = "seed-row";
  const sel = document.createElement("select");
  sel.id = id;
  const opts = [["", "Default (no color)"], ["image", layer.type === "text" ? "Image beneath" : "From image"],
    ...THEMES.map((t) => [t, t]), ["custom", "Custom"]];
  for (const [v, t] of opts) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = t;
    sel.append(o);
  }
  const pick = document.createElement("input");
  pick.type = "color";
  pick.setAttribute("aria-label", "Custom color");
  const c = layer.color;
  if (c && c.startsWith("#")) { sel.value = "custom"; pick.value = c; }
  else { sel.value = c || ""; pick.value = "#ffcc00"; }
  pick.hidden = sel.value !== "custom";
  sel.addEventListener("change", () => {
    pick.hidden = sel.value !== "custom";
    set(sel.value === "custom" ? pick.value : sel.value || null);
  });
  pick.addEventListener("input", () => set(pick.value));
  row.append(sel, pick);
  wrap.append(row);
  return wrap;
}

// Palette list plus "Custom": comma-separated colors typed by the user.
function overlayField(layer, wrap, id, set) {
  const row = document.createElement("span");
  row.className = "seed-row";
  const sel = document.createElement("select");
  sel.id = id;
  for (const [v, t] of [["", "None"], ...PALETTES.map((p) => [p, p]), ["custom", "Custom"]]) {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = t;
    sel.append(o);
  }
  const custom = document.createElement("input");
  custom.type = "text";
  custom.placeholder = "#ff0000,#0000ff";
  custom.setAttribute("aria-label", "Custom overlay colors");
  const cur = layer.overlay || "";
  if (!cur || PALETTES.includes(cur)) sel.value = cur;
  else { sel.value = "custom"; custom.value = cur; }
  custom.hidden = sel.value !== "custom";
  sel.addEventListener("change", () => {
    custom.hidden = sel.value !== "custom";
    set(sel.value === "custom" ? custom.value.trim() || null : sel.value || null);
  });
  custom.addEventListener("change", () => set(custom.value.trim() || null));
  row.append(sel, custom);
  wrap.append(row);
  return wrap;
}

function buildEditor() {
  const ed = $("cmp_editor");
  ed.replaceChildren();
  const layer = cmp.layers[cmp.selected];
  if (!layer) return;
  const h = document.createElement("h3");
  h.textContent = layer.type === "text" ? "Text layer" : `Image layer — ${layer.fileName}`;
  ed.append(h);
  const specific = layer.type === "text" ? TEXT_FIELDS : IMAGE_FIELDS;
  // Pair small numeric fields two to a row.
  let row = null;
  for (const spec of [...specific, ...COMMON_FIELDS]) {
    const el = field(layer, spec);
    if (spec[2] === "int") {
      if (!row) { row = document.createElement("div"); row.className = "field-row"; ed.append(row); }
      row.append(el);
      if (row.children.length === 2) row = null;
    } else {
      row = null;
      ed.append(el);
    }
  }
}

// ---------- canvas controls & scene download ----------

for (const id of ["cmp_cols", "cmp_rows", "cmp_bg_mode", "cmp_bg", "cmp_overlay", "cmp_overlay_direction"]) {
  $(id).addEventListener("change", () => { $("cmp_bg").hidden = $("cmp_bg_mode").value !== "custom"; autoRender(); });
}
$("cmp_bg").hidden = true;

$("cmp_dl_scene").addEventListener("click", () => {
  if (!state.result || !state.result.scene) return;
  download("scene.json", JSON.stringify(state.result.scene, null, 2) + "\n", "application/json");
});

// ---------- preview overlay: move and resize layers ----------

function cellPx() {
  const m = state.measure;
  return { w: m.w / cmp.canvas.cols, h: m.h / cmp.canvas.rows };
}

// Called from app.js whenever the preview reports its content box.
function composeShowOverlay(d) {
  if (!cmp.canvas || !d || d.w < 4) return;
  state.measure = d;
  drawOverlay();
}

function drawOverlay() {
  const ov = $("cmp-overlay");
  ov.replaceChildren();
  if (state.tab !== "compose" || !cmp.canvas || !state.measure) { ov.hidden = true; return; }
  const m = state.measure;
  const c = cellPx();
  for (const p of cmp.placed) {
    const box = document.createElement("div");
    box.className = "cmp-box" + (p.index === cmp.selected ? " selected" : "");
    // Clip the box to the canvas: layers may hang off its edges.
    const x0 = Math.max(0, p.x), y0 = Math.max(0, p.y);
    const x1 = Math.min(cmp.canvas.cols, p.x + p.w), y1 = Math.min(cmp.canvas.rows, p.y + p.h);
    if (x1 <= x0 || y1 <= y0) continue;
    Object.assign(box.style, {
      left: `${m.x + x0 * c.w}px`, top: `${m.y + y0 * c.h}px`,
      width: `${(x1 - x0) * c.w}px`, height: `${(y1 - y0) * c.h}px`,
    });
    box.title = `${layerLabel(cmp.layers[p.index] || {}, p.index)} — drag to move, corner to resize`;
    const label = document.createElement("div");
    label.className = "cmp-label";
    label.textContent = `${p.w} × ${p.h} @ ${p.x},${p.y}`;
    const handle = document.createElement("div");
    handle.className = "handle";
    box.append(label, handle);
    box.addEventListener("pointerdown", startMove(p, box));
    handle.addEventListener("pointerdown", startResize(p, box));
    ov.append(box);
  }
  ov.hidden = false;
}

function dragSession(e, el, onMove, onUp) {
  e.preventDefault();
  e.stopPropagation();
  el.setPointerCapture(e.pointerId);
  $("preview-wrap").classList.add("dragging");
  const start = { x: e.clientX, y: e.clientY };
  const move = (ev) => onMove(ev.clientX - start.x, ev.clientY - start.y, ev);
  const up = (ev) => {
    el.removeEventListener("pointermove", move);
    el.removeEventListener("pointerup", up);
    el.removeEventListener("pointercancel", up);
    try { el.releasePointerCapture(ev.pointerId); } catch (_) {}
    $("preview-wrap").classList.remove("dragging");
    onUp(ev.clientX - start.x, ev.clientY - start.y, ev);
  };
  el.addEventListener("pointermove", move);
  el.addEventListener("pointerup", up);
  el.addEventListener("pointercancel", up);
}

function startMove(p, box) {
  return (e) => {
    if (cmp.selected !== p.index) { selectLayer(p.index); return; } // first click selects
    const left = parseFloat(box.style.left), top = parseFloat(box.style.top);
    const c = cellPx();
    dragSession(e, box,
      (dx, dy) => {
        box.style.left = `${left + dx}px`;
        box.style.top = `${top + dy}px`;
        box.firstChild.textContent = `@ ${p.x + Math.round(dx / c.w)},${p.y + Math.round(dy / c.h)}`;
      },
      (dx, dy) => {
        const mc = Math.round(dx / c.w), mr = Math.round(dy / c.h);
        if (!mc && !mr) { drawOverlay(); return; }
        placeLayer(p, p.x + mc, p.y + mr);
      });
  };
}

function startResize(p, box) {
  return (e) => {
    const w0 = parseFloat(box.style.width), h0 = parseFloat(box.style.height);
    const c = cellPx();
    dragSession(e, e.currentTarget,
      (dx, dy, ev) => {
        let w = Math.max(c.w * 2, w0 + dx);
        let h = Math.max(c.h, h0 + dy);
        if (ev.shiftKey) h = w * (h0 / w0); // aspect lock
        box.style.width = `${w}px`;
        box.style.height = `${h}px`;
        box.firstChild.textContent = `${Math.round(w / c.w)} × ${Math.round(h / c.h)}`;
      },
      (dx, dy, ev) => {
        const layer = cmp.layers[p.index];
        const cols = Math.max(2, Math.round((w0 + dx) / c.w));
        layer.cols = Math.min(500, cols);
        if (ev.shiftKey) delete layer.rows; // aspect-locked: height follows
        else layer.rows = Math.min(500, Math.max(1, Math.round((h0 + dy) / c.h)));
        buildEditor();
        composeRender();
      });
  };
}

// Exact placement: the dragged position becomes x/y (the anchor no longer applies).
function placeLayer(p, x, y) {
  const layer = cmp.layers[p.index];
  layer.x = x - (layer.dx || 0);
  layer.y = y - (layer.dy || 0);
  buildEditor();
  composeRender();
}

document.addEventListener("keydown", (e) => {
  if (state.tab !== "compose" || cmp.selected < 0 || !cmp.placed.length) return;
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
  const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
  if (!step) return;
  const p = cmp.placed.find((q) => q.index === cmp.selected);
  if (!p) return;
  e.preventDefault();
  const n = e.shiftKey ? 5 : 1;
  p.x += step[0] * n;
  p.y += step[1] * n;
  placeLayer(p, p.x, p.y);
});

refreshList();
