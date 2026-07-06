"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  file: null,        // uploaded File
  fileStem: "ascii-art",
  result: null,      // last /api/render response
  rendering: false,
  queued: false,
};

const PROFILES = {
  terminal: { cols: 100, html_font_size: 12, html_fill_spaces: false },
  web: { cols: 160, html_font_size: 10, html_fill_spaces: true },
};

// ---------- option collection ----------

function num(id) {
  const v = $(id).value.trim();
  if (v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function collectOptions() {
  const source = $("panel-text").hidden ? "image" : "text";
  return {
    source,
    // text source
    text: $("text").value,
    text_style: $("text_style").value,
    text_width: num("text_width"),
    text_font_size: num("text_font_size"),
    banner_char: $("banner_char").value || "#",
    // image source
    mode: $("mode").value,
    cols: num("cols"),
    quality: $("quality").value,
    ascii_preset: $("ascii_preset").value,
    unicode_mode: $("unicode_mode").value,
    cell_w: num("cell_w"),
    cell_h: num("cell_h"),
    topk: num("topk"),
    threshold: num("threshold"),
    dither: $("dither").checked,
    gamma: num("gamma"),
    autocontrast: $("autocontrast").checked,
    invert: $("invert").checked,
    // colorize
    colorize: $("colorize").checked,
    keep_top: num("keep_top"),
    color_top: $("color_top").checked,
    max_rows: num("max_rows"),
    max_cols: num("max_cols"),
    out_rows: num("out_rows"),
    out_cols: num("out_cols"),
    html_font_size: num("html_font_size"),
    html_line_height: num("html_line_height"),
    html_fill_spaces: $("html_fill_spaces").checked,
    // matrix
    matrix: $("matrix").checked,
    matrix_top: $("matrix_top").checked,
    matrix_seed: num("matrix_seed"),
    matrix_gamma: num("matrix_gamma"),
    matrix_color: $("matrix_theme").value === "custom"
      ? $("matrix_custom_color").value
      : $("matrix_theme").value,
    matrix_fg_min: num("matrix_fg_min"),
    matrix_fg_max: num("matrix_fg_max"),
    matrix_bg_min: num("matrix_bg_min"),
    matrix_bg_max: num("matrix_bg_max"),
    matrix_chars: $("matrix_chars").value,
    matrix_fill_spaces: $("matrix_fill_spaces").checked,
    matrix_mask: $("matrix_mask").checked,
    matrix_mask_boost: num("matrix_mask_boost"),
    matrix_mask_density_floor: num("matrix_mask_density_floor"),
    matrix_bg_dim: num("matrix_bg_dim"),
    matrix_bg_density: num("matrix_bg_density"),
    // animation
    animate: $("matrix").checked && $("animate").checked,
    anim_frames: num("anim_frames"),
    anim_fps: num("anim_fps"),
    anim_tail: num("anim_tail"),
    anim_reveal: $("anim_reveal").checked,
    // caption
    caption_text: $("caption_text").value.trim() || null,
    caption_pos: $("caption_pos").value,
    caption_style: $("caption_style").value,
    caption_scale: num("caption_scale"),
    caption_align: $("caption_align").value,
    caption_color: $("caption_color_mode").value === "custom"
      ? $("caption_custom_color").value
      : ($("caption_color_mode").value || null),
  };
}

// ---------- rendering ----------

function setStatus(msg, cls) {
  const el = $("status");
  el.textContent = msg;
  el.className = cls || "";
}

function canRender() {
  const textMode = !$("panel-text").hidden;
  return textMode ? $("text").value.trim() !== "" : state.file !== null;
}

async function render() {
  if (!canRender()) return;
  if (state.rendering) { state.queued = true; return; }
  state.rendering = true;
  setStatus("Rendering…", "busy");

  const form = new FormData();
  if (state.file) form.append("image", state.file);
  form.append("options", JSON.stringify(collectOptions()));

  try {
    const res = await fetch("/api/render", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);

    state.result = body;
    $("preview").srcdoc = body.html;
    if (body.seed !== null && body.seed !== undefined) {
      $("matrix_seed").value = body.seed;
    }
    for (const b of ["dl-ans", "dl-html", "dl-txt"]) $(b).disabled = false;
    $("dl-gif").disabled = !body.gif_b64;

    const lines = body.ascii.split("\n").length;
    let msg = `Rendered ${lines} lines in ${body.elapsed_ms} ms`;
    if (body.warning) msg += ` — ${body.warning}`;
    setStatus(msg, body.warning ? "error" : "");
  } catch (err) {
    setStatus(`Error: ${err.message}`, "error");
  } finally {
    state.rendering = false;
    if (state.queued) { state.queued = false; render(); }
  }
}

let debounceTimer = null;
function autoRender() {
  if (!$("auto").checked) return;
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(render, 350);
}

// ---------- downloads ----------

function download(name, content, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([content], { type }));
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}

$("dl-ans").addEventListener("click", () =>
  download(`${state.fileStem}.ans`, state.result.ansi, "text/plain"));
$("dl-html").addEventListener("click", () =>
  download(`${state.fileStem}.html`, state.result.html, "text/html"));
$("dl-txt").addEventListener("click", () =>
  download(`${state.fileStem}.txt`, state.result.ascii, "text/plain"));
$("dl-gif").addEventListener("click", () => {
  const bin = atob(state.result.gif_b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  download(`${state.fileStem}.gif`, bytes, "image/gif");
});

// ---------- file handling ----------

function loadFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setStatus("That doesn't look like an image file.", "error");
    return;
  }
  state.file = file;
  state.fileStem = file.name.replace(/\.[^.]+$/, "") || "ascii-art";
  const thumb = $("thumb");
  thumb.src = URL.createObjectURL(file);
  thumb.hidden = false;
  $("drop-hint").innerHTML = `${file.name}<br>(click to change)`;
  render();
}

const dz = $("dropzone");
dz.addEventListener("click", () => $("file").click());
dz.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") $("file").click(); });
$("file").addEventListener("change", (e) => loadFile(e.target.files[0]));
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("drag"); });
dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
dz.addEventListener("drop", (e) => {
  e.preventDefault();
  dz.classList.remove("drag");
  loadFile(e.dataTransfer.files[0]);
});

// ---------- tabs & visibility ----------

function setTab(image) {
  $("panel-image").hidden = !image;
  $("panel-text").hidden = image;
  $("tab-image").classList.toggle("active", image);
  $("tab-text").classList.toggle("active", !image);
  autoRender();
}
$("tab-image").addEventListener("click", () => setTab(true));
$("tab-text").addEventListener("click", () => setTab(false));

function syncVisibility() {
  const braille = $("mode").value === "braille";
  $("braille-knobs").hidden = !braille;
  $("glyph-knobs").hidden = braille;

  const style = $("text_style").value;
  $("text-font-field").hidden = style === "box" || style === "banner";
  $("banner-char-field").hidden = style !== "banner";

  $("color-knobs").hidden = !$("colorize").checked;
  $("matrix-knobs").hidden = !$("matrix").checked;
  $("mask-knobs").hidden = !$("matrix_mask").checked;
  $("anim-knobs").hidden = !$("animate").checked;
  $("custom-color-field").hidden = $("matrix_theme").value !== "custom";
  $("caption-color-field").hidden = $("caption_color_mode").value !== "custom";
}

// ---------- profiles ----------

$("profile").addEventListener("change", () => {
  const p = PROFILES[$("profile").value];
  $("cols").value = p.cols;
  $("html_font_size").value = p.html_font_size;
  $("html_fill_spaces").checked = p.html_fill_spaces;
  autoRender();
});

// ---------- misc wiring ----------

$("render").addEventListener("click", render);

$("reroll").addEventListener("click", (e) => {
  e.preventDefault();
  $("matrix_seed").value = "";
  render();
});

for (const id of ["threshold", "gamma", "matrix_gamma", "caption_scale"]) {
  $(id).addEventListener("input", () => { $(`${id}-out`).value = $(id).value; });
}

document.querySelectorAll("#controls input, #controls select, #controls textarea").forEach((el) => {
  el.addEventListener("change", () => { syncVisibility(); autoRender(); });
  if (el.tagName === "TEXTAREA" || el.type === "text" || el.type === "range") {
    el.addEventListener("input", autoRender);
  }
});

syncVisibility();
