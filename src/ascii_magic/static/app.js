"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  file: null,        // uploaded image File
  videoFile: null,   // uploaded video File
  fileStem: "ascii-art",
  imgW: 0,           // natural dimensions of the uploaded image
  imgH: 0,
  tab: "image",      // image | text | video
  result: null,      // last /api/render response
  rendering: false,
  queued: false,
};

const PROFILES = {
  terminal: { fallback_cols: 100, html_font_size: 12, html_fill_spaces: false },
  web: { fallback_cols: 160, html_font_size: 10, html_fill_spaces: true },
};

// ---------- option collection ----------

function num(id) {
  const v = $(id).value.trim();
  if (v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function collectOptions() {
  const source = state.tab;
  return {
    source,
    // video source
    video_fps: num("video_fps"),
    video_max_frames: num("video_max_frames"),
    video_mode: $("video_mode").value,
    // text source
    text: $("text").value,
    text_style: $("text_style").value,
    text_width: num("text_width"),
    text_font_size: num("text_font_size"),
    banner_char: $("banner_char").value || "#",
    // image source (width shared with video via its own input)
    mode: $("mode").value,
    cols: source === "video" ? num("video_cols") : num("cols"),
    rotate: num("rotate"),
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
    // animation (rain animation is an image-mode feature; video is already animated)
    animate: state.tab !== "video" && $("matrix").checked && $("animate").checked,
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
  if (state.tab === "text") return $("text").value.trim() !== "";
  if (state.tab === "video") return state.videoFile !== null;
  return state.file !== null;
}

async function render() {
  if (!canRender()) return;
  if (state.rendering) { state.queued = true; return; }
  state.rendering = true;
  setStatus(state.tab === "video" ? "Rendering video… this takes a few seconds" : "Rendering…", "busy");

  const form = new FormData();
  if (state.tab === "video") {
    form.append("image", state.videoFile);
  } else if (state.file) {
    form.append("image", state.file);
  }
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
    $("dl-frames").disabled = !body.frames_text;

    let msg;
    if (body.video) {
      msg = `Rendered ${body.video.frames} video frames @ ${body.video.fps} fps in ${body.elapsed_ms} ms`;
    } else {
      msg = `Rendered ${body.ascii.split("\n").length} lines in ${body.elapsed_ms} ms`;
    }
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
  if (state.tab === "video") return; // video renders are seconds, not ms — explicit only
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
$("dl-frames").addEventListener("click", () =>
  download(`${state.fileStem}.frames`, state.result.frames_text, "text/plain"));

// ---------- sizing ----------

function autoSize() {
  // Fit the art to the preview pane at the current font size, preserving the
  // image's aspect ratio. Sets the Width knob; the user tunes from there.
  if (!state.imgW || !state.imgH) return;
  const pane = $("preview").getBoundingClientRect();
  const fontPx = num("html_font_size") || 12;
  const charW = fontPx * 0.62; // monospace advance ~= 0.62em
  const pad = 40;              // preview body padding + scrollbar allowance
  const maxCols = Math.floor((pane.width - pad) / charW);
  const maxRows = Math.floor((pane.height - pad) / fontPx);
  const rot = num("rotate") || 0;
  const [w, h] = rot % 180 ? [state.imgH, state.imgW] : [state.imgW, state.imgH];
  const aspect = h / w;
  const rowFactor = 0.5;       // both glyph (8x16) and braille (2x4) cells are 1:2

  let cols = maxCols;
  if (Math.ceil(cols * aspect * rowFactor) > maxRows) {
    cols = Math.floor(maxRows / (aspect * rowFactor));
  }
  $("cols").value = Math.max(20, Math.min(cols, 400));
}

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

  const probe = new Image();
  probe.onload = () => {
    state.imgW = probe.naturalWidth;
    state.imgH = probe.naturalHeight;
    autoSize();
    render();
  };
  probe.onerror = () => render();
  probe.src = thumb.src;
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

const TABS = ["image", "text", "video"];

function setTab(name) {
  state.tab = name;
  for (const t of TABS) {
    $(`panel-${t}`).hidden = t !== name;
    $(`tab-${t}`).classList.toggle("active", t === name);
  }
  syncVisibility();
  if (name !== "video") autoRender();
}
for (const t of TABS) {
  $(`tab-${t}`).addEventListener("click", () => setTab(t));
}

$("video_file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  if (!f) return;
  state.videoFile = f;
  state.fileStem = f.name.replace(/\.[^.]+$/, "") || "ascii-video";
  render(); // one render on selection; knob changes need the Render button
});

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

  // Colorize doesn't apply to video renders (frames colorize themselves);
  // matrix and captions DO — but rain animation doesn't (video is already
  // animated).
  const isVideo = state.tab === "video";
  for (const id of ["sec-colorize", "sec-html"]) {
    $(id).hidden = isVideo;
  }
  $("animate-row").hidden = isVideo;
  if (isVideo) $("anim-knobs").hidden = true;
}

// ---------- profiles ----------

$("profile").addEventListener("change", () => {
  const p = PROFILES[$("profile").value];
  $("html_font_size").value = p.html_font_size;
  $("html_fill_spaces").checked = p.html_fill_spaces;
  if (state.imgW) {
    autoSize(); // font size changed, so the fit changes too
  } else {
    $("cols").value = p.fallback_cols;
  }
  autoRender();
});

// ---------- misc wiring ----------

$("render").addEventListener("click", render);

$("autosize").addEventListener("click", (e) => {
  e.preventDefault();
  autoSize();
  render();
});

$("rotate").addEventListener("change", () => {
  autoSize(); // 90/270 swap the aspect ratio, so re-fit
});

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
