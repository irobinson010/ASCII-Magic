"use strict";

// Interface polish layered over app.js / compose.js / translate.js: the
// download menu, empty state, drop-anywhere, the profile switch, card
// summaries, and keyboard shortcuts. Relies on app.js globals: $, state,
// setTab, render, setStatus.

// ---------- animation effect pickers (one list, two selects) ----------

const EFFECT_GROUPS = [
  ["Motion", [["wave", "Wave (flag)"], ["ripple", "Ripple"], ["twist", "Twist (ribbon)"],
    ["swing", "Swing"], ["bounce", "Bounce"], ["shake", "Shake"], ["glitch", "Glitch"],
    ["scroll", "Scroll (marquee)"]]],
  ["Turn and flip", [["spin", "Spin (3D)"], ["rotate", "Rotate (wheel)"], ["flip-left", "Flip left"],
    ["flip-right", "Flip right"], ["flip-up", "Flip up"], ["flip-down", "Flip down"],
    ["flip-diagonal", "Flip diagonal"], ["flip-antidiagonal", "Flip other diagonal"]]],
  ["Size", [["stretch", "Stretch sideways"], ["stretch-v", "Stretch vertically"],
    ["squash", "Squash (jelly)"], ["zoom", "Zoom"], ["pulse", "Pulse (heartbeat)"]]],
  ["Reveal", [["typewriter", "Typewriter"], ["fade", "Fade"]]],
];

document.querySelectorAll("select[data-effects]").forEach((sel) => {
  sel.append(new Option(sel.id === "text_anim" ? "None (still)" : "Nothing", ""));
  for (const [label, items] of EFFECT_GROUPS) {
    const g = document.createElement("optgroup");
    g.label = label;
    for (const [value, text] of items) g.append(new Option(text, value));
    sel.append(g);
  }
});

// ---------- "For: Terminal / Web page" ----------

const segButtons = document.querySelectorAll(".segmented [data-profile]");
function syncSegmented() {
  for (const b of segButtons) b.setAttribute("aria-checked", String(b.dataset.profile === $("profile").value));
}
for (const b of segButtons) {
  b.addEventListener("click", () => {
    if ($("profile").value === b.dataset.profile) return;
    $("profile").value = b.dataset.profile;
    $("profile").dispatchEvent(new Event("change"));
    syncSegmented();
  });
}
syncSegmented();

// ---------- download menu ----------

const dlToggle = $("dl-toggle");
const dlMenu = $("dl-menu");
const DL_ITEMS = ["dl-ans", "dl-html", "dl-txt", "dl-gif", "dl-frames", "dl-mp4"];

function setDisabled(el, value) {
  // Write only on change: even a same-value write queues a mutation record,
  // which would re-trigger the observer below forever.
  if (el.disabled !== value) el.disabled = value;
}
function refreshDownloads() {
  const any = DL_ITEMS.some((id) => !$(id).disabled);
  setDisabled(dlToggle, !any);
  setDisabled($("copy-txt"), $("dl-txt").disabled);
  if (!any && !dlMenu.hidden) closeMenu();
}
function openMenu() {
  dlMenu.hidden = false;
  dlToggle.setAttribute("aria-expanded", "true");
  const first = dlMenu.querySelector(".menu-item:not(:disabled)");
  if (first) first.focus();
}
function closeMenu() {
  dlMenu.hidden = true;
  dlToggle.setAttribute("aria-expanded", "false");
}
dlToggle.addEventListener("click", (e) => {
  e.stopPropagation();
  dlMenu.hidden ? openMenu() : closeMenu();
});
dlMenu.addEventListener("click", (e) => {
  if (e.target.closest(".menu-item")) closeMenu(); // app.js handlers already ran
});
document.addEventListener("click", (e) => {
  if (!dlMenu.hidden && !e.target.closest(".menu")) closeMenu();
});
dlMenu.addEventListener("keydown", (e) => {
  const items = [...dlMenu.querySelectorAll(".menu-item:not(:disabled)")];
  const i = items.indexOf(document.activeElement);
  if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length].focus(); }
  if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !dlMenu.hidden) { closeMenu(); dlToggle.focus(); }
});
// app.js and compose.js enable/disable the format buttons after each render.
const dlObserver = new MutationObserver(refreshDownloads);
for (const id of DL_ITEMS) dlObserver.observe($(id), { attributes: true, attributeFilter: ["disabled"] });
refreshDownloads();

$("copy-txt").addEventListener("click", async () => {
  const text = state.result && state.result.ascii;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    setStatus("Copied the text to the clipboard.", "done");
  } catch (_) {
    setStatus("Couldn't copy (the browser blocked clipboard access). Use Download → Plain text.", "error");
  }
});

// ---------- empty state & drop anywhere ----------

const empty = $("empty");
// Hide the welcome screen once anything has been rendered into the preview.
new MutationObserver(() => {
  if ($("preview").getAttribute("srcdoc")) empty.hidden = true;
}).observe($("preview"), { attributes: true, attributeFilter: ["srcdoc"] });

function chooseFile(inputId) {
  // Opening the picker must happen inside the click for browsers to allow it.
  $(inputId).click();
}
document.querySelectorAll("[data-quick]").forEach((b) => {
  b.addEventListener("click", () => {
    const kind = b.dataset.quick;
    setTab(kind);
    if (kind === "image") chooseFile("file");
    if (kind === "video") chooseFile("video_file");
    if (kind === "text") {
      empty.hidden = true;
      $("text").focus();
    }
  });
});

function feedFile(inputId, file) {
  const dt = new DataTransfer();
  dt.items.add(file);
  $(inputId).files = dt.files;
  $(inputId).dispatchEvent(new Event("change"));
}

const veil = $("drop-veil");
let dragDepth = 0;
const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
window.addEventListener("dragenter", (e) => {
  if (!hasFiles(e)) return;
  dragDepth++;
  veil.hidden = false;
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) veil.hidden = true;
});
window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener("drop", (e) => {
  dragDepth = 0;
  veil.hidden = true;
  if (e.defaultPrevented || !hasFiles(e)) return; // the image drop zone handled it
  e.preventDefault();
  const file = e.dataTransfer.files[0];
  if (!file) return;
  const isVideo = file.type.startsWith("video/") || /\.(mkv|webm|mov|avi|mp4)$/i.test(file.name);
  if (file.type.startsWith("image/") && file.type !== "image/gif") {
    if (state.tab === "compose") { feedFile("cmp_file", file); return; }
    setTab("image");
    feedFile("file", file);
  } else if (isVideo || file.type === "image/gif") {
    setTab("video");
    feedFile("video_file", file);
  } else {
    setStatus(`Can't use ${file.name}: drop an image or a video.`, "error");
  }
});

// Switching to a tab with nothing to show used to leave the previous tab's
// art (and its downloads) in place; show the start screen instead.
function tabHasContent(name) {
  if (name === "image") return !!state.file;
  if (name === "text") return $("text").value.trim() !== "";
  if (name === "video") return !!state.videoFile;
  if (name === "compose") return typeof cmp !== "undefined" && cmp.layers.length > 0;
  return true;
}
const appSetTab = setTab;
// eslint-disable-next-line no-global-assign
setTab = function (name) {
  appSetTab(name);
  if (tabHasContent(name)) return;
  state.result = null;
  $("preview").srcdoc = "";
  for (const id of DL_ITEMS) $(id).disabled = true;
  empty.hidden = false;
  setStatus(name === "compose" ? "Add an image or text layer to begin."
    : name === "text" ? "Type some text to begin."
    : name === "video" ? "Choose a video to begin."
    : "Choose an image to begin.", "");
};

$("video_file").addEventListener("change", (e) => {
  const f = e.target.files[0];
  $("video-name").textContent = f ? `Selected: ${f.name}` : "";
});

// ---------- contextual hints & card summaries ----------

const MODE_HINTS = {
  glyph: "Picks the character that best matches each patch. Great for logos, drawings, and cartoons.",
  braille: "Packs 2×4 dots into each character for the most detail. Great for photos; try Invert on dark terminals.",
};
const OVERLAY_STOPS = {
  rainbow: ["#ff0000", "#ff8000", "#ffee00", "#00c040", "#0080ff", "#8000ff"],
  sunset: ["#ff5e62", "#ff9966", "#ffd86b"],
  ocean: ["#00c6ff", "#0072ff", "#3f2b96"],
  fire: ["#ffe259", "#ff8c00", "#e52d27"],
  forest: ["#a8e063", "#56ab2f", "#134e2a"],
  neon: ["#00e5ff", "#ff00e5"],
  matrix: ["#00ff41", "#008f11"],
  mono: ["#ffffff", "#606060"],
};

function setState(key, text, on) {
  const el = document.querySelector(`[data-state="${key}"]`);
  if (!el) return;
  el.textContent = text;
  el.classList.toggle("on", !!on);
}

function refreshHints() {
  $("mode-hint").textContent = MODE_HINTS[$("mode").value] || "";
  const plain = ["box", "banner", "figlet"].includes($("text_style").value);
  $("text-style-hint").hidden = !plain || !!$("text_anim").value;

  const cap = $("caption_text").value.trim();
  setState("caption", cap ? `“${cap.length > 18 ? cap.slice(0, 17) + "…" : cap}” · ${$("caption_pos").value}` : "Off", !!cap);

  const w = $("out_cols").value, h = $("out_rows").value;
  const mw = $("max_cols").value, mh = $("max_rows").value;
  setState("size", w || h ? `${w || "auto"} × ${h || "auto"}` : (mw || mh ? "Capped" : "Auto"), !!(w || h || mw || mh));

  const preset = $("overlay_preset").value;
  const stops = preset === "custom" ? [$("overlay_c1").value, $("overlay_c2").value] : OVERLAY_STOPS[preset];
  $("overlay-swatch").style.background = stops
    ? (stops.length > 1 ? `linear-gradient(90deg, ${stops.join(", ")})` : stops[0])
    : "none";
}
const controls = $("controls");
controls.addEventListener("input", refreshHints);
controls.addEventListener("change", refreshHints);
// The resize handles write the size fields directly (no events), so also
// refresh after each render, when the status line changes.
new MutationObserver(refreshHints).observe($("status"), { childList: true, characterData: true, subtree: true });
refreshHints();

// ---------- remember which cards are open ----------

document.querySelectorAll("details[data-remember]").forEach((d, i) => {
  const key = `am.open.${d.id || d.closest("section").id || i}`;
  try {
    const v = localStorage.getItem(key);
    if (v !== null) d.open = v === "1";
  } catch (_) { /* storage blocked: use the default */ }
  d.addEventListener("toggle", () => {
    try { localStorage.setItem(key, d.open ? "1" : "0"); } catch (_) { /* ignore */ }
  });
});

// ---------- keyboard ----------

document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    render();
  }
});
