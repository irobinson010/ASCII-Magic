"use strict";

// "Translate" controls next to text boxes. The translation replaces the
// box's text so it can be proofread and edited before rendering (machine
// translation of short phrases is hit and miss). Relies on app.js: $,
// setStatus, readJson, autoRender.

const tr = { engine: false, installed: [], canInstall: false, available: [], loaded: false };

const LANG_NAMES = {
  ja: "Japanese", zh: "Chinese", ko: "Korean", es: "Spanish", fr: "French", de: "German",
  it: "Italian", pt: "Portuguese", ru: "Russian", ar: "Arabic", hi: "Hindi", nl: "Dutch",
  pl: "Polish", tr: "Turkish", uk: "Ukrainian", vi: "Vietnamese", id: "Indonesian", sv: "Swedish",
};

function langName(code) {
  const hit = tr.available.find((a) => a[1] === code && a[0] === "en");
  return (hit && hit[3]) || LANG_NAMES[code] || code;
}

// Targets from English: installed first, then (if the server may download)
// everything else the model index offers.
function targetOptions() {
  const installed = tr.installed.filter(([s]) => s === "en").map(([, d]) => d);
  const opts = installed.map((d) => ({ code: d, label: langName(d), installed: true }));
  if (tr.canInstall) {
    const more = tr.available
      .filter(([s, d]) => s === "en" && !installed.includes(d))
      .map(([, d, , dn]) => ({ code: d, label: `${dn || langName(d)} (download)`, installed: false }));
    more.sort((a, b) => (a.code === "ja" ? -1 : b.code === "ja" ? 1 : a.label.localeCompare(b.label)));
    opts.push(...more);
  }
  return opts;
}

function fillSelect(sel) {
  const prev = sel.value;
  sel.replaceChildren();
  const opts = targetOptions();
  for (const o of opts) {
    const el = document.createElement("option");
    el.value = o.code;
    el.textContent = o.label;
    el.dataset.installed = o.installed ? "1" : "";
    sel.append(el);
  }
  if (prev && opts.some((o) => o.code === prev)) sel.value = prev;
  else if (opts.some((o) => o.code === "ja")) sel.value = "ja";
  const row = sel.closest(".translate-row");
  const btn = row && row.querySelector(".translate-btn");
  let why = "";
  if (!tr.engine) why = 'Translation needs the server installed with the [translate] extra: pip install "ascii-magic-tools[translate]"';
  else if (!opts.length) why = "No translation models installed. In a terminal: ascii-magic translate install en ja";
  sel.disabled = !!why;
  if (btn) { btn.disabled = !!why; btn.title = why || "Translate from English (machine translation: proofread it)"; }
  if (!opts.length) {
    const el = document.createElement("option");
    el.textContent = tr.engine ? "No languages installed" : "Translation unavailable";
    sel.append(el);
  }
}

async function loadLanguages() {
  try {
    const res = await fetch("/api/translate/languages");
    const d = await res.json();
    Object.assign(tr, {
      engine: !!d.engine, installed: d.installed || [], canInstall: !!d.can_install,
      available: d.available || [], loaded: true,
    });
  } catch (_) {
    tr.loaded = true;
  }
  document.querySelectorAll(".translate-to").forEach(fillSelect);
}

async function translateInto(targetId, to) {
  const box = $(targetId);
  if (!box) return;
  const text = box.value.trim();
  if (!text) { setStatus("Type some English text first.", "error"); return; }
  const opt = targetOptions().find((o) => o.code === to);
  try {
    if (opt && !opt.installed) {
      setStatus(`Downloading the English → ${opt.label.replace(" (download)", "")} model (~100 MB, once)…`, "busy");
      const res = await fetch("/api/translate/install", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ from: "en", to }),
      });
      const body = await readJson(res);
      if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
      tr.installed = body.installed;
      document.querySelectorAll(".translate-to").forEach(fillSelect);
    }
    setStatus("Translating…", "busy");
    const res = await fetch("/api/translate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: box.value, from: "en", to }),
    });
    const body = await readJson(res);
    if (!res.ok) throw new Error(body.detail || `HTTP ${res.status}`);
    box.dataset.original = box.value;
    box.value = body.text;
    // Same events as typing, so previews, compose layers and auto-render update.
    box.dispatchEvent(new Event("input", { bubbles: true }));
    box.dispatchEvent(new Event("change", { bubbles: true }));
    setStatus(`Translated to ${langName(to)}. Machine translation: check it and edit the text if needed.`, "");
  } catch (err) {
    setStatus(`Translation failed: ${err.message}`, "error");
  }
}

function wireRow(row) {
  const sel = row.querySelector(".translate-to");
  const btn = row.querySelector(".translate-btn");
  btn.addEventListener("click", (e) => {
    e.preventDefault();
    translateInto(row.dataset.target, sel.value);
  });
  if (tr.loaded) fillSelect(sel);
}

// For compose.js's layer editor: a control bound to a given text box id.
window.translateControl = function (targetId) {
  const row = document.createElement("span");
  row.className = "seed-row translate-row";
  row.dataset.target = targetId;
  const sel = document.createElement("select");
  sel.className = "translate-to";
  sel.setAttribute("aria-label", "Translate to");
  const btn = document.createElement("button");
  btn.className = "translate-btn";
  btn.type = "button";
  btn.textContent = "Translate";
  row.append(sel, btn);
  wireRow(row);
  return row;
};

document.querySelectorAll(".translate-row").forEach(wireRow);
loadLanguages();
