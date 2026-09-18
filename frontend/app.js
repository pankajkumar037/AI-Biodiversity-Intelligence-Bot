/* Darukaa UI. Plain JS, no build step. Talks to the FastAPI backend over SSE. */
(() => {
  "use strict";

  // ── config and state ───────────────────────────────────────────
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const PIPELINE = [
    "intake", "normalize", "geo_enrich", "intent", "slot_check",
    "diagnose", "root_cause", "leverage", "candidates", "combine", "sequence",
    "plan_queries", "retrieve", "adjudicate", "verify", "render",
  ];
  const TERMINALS = { ask: "asked a question", explain: "answered from memory",
                      concept: "quoted the corpus", out_of_scope: "redirected" };

  const state = {
    apiBase: localStorage.getItem("darukaa.api") || "",
    sessionId: localStorage.getItem("darukaa.session") || newId(),
    mode: "text",
    busy: false,
    trace: {},           // accumulates across nodes for the current turn
    turns: [],           // {turn, message, answer, trace, result}
    lastResult: null,
  };
  localStorage.setItem("darukaa.session", state.sessionId);

  function newId() {
    return "s-" + Math.random().toString(36).slice(2, 8) + Date.now().toString(36).slice(-4);
  }
  const api = (path) => (state.apiBase || "") + path;
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n, d = 2) => (typeof n === "number" ? n.toFixed(d) : "—");
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // ── boot ───────────────────────────────────────────────────────
  $("#session-id").textContent = state.sessionId;
  $("#api-base").value = state.apiBase;
  $("#api-base").addEventListener("change", (e) => {
    state.apiBase = e.target.value.trim().replace(/\/$/, "");
    localStorage.setItem("darukaa.api", state.apiBase);
    checkHealth();
  });
  buildRail();
  checkHealth();
  loadStats();

  async function checkHealth() {
    const pill = $("#health");
    pill.className = "pill pill-muted"; pill.textContent = "checking backend…";
    try {
      const r = await fetch(api("/health")); const j = await r.json();
      const ready = (j.indexes || []).filter((i) => i.status === "READY").length;
      pill.className = "pill pill-ok";
      pill.textContent = `backend ok · ${j.counts.chunks} chunks · ${ready}/2 indexes`;
    } catch (e) {
      pill.className = "pill pill-bad"; pill.textContent = "backend unreachable";
    }
  }
  async function loadStats() {
    try {
      const j = await (await fetch(api("/knowledge/stats"))).json();
      const dd = $$("#stats dd");
      dd[0].textContent = j.chunks; dd[1].textContent = `${j.traversable_edges} / ${j.edges}`;
      dd[2].textContent = j.n_documents;
    } catch (e) { /* stats are decoration */ }
  }

  // ── pipeline rail ──────────────────────────────────────────────
  function buildRail() {
    const rail = $("#rail");
    rail.innerHTML = PIPELINE.map((n) =>
      `<span class="rail-node" data-node="${n}"><i class="dot"></i>${n}<span class="ms"></span></span>`
    ).join("");
  }
  function railReset() {
    $$(".rail-node").forEach((el) => { el.className = "rail-node"; $(".ms", el).textContent = ""; });
  }
  function railMark(node, ms) {
    const el = $(`.rail-node[data-node="${node}"]`);
    if (!el) return;
    el.classList.remove("active"); el.classList.add("done");
    if (ms != null) $(".ms", el).textContent = ` ${ms}ms`;
    const next = PIPELINE[PIPELINE.indexOf(node) + 1];
    const nel = next && $(`.rail-node[data-node="${next}"]`);
    if (nel && !nel.classList.contains("done")) nel.classList.add("active");
  }
  function railFinish() { $$(".rail-node.active").forEach((el) => el.classList.remove("active")); }

  // ── mode and tabs ──────────────────────────────────────────────
  $("#mode").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    state.mode = b.dataset.mode;
    $$("#mode button").forEach((x) => x.classList.toggle("on", x === b));
    $$(".composer [data-for]").forEach((x) => { x.hidden = x.dataset.for !== state.mode; });
    $("#composer-hint").textContent = {
      text: "Enter to send · Shift+Enter for a new line",
      json: "Structured profile goes straight to /analyze — no conversation, no clarifying question",
      coords: "Coordinates trigger SoilGrids and NASA POWER lookups; user values still win",
    }[state.mode];
  });
  $("#tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    $$("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
    $$(".panel").forEach((p) => p.classList.toggle("on", p.dataset.panel === b.dataset.tab));
  });
  function setTabCount(tab, n) {
    const b = $(`#tabs button[data-tab="${tab}"]`);
    let c = $(".cnt", b); if (!c) { c = document.createElement("span"); c.className = "cnt"; b.appendChild(c); }
    c.textContent = n ? n : "";
  }

  // ── composer ───────────────────────────────────────────────────
  $("#text-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#composer").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (state.busy) return;
    if (state.mode === "text") {
      const t = $("#text-input").value.trim(); if (!t) return;
      $("#text-input").value = "";
      await runTurn({ kind: "chat", message: t });
    } else if (state.mode === "json") {
      let profile;
      try { profile = JSON.parse($("#json-input").value); }
      catch (err) { addError("That JSON does not parse: " + err.message); return; }
      const constraints = $("#json-constraints").value.split(",").map((s) => s.trim()).filter(Boolean);
      await runTurn({ kind: "analyze", profile, constraints });
    } else {
      const lat = parseFloat($("#lat").value), lon = parseFloat($("#lon").value);
      if (Number.isNaN(lat) || Number.isNaN(lon)) { addError("Give both a latitude and a longitude."); return; }
      const note = $("#coords-note").value.trim();
      await runTurn({ kind: "chat", message: note || `Site at ${lat}, ${lon}`, profile_patch: { lat, lon } });
    }
  });

  $("#reset").addEventListener("click", async () => {
    if (state.busy) return;
    try { await fetch(api(`/session/${state.sessionId}/reset`), { method: "POST" }); } catch (e) { /* fine */ }
    state.sessionId = newId(); localStorage.setItem("darukaa.session", state.sessionId);
    $("#session-id").textContent = state.sessionId;
    state.turns = []; state.trace = {}; state.lastResult = null;
    $("#messages").innerHTML = "";
    railReset(); renderAll({}); setConfidence(null);
    $("#download").disabled = true;
    $$("#demo-turns li").forEach((li) => (li.className = ""));
  });

  $("#demo").addEventListener("click", async () => {
    if (state.busy) return;
    const lis = $$("#demo-turns li");
    $("#mode button[data-mode=text]").click();
    for (let i = 0; i < lis.length; i++) {
      lis.forEach((li, j) => (li.className = j < i ? "done" : j === i ? "active" : ""));
      await runTurn({ kind: "chat", message: lis[i].textContent.trim() });
      if (state.lastResult == null) break;
    }
    lis.forEach((li) => (li.className = "done"));
  });

  $("#download").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify({ session_id: state.sessionId, turns: state.turns }, null, 2)],
                          { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = `darukaa-${state.sessionId}.json`; a.click();
    URL.revokeObjectURL(a.href);
  });

  // ── a turn ─────────────────────────────────────────────────────
  async function runTurn(req) {
    state.busy = true; $("#send").disabled = true; $("#demo").disabled = true;
    state.trace = {}; state.lastResult = null;
    railReset(); setConfidence(null);
    $$(".rail-node")[0].classList.add("active");

    const userText = req.kind === "chat" ? req.message : "structured profile → /analyze";
    addUser(userText);
    const { think, body } = addAssistant();
    const log = $("ol", think);

    const url = req.kind === "chat" ? "/chat/stream" : "/analyze/stream";
    const payload = req.kind === "chat"
      ? { session_id: state.sessionId, message: req.message, profile_patch: req.profile_patch || null }
      : { profile: req.profile, constraints: req.constraints, audience: $("#audience").value || null };
    if (req.kind === "chat" && $("#audience").value) {
      payload.profile_patch = { ...(payload.profile_patch || {}), audience: $("#audience").value };
    }

    let finalResult = null;
    try {
      await streamSSE(api(url), payload, (ev) => {
        if (ev.type === "node") {
          onNode(ev, log);
        } else if (ev.type === "done") {
          finalResult = ev.result;
          think.classList.remove("running");
          $(".state", think).textContent = `done in ${(ev.elapsed_ms / 1000).toFixed(1)}s`;
        } else if (ev.type === "error") {
          throw new Error(`${ev.error}: ${ev.message}`);
        }
      });
    } catch (err) {
      think.classList.remove("running");
      $(".state", think).textContent = "failed";
      body.innerHTML = `<p class="msg-error">${esc(err.message)}</p>`;
      state.busy = false; $("#send").disabled = false; $("#demo").disabled = false;
      return;
    }

    railFinish();
    if (finalResult) {
      state.lastResult = finalResult;
      if (finalResult.trace) state.trace = finalResult.trace;
      renderAll(state.trace);
      setConfidence(finalResult.confidence);
      await revealAnswer(body, finalResult);
      state.turns.push({ turn: finalResult.turn ?? state.turns.length + 1, message: userText,
                         answer: finalResult.answer, question: finalResult.question,
                         intent: finalResult.intent, confidence: finalResult.confidence,
                         trace: finalResult.trace });
      $("#download").disabled = false;
      think.open = false;
    }
    state.busy = false; $("#send").disabled = false; $("#demo").disabled = false;
  }

  async function streamSSE(url, payload, onEvent) {
    const res = await fetch(url, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        for (const line of chunk.split("\n")) {
          if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
        }
      }
    }
  }

  // ── node events → thinking log + live tabs ─────────────────────
  function onNode(ev, log) {
    const u = ev.update || {};
    if (u.trace) Object.assign(state.trace, u.trace);
    const desc = describe(ev.node, u);
    const li = document.createElement("li");
    if (desc.warn) li.classList.add("warn");
    li.innerHTML = `<span class="n">${esc(ev.node)}</span><span class="d" title="${esc(desc.text)}">${esc(desc.text)}</span><span class="t">${ev.elapsed_ms}ms</span>`;
    log.appendChild(li);
    log.parentElement.scrollIntoView({ block: "nearest" });
    if (TERMINALS[ev.node]) { railFinish(); } else { railMark(ev.node, ev.elapsed_ms); }
    renderAll(state.trace);
  }

  function describe(node, u) {
    const t = u.trace || {};
    switch (node) {
      case "intake": {
        const ex = t.intake?.extracted || {}; const k = Object.keys(ex);
        const rej = t.intake?.rejected || [];
        return { text: k.length ? `read ${k.join(", ")}` : "nothing new stated", warn: rej.length > 0 };
      }
      case "normalize": { const k = Object.keys(t.inferred || {}); return { text: k.length ? `inferred ${k.join(", ")} (marked as such)` : "nothing to infer" }; }
      case "geo_enrich": { const g = t.geo; return { text: g ? (g.result === "no data" ? "coordinates known, no API data" : `SoilGrids/NASA POWER at ${fmt(g.lat, 3)}, ${fmt(g.lon, 3)}`) : "no location" }; }
      case "intent": return { text: `classified as ${u.intent || t.intent || "new_info"}` };
      case "slot_check": { const s = t.slot_check; return { text: u.question ? `asking for ${s?.asked} (highest flip score)` : "enough is known to proceed" }; }
      case "diagnose": { const d = t.diagnosis || {}; const f = d.flags || []; const p = (d.patterns || []).map((x) => x.name);
        return { text: f.length ? `${f.length} flags: ${f.join(", ")}${p.length ? " · patterns: " + p.join(", ") : ""}` : "no flags raised" }; }
      case "root_cause": return { text: `${(t.root_causes || []).length} causal chains traced upstream` };
      case "leverage": { const l = t.leverage || []; return { text: l.length ? `highest leverage: ${l.slice(0, 3).map((x) => x.node).join(", ")}` : "no leverage points" }; }
      case "candidates": { const c = t.candidates || []; const e = t.excluded || []; const d = c.filter((x) => x.downgraded).length;
        return { text: `${c.length} scored, ${d} downgraded, ${e.length} excluded` }; }
      case "combine": { const c = (t.combinations || [])[0]; return { text: c ? `best pair: ${c.combo.join(" + ")}${c.synergy ? " (synergy)" : ""}` : "no pairs" }; }
      case "sequence": { const s = t.sequence || []; return { text: s.map((x) => x.practice_id).join(" → ") || "no sequence" }; }
      case "plan_queries": return { text: "support, risk and path queries planned" };
      case "retrieve": { const q = t.queries || []; const e = t.evidence || []; const worst = Math.max(0, ...q.map((x) => x.filter_level));
        return { text: `${q.length} queries → ${e.length} chunks${worst ? ` (filter fell back to level ${worst})` : ""}`, warn: worst > 0 }; }
      case "adjudicate": return { text: `reasoning call, attempt ${u.verify_attempts ?? "?"}` };
      case "verify": { const v = t.verification; const fb = u.verify_feedback;
        if (fb) { const n = (fb.match(/^- /gm) || []).length; return { text: `${n} check(s) failed → retrying with feedback`, warn: true }; }
        if (v?.degraded) return { text: `retries spent → degraded: ${v.stripped_numbers || 0} numbers stripped, ${v.dropped_steps || 0} steps dropped`, warn: true };
        return { text: "all checks passed" }; }
      case "render": return { text: "answer assembled from verified output" };
      default: return { text: TERMINALS[node] || "" };
    }
  }

  // ── messages ───────────────────────────────────────────────────
  function addUser(text) {
    const el = document.createElement("div"); el.className = "msg msg-user";
    el.innerHTML = `<span class="who">you</span><div class="body">${esc(text)}</div>`;
    $("#messages").appendChild(el); el.scrollIntoView({ block: "end" });
  }
  function addError(text) {
    const el = document.createElement("div"); el.className = "msg msg-error"; el.textContent = text;
    $("#messages").appendChild(el); el.scrollIntoView({ block: "end" });
  }
  function addAssistant() {
    const el = document.createElement("div"); el.className = "msg msg-assistant";
    el.innerHTML = `<span class="who">dk</span><div class="body">
      <details class="think running" open><summary><span>pipeline</span><span class="state">running…</span></summary><ol></ol></details>
      <div class="answer"><p class="plain cursor"></p></div></div>`;
    $("#messages").appendChild(el); el.scrollIntoView({ block: "end" });
    return { think: $(".think", el), body: $(".answer", el) };
  }

  // The renderer's text has fixed section headers; parse them into blocks so the
  // answer reads as a document rather than a code dump.
  const HEADERS = ["SITE", "DIAGNOSIS", "RECOMMENDED", "DOWNGRADED", "EXCLUDED", "REASONING",
                   "REVIEW", "CONFIDENCE", "SOURCES", "WHAT CHANGED", "NOTES"];
  function parseAnswer(text) {
    const lines = (text || "").split("\n"); const secs = []; let cur = null;
    for (const line of lines) {
      const h = HEADERS.find((x) => line.startsWith(x + " ") || line === x || line.startsWith(x + "\t"));
      if (h) { cur = { h, inline: line.slice(h.length).trim(), body: [] }; secs.push(cur); }
      else if (cur) cur.body.push(line);
    }
    return secs;
  }
  function recCards(body) {
    const cards = []; let c = null;
    for (const raw of body) {
      const m = raw.match(/^\[(\d+)\] (.*)$/);
      if (m) { c = { n: m[1], title: m[2], rows: [] }; cards.push(c); continue; }
      const r = raw.match(/^\s{4}([A-Za-z ]+?):\s(.*)$/);
      if (r && c) c.rows.push({ k: r[1], v: r[2] });
    }
    return cards;
  }
  function sectionHTML(s) {
    const cls = "sec sec-" + s.h.toLowerCase().replace(" ", "-");
    if (s.h === "RECOMMENDED" && s.body.some((l) => /^\[\d+\]/.test(l))) {
      const cards = recCards(s.body).map((c) => `<div class="rec"><p class="rec-t">${esc(c.n)}. ${esc(c.title)}</p>${
        c.rows.map((r) => `<div class="row ${r.k.toLowerCase()==="risk"?"risk":""} ${r.k.toLowerCase()==="estimate"?"est":""}"><b>${esc(r.k)}</b><span>${esc(r.v)}</span></div>`).join("")
      }</div>`).join("");
      return `<div class="${cls}"><p class="sec-h">${s.h}</p>${cards}</div>`;
    }
    const inline = s.inline ? esc(s.inline) + (s.body.length ? "\n" : "") : "";
    const body = s.body.map((l) => esc(l)).join("\n").replace(/\n+$/, "");
    return `<div class="${cls}"><p class="sec-h">${s.h}</p><pre class="sec-b">${inline}${body}</pre></div>`;
  }

  async function revealAnswer(body, result) {
    body.innerHTML = "";
    if (result.question && (!result.trace?.diagnosis)) {
      const [q, why] = String(result.answer || result.question).split("\n\nWhy this matters: ");
      body.innerHTML = `<p class="question">${esc(q)}</p>${why ? `<p class="question-why">Why this matters: ${esc(why)}</p>` : ""}`;
      return;
    }
    const secs = parseAnswer(result.answer);
    if (!secs.length) {
      const p = document.createElement("p"); p.className = "plain cursor"; body.appendChild(p);
      const words = String(result.answer || "").split(/(\s+)/);
      for (const w of words) { p.textContent += w; if (w.trim()) await sleep(6); }
      p.classList.remove("cursor");
      return;
    }
    for (const s of secs) {
      const wrap = document.createElement("div"); wrap.innerHTML = sectionHTML(s);
      body.appendChild(wrap.firstElementChild);
      wrap.remove();
      body.lastElementChild.scrollIntoView({ block: "nearest" });
      await sleep(s.h === "RECOMMENDED" ? 220 : 110);
    }
  }

  // ── inspector panels ───────────────────────────────────────────
  function renderAll(t) {
    renderProfile(t); renderReasoning(t); renderGraph(t); renderRetrieval(t);
    renderVerification(t); renderLandscape(t);
  }
  const panel = (name) => $(`.panel[data-panel="${name}"]`);

  function setConfidence(c) {
    const el = $("#conf");
    if (!c) { el.innerHTML = `<span class="conf-label">confidence</span><span class="conf-value">—</span>`; return; }
    const b = c.breakdown || {};
    el.innerHTML = `<span class="conf-label">confidence</span><span class="conf-value ${c.band}">${c.band} ${fmt(c.value)}</span>
      <div class="conf-bars">${["evidence", "context", "agreement", "data_quality"].map((k) =>
        `<span class="conf-bar"><i style="--w:${Math.round((b[k] || 0) * 100)}%"></i>${k.replace("_", " ")} ${fmt(b[k])}</span>`).join("")}</div>`;
  }

  function renderProfile(t) {
    const p = t.profile; const el = panel("profile");
    if (!p || !Object.keys(p).length) {
      const ex = t.intake?.extracted;
      el.innerHTML = ex && Object.keys(ex).length
        ? `<p class="sub">extracted this turn</p>` + kvTable(ex) : `<p class="empty">No site yet.</p>`;
      return;
    }
    const rows = Object.entries(p).map(([k, v]) =>
      `<tr><td class="mono">${esc(k)}</td><td class="mono">${esc(fmtVal(v.value))}</td><td><span class="src ${esc(v.source)}">${esc(v.source)}</span></td><td class="num">${v.uncertainty != null ? fmt(v.uncertainty) : ""}</td></tr>`).join("");
    let html = `<table class="tbl"><thead><tr><th>field</th><th>value</th><th>source</th><th>unc.</th></tr></thead><tbody>${rows}</tbody></table>`;
    const rej = t.intake?.rejected || [];
    if (rej.length) html += `<p class="note warn">${rej.map(esc).join("<br>")}</p>`;
    if (t.inferred && Object.keys(t.inferred).length)
      html += `<p class="note">Inferred, not stated: ${Object.entries(t.inferred).map(([k, v]) => `${esc(k)} = ${esc(v)}`).join(", ")}. A user value would replace these.</p>`;
    if (t.slot_check) html += `<p class="sub">value of information</p><table class="tbl"><thead><tr><th>slot</th><th>flip score</th><th>outcomes</th></tr></thead><tbody>${
      (t.slot_check.ranking || []).map((r) => `<tr class="${r.slot === t.slot_check.asked ? "" : "dim"}"><td class="mono">${esc(r.slot)}</td><td class="num">${fmt(r.flip_score)}</td><td class="num">${r.distinct_outcomes}</td></tr>`).join("")}</tbody></table>`;
    el.innerHTML = html;
    setTabCount("profile", Object.keys(p).length);
  }
  function fmtVal(v) { return typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "")) : v; }
  function kvTable(o) {
    return `<table class="tbl"><tbody>${Object.entries(o).map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td class="mono">${esc(fmtVal(v))}</td></tr>`).join("")}</tbody></table>`;
  }

  function renderReasoning(t) {
    const el = panel("reasoning"); const d = t.diagnosis;
    if (!d) { el.innerHTML = `<p class="empty">Diagnosis and ranking appear here as the pipeline runs.</p>`; return; }
    let html = `<p class="sub">flags · rules fired</p><div class="chips">${
      (d.flags || []).map((f) => `<span class="chip flag">${esc(f)}</span>`).join("")}${
      (d.fired_rules || []).map((r) => `<span class="chip">${esc(r)}</span>`).join("")}</div>`;
    if ((d.patterns || []).length) html += `<p class="sub">compound patterns</p><div class="chips">${
      d.patterns.map((p) => `<span class="chip pattern" title="${esc((p.loop || []).join(" → "))}">${esc(p.name)} [${esc(p.rule_id)}]</span>`).join("")}</div>`;

    if ((t.root_causes || []).length) html += `<p class="sub">root causes (upstream)</p>` +
      t.root_causes.slice(0, 6).map((c) => `<div class="path"><span class="w">${esc(c.id)}</span> ${c.path.map(esc).join(' <span class="arr">→</span> ')}</div>`).join("");

    if ((t.leverage || []).length) html += `<p class="sub">leverage</p><table class="tbl"><thead><tr><th>node</th><th>score</th><th>reaches</th><th>loop</th></tr></thead><tbody>${
      t.leverage.slice(0, 5).map((l) => `<tr><td class="mono">${esc(l.node)}</td><td class="num">${fmt(l.score)}</td><td class="mono">${(l.reaches || []).map(esc).join(", ")}</td><td>${l.on_loop ? "yes" : ""}</td></tr>`).join("")}</tbody></table>`;

    if ((t.candidates || []).length) {
      html += `<p class="sub">candidates (ranking scores, not effect sizes)</p>` + t.candidates.map((c) => {
        const risk = (c.risks_applied || [])[0];
        return `<div class="cand"><div class="cand-h"><b>${esc(c.name || c.practice_id)}</b><span class="score">${fmt(c.suitability)}</span></div>
          <div class="bar ${c.downgraded ? "down" : ""}"><i style="--w:${Math.round(c.suitability * 100)}%"></i></div>
          <div class="cand-m">coverage ${fmt(c.coverage)} · benefit ${c.benefit != null ? (c.benefit >= 0 ? "+" : "") + fmt(c.benefit) : "—"} · penalty ${fmt(c.penalty)}</div>
          ${risk ? `<div class="cand-r">↓ ${esc(risk.risk)} (−${fmt(risk.penalty)})</div>` : ""}</div>`;
      }).join("");
    }
    if ((t.excluded || []).length) html += `<p class="sub">excluded</p>` + t.excluded.map((e) =>
      `<div class="cand excluded"><div class="cand-h"><b>${esc(e.name || e.practice_id)}</b></div><div class="cand-m">${esc(e.reason)}</div></div>`).join("");
    if ((t.combinations || []).length) html += `<p class="sub">best combinations</p>` + t.combinations.map((c) =>
      `<div class="path">${c.combo.map(esc).join(' <span class="arr">+</span> ')} <span class="w">${fmt(c.score)}</span>${c.synergy ? `<br><span class="w">${esc(c.synergy)}</span>` : ""}</div>`).join("");
    if ((t.sequence || []).length) html += `<p class="sub">sequence</p><table class="tbl"><thead><tr><th>#</th><th>practice</th><th>horizon</th><th>needs first</th></tr></thead><tbody>${
      t.sequence.map((s) => `<tr><td class="num">${s.order}</td><td>${esc(s.name)}${s.added_as_prerequisite ? ' <span class="chip">prerequisite</span>' : ""}</td><td class="mono">${esc(s.time_horizon)}</td><td class="mono">${(s.requires_first || []).map(esc).join(", ")}</td></tr>`).join("")}</tbody></table>`;
    el.innerHTML = html;
    setTabCount("reasoning", (d.flags || []).length);
  }

  // Causal graph: the paths the engine actually walked, laid out left to right.
  function renderGraph(t) {
    const el = panel("graph");
    const edges = new Map(); const depth = new Map(); const practices = new Set(); let order = 0; const seen = new Map();
    const touch = (n, d) => { depth.set(n, Math.max(depth.get(n) ?? 0, d)); if (!seen.has(n)) seen.set(n, order++); };
    const addPath = (path, cls, w) => {
      path.forEach((n, i) => touch(n, i));
      for (let i = 0; i < path.length - 1; i++) {
        const k = path[i] + "→" + path[i + 1];
        if (!edges.has(k)) edges.set(k, { a: path[i], b: path[i + 1], cls, w });
      }
    };
    (t.root_causes || []).slice(0, 6).forEach((c) => addPath(c.path, "cause", null));
    (t.candidates || []).slice(0, 4).forEach((c) => {
      (c.paths || []).slice(0, 3).forEach((p) => { practices.add(p.path[0]); addPath(p.path, p.weight < 0 ? "neg" : "pos", p.weight); });
    });
    if (!edges.size) { el.innerHTML = `<p class="empty">Causal paths the engine actually traversed.</p>`; return; }

    const cols = new Map();
    for (const [n, d] of depth) { if (!cols.has(d)) cols.set(d, []); cols.get(d).push(n); }
    for (const list of cols.values()) list.sort((a, b) => seen.get(a) - seen.get(b));
    const maxD = Math.max(...cols.keys()); const maxRows = Math.max(...Array.from(cols.values(), (l) => l.length));
    const W = 440, BW = 112, BH = 18, GX = 44, GY = 8, PAD = 12;
    const colX = (d) => PAD + d * ((W - 2 * PAD - BW) / Math.max(1, maxD));
    const H = PAD * 2 + maxRows * (BH + GY);
    const pos = new Map();
    for (const [d, list] of cols) list.forEach((n, i) => pos.set(n, { x: colX(d), y: PAD + i * (BH + GY) }));

    const edgeSvg = Array.from(edges.values()).map((e) => {
      const a = pos.get(e.a), b = pos.get(e.b); if (!a || !b) return "";
      const x1 = a.x + BW, y1 = a.y + BH / 2, x2 = b.x, y2 = b.y + BH / 2; const cx = (x1 + x2) / 2;
      return `<path class="e ${e.cls}" d="M${x1},${y1} C${cx},${y1} ${cx},${y2} ${x2},${y2}"><title>${esc(e.a)} → ${esc(e.b)}${e.w != null ? ` (${fmt(e.w, 3)})` : ""}</title></path>`;
    }).join("");
    const nodeSvg = Array.from(pos.entries()).map(([n, p]) =>
      `<g class="${practices.has(n) ? "n-practice" : "n-metric"}" transform="translate(${p.x},${p.y})"><rect width="${BW}" height="${BH}" rx="2"/><text x="6" y="12.5">${esc(n.length > 17 ? n.slice(0, 16) + "…" : n)}<title>${esc(n)}</title></text></g>`).join("");
    el.innerHTML = `<div class="legend"><span class="pos"><i></i>raises</span><span class="neg"><i></i>lowers</span><span class="cause"><i></i>root-cause chain</span></div>
      <svg class="graph-svg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">${edgeSvg}${nodeSvg}</svg>
      <p class="note">Only edges whose condition holds for this site are drawn. Dashed edges are the trade-offs — the reason a practice gets downgraded rather than recommended.</p>`;
    setTabCount("graph", edges.size);
  }

  function renderRetrieval(t) {
    const el = panel("retrieval"); const qs = t.queries || []; const ev = t.evidence || [];
    if (t.concept) { el.innerHTML = `<p class="sub">concept lookup</p><div class="q"><div class="q-text">${esc(t.concept.query)}</div><div class="q-m">filter level ${t.concept.filter_level} · kept ${t.concept.kept.length}</div></div>`; return; }
    if (!qs.length && !ev.length) { el.innerHTML = `<p class="empty">Every query, its filter level, and what was kept.</p>`; return; }
    let html = `<p class="sub">queries (never the raw message)</p>` + qs.map((q) =>
      `<div class="q"><div class="q-h"><span class="purpose ${esc(q.purpose)}">${esc(q.purpose)}</span><span class="q-text">${esc(q.q)}</span></div>
       <div class="q-m">${q.practice_id ? esc(q.practice_id) + " · " : ""}filter level ${q.filter_level} · ${q.n_candidates} candidates · kept ${(q.kept || []).length}</div></div>`).join("");
    if (ev.length) html += `<p class="sub">evidence block (S# as cited)</p>` + ev.map((e) =>
      `<div class="evi"><div class="evi-h"><span class="lbl">${esc(e.label)}</span><span class="doc">${esc(e.doc_title)}${e.page != null ? `, p${e.page}` : ""}</span><span class="meta">${esc(e.evidence_level || "")} · ${fmt(e.score, 3)}</span></div></div>`).join("");
    el.innerHTML = html;
    setTabCount("retrieval", ev.length);
  }

  function renderVerification(t) {
    const el = panel("verification"); const v = t.verification;
    if (!v) { el.innerHTML = `<p class="empty">V1–V10 results, retries, and anything stripped.</p>`; return; }
    const order = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"];
    const label = { V1: "structure", V2: "sources exist", V3: "numbers ↔ claims", V4: "direction", V5: "climate fit", V6: "mechanism + risk query", V7: "mechanism supported", V8: "no filler", V9: "steps anchored", V10: "≥3 variables" };
    let html = `<p class="sub">checks · ${v.attempts} attempt${v.attempts === 1 ? "" : "s"}${v.degraded ? " · degraded" : ""}</p><div class="chips">${
      order.map((k) => { const s = v.checks?.[k]; return s ? `<span class="chip ${s}" title="${esc(label[k])}">${k} ${esc(label[k])}</span>` : ""; }).join("")}</div>`;
    if (v.degraded) html += `<p class="note warn">Retries exhausted. Removed rather than shown unverified: ${v.stripped_numbers || 0} numbers, ${v.dropped_steps || 0} reasoning steps, ${v.dropped_estimates || 0} estimates, ${v.dropped_recommendations || 0} recommendations.</p>`;
    if ((v.unresolved || []).length) html += `<p class="sub">unresolved</p>` + v.unresolved.map((u) => `<p class="note">${esc(u)}</p>`).join("");
    if ((v.warnings || []).length) html += `<p class="sub">warnings</p>` + v.warnings.map((u) => `<p class="note">${esc(u)}</p>`).join("");
    if (t.confidence) { const b = t.confidence.breakdown; html += `<p class="sub">confidence = evidence × context × agreement × data quality</p><table class="tbl"><tbody>${
      Object.entries(b).map(([k, x]) => `<tr><td class="mono">${esc(k)}</td><td class="num">${fmt(x)}</td><td><div class="bar"><i style="--w:${Math.round(x * 100)}%"></i></div></td></tr>`).join("")}<tr><td class="mono">= ${t.confidence.band}</td><td class="num">${fmt(t.confidence.value)}</td><td></td></tr></tbody></table>`; }
    if (t.timing_ms) html += `<p class="sub">timing</p>` + kvTable(Object.fromEntries(Object.entries(t.timing_ms).map(([k, x]) => [k, `${x} ms`])));
    el.innerHTML = html;
    const fails = Object.values(v.checks || {}).filter((s) => s === "fail").length;
    setTabCount("verification", fails ? `${fails} fail` : "");
  }

  function renderLandscape(t) {
    const el = panel("landscape"); const g = t.geo;
    if (!g) { el.innerHTML = `<p class="empty">Geo enrichment for the site, when coordinates are known.</p>`; return; }
    let html = `<p class="sub">location</p>` + kvTable({ lat: g.lat, lon: g.lon });
    if (g.soil && Object.keys(g.soil).length) html += `<p class="sub">SoilGrids v2.0 (250 m model, not a soil test)</p>` + kvTable(g.soil);
    if (g.climate && Object.keys(g.climate).length) html += `<p class="sub">NASA POWER climatology (0.5° grid)</p>` + kvTable(g.climate);
    if (g.result === "no data") html += `<p class="note warn">Coordinates were known but neither API returned data. The profile falls through to regional defaults or a question.</p>`;
    html += `<p class="note">Land-cover, habitat-diversity and fragmentation metrics are not computed in this build. The threshold rules for them exist and fire if you supply <code>natural_cover_percent</code> or <code>habitat_diversity_index</code> directly.</p>`;
    el.innerHTML = html;
  }
})();
