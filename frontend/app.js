/* Darukaa — conversation UI. Plain JS, no build step. Streams pipeline events over SSE. */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmt = (n, d = 2) => (typeof n === "number" ? n.toFixed(d) : "—");
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const newId = () => "s-" + Math.random().toString(36).slice(2, 8) + Date.now().toString(36).slice(-4);

  const state = {
    apiBase: localStorage.getItem("darukaa.api") || "",
    sessionId: localStorage.getItem("darukaa.session") || newId(),
    mode: "text", busy: false, turns: [], abort: null,
  };
  localStorage.setItem("darukaa.session", state.sessionId);
  const api = (p) => state.apiBase + p;

  // What the system is doing *after* a node finishes, i.e. what the next node does.
  const NEXT_LABEL = {
    start: "Working out what you're asking",
    intake: "Filling in what follows from it",
    normalize: "Looking up the location",
    geo_enrich: "Checking what's still needed",
    intent: "Reading what you said",
    slot_check: "Diagnosing the site against threshold rules",
    diagnose: "Tracing causes upstream through the graph",
    root_cause: "Finding the leverage points",
    leverage: "Scoring practices for this site",
    candidates: "Pairing practices that offset each other's risks",
    combine: "Ordering the plan by prerequisites",
    sequence: "Planning evidence searches",
    plan_queries: "Searching the corpus",
    retrieve: "Reasoning over the evidence",
    adjudicate: "Checking every claim against its source",
    verify: "Writing it up",
    render: "Done",
  };
  const TERMINAL = { ask: "Asked a question", concept: "Quoted the corpus", out_of_scope: "Redirected" };

  // ── boot ───────────────────────────────────────────────────────
  $("#session-note").textContent = `Session ${state.sessionId}.`;
  health();
  autoGrow($("#text-input"));

  async function health() {
    const el = $("#health");
    try {
      const j = await (await fetch(api("/health"))).json();
      el.className = "status ok"; $("span", el).textContent = `${j.counts.chunks} chunks indexed`;
    } catch { el.className = "status bad"; $("span", el).textContent = "backend offline"; }
  }
  function autoGrow(ta) {
    const fit = () => { ta.style.height = "auto"; ta.style.height = Math.min(200, ta.scrollHeight) + "px"; };
    ta.addEventListener("input", fit); fit();
  }

  // ── controls ───────────────────────────────────────────────────
  function clearComposer() {
    $$(".composer-body input, .composer-body textarea").forEach((el) => { el.value = ""; });
    $("#text-input").style.height = "auto";
  }
  $("#mode").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (b.dataset.mode === state.mode) return;
    state.mode = b.dataset.mode;
    $$("#mode button").forEach((x) => x.classList.toggle("on", x === b));
    $$(".composer-body [data-for]").forEach((x) => { x.hidden = x.dataset.for !== state.mode; });
    // Nothing typed in one mode may leak into another: a coordinate left behind
    // would silently attach to a text turn.
    clearComposer();
    const focus = { text: "#text-input", json: "#json-input", coords: "#lat" }[state.mode];
    $(focus).focus();
  });
  $("#starters").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-q]"); if (!b || state.busy) return;
    runTurn({ kind: "chat", message: b.dataset.q });
  });
  $("#text-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.busy) { state.abort?.abort(); return; }
    if (state.mode === "text") {
      const t = $("#text-input").value.trim(); if (!t) return;
      $("#text-input").value = ""; $("#text-input").style.height = "auto";
      runTurn({ kind: "chat", message: t });
    } else if (state.mode === "json") {
      let profile; try { profile = JSON.parse($("#json-input").value); } catch (err) { addError("That JSON does not parse: " + err.message); return; }
      const constraints = $("#json-constraints").value.split(",").map((s) => s.trim()).filter(Boolean);
      clearComposer();
      runTurn({ kind: "analyze", profile, constraints });
    } else {
      const lat = parseFloat($("#lat").value), lon = parseFloat($("#lon").value);
      if (Number.isNaN(lat) || Number.isNaN(lon)) { addError("Give both a latitude and a longitude."); return; }
      const note = $("#coords-note").value.trim();
      clearComposer();
      runTurn({ kind: "chat", message: note || `Site at ${lat}, ${lon}`, profile_patch: { lat, lon } });
    }
  });
  $("#new-chat").addEventListener("click", async () => {
    if (state.busy) return;
    try { await fetch(api(`/session/${state.sessionId}/reset`), { method: "POST" }); } catch { /* fine */ }
    state.sessionId = newId(); localStorage.setItem("darukaa.session", state.sessionId);
    state.turns = []; $("#messages").innerHTML = ""; $("#hero").hidden = false;
    $("#session-note").textContent = `Session ${state.sessionId}.`;
    window.scrollTo({ top: 0 });
  });

  // ── a turn ─────────────────────────────────────────────────────
  function setBusy(on) {
    state.busy = on;
    $("#send").classList.toggle("stop", on);
    $("#send").setAttribute("aria-label", on ? "Stop" : "Send");
    $("#send").title = on ? "Stop this turn" : "";
    $("#new-chat").disabled = on;
  }

  async function runTurn(req) {
    setBusy(true); $("#hero").hidden = true;
    state.abort = new AbortController();

    if (req.kind === "chat") addUser(req.message);
    else addUser(JSON.stringify(req.profile, null, 2), "structured profile → /analyze");

    const msg = addAssistant();
    const trace = {};
    let result = null, ok = false;

    const url = req.kind === "chat" ? "/chat/stream" : "/analyze/stream";
    const payload = req.kind === "chat"
      ? { session_id: state.sessionId, message: req.message, profile_patch: req.profile_patch || null }
      : { profile: req.profile, constraints: req.constraints, audience: $("#audience").value || null };
    if (req.kind === "chat" && $("#audience").value) payload.profile_patch = { ...(payload.profile_patch || {}), audience: $("#audience").value };

    const started = Date.now();
    try {
      await streamSSE(api(url), payload, (ev) => {
        if (ev.type === "node") onNode(msg, trace, ev);
        else if (ev.type === "done") result = ev.result;
        else if (ev.type === "error") throw new Error(`${ev.error}: ${ev.message}`);
      }, state.abort.signal);
      if (result?.trace) Object.assign(trace, result.trace);
      finishThinking(msg, Date.now() - started, trace, result);
      await revealAnswer(msg, result, trace);
      addFooter(msg, result, trace);
      state.turns.push({ turn: result?.turn ?? state.turns.length + 1, message: req.message || req.profile,
                         intent: result?.intent, question: result?.question, answer: result?.answer,
                         confidence: result?.confidence, trace });
      ok = true;
    } catch (err) {
      msg.think.classList.remove("running");
      const stopped = err.name === "AbortError";
      $(".label", msg.think).textContent = stopped
        ? `Stopped after ${((Date.now() - started) / 1000).toFixed(0)}s` : "Something went wrong";
      msg.answer.innerHTML = stopped
        ? `<p class="lead">Stopped. The steps that finished are in the block above; nothing was written up.</p>`
        : `<p class="m-error">${esc(err.message)}</p>`;
    }
    state.abort = null; setBusy(false);
    return ok;
  }

  async function streamSSE(url, payload, onEvent, signal) {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal });
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
    const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = "";
    for (;;) {
      const { value, done } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let i; while ((i = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, i); buf = buf.slice(i + 2);
        for (const line of chunk.split("\n")) if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)));
      }
    }
  }

  // ── messages ───────────────────────────────────────────────────
  function addUser(text, kind) {
    const el = document.createElement("div"); el.className = "m-user";
    el.innerHTML = `<div class="bubble">${kind ? `<span class="kind">${esc(kind)}</span><pre>${esc(text)}</pre>` : esc(text)}</div>`;
    $("#messages").appendChild(el); scrollDown();
  }
  function addError(text) {
    const el = document.createElement("div"); el.className = "m-error"; el.textContent = text;
    $("#messages").appendChild(el); scrollDown();
  }
  function addAssistant() {
    const el = document.createElement("div"); el.className = "m-assistant";
    el.innerHTML = `<div class="avatar">d</div><div class="col">
      <details class="think running" open>
        <summary><i class="spin"></i><span class="label">${NEXT_LABEL.start}</span>
          <svg class="chev" viewBox="0 0 14 14" fill="none"><path d="M3 5l4 4 4-4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg></summary>
        <ol class="steps"></ol>
      </details>
      <div class="answer"><p class="plain cursor"></p></div>
      <div class="foot" hidden></div><div class="details"></div></div>`;
    $("#messages").appendChild(el); scrollDown();
    return { el, think: $(".think", el), steps: $(".steps", el), answer: $(".answer", el), foot: $(".foot", el), details: $(".details", el) };
  }
  function scrollDown() { window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }); }

  // ── thinking block ─────────────────────────────────────────────
  function onNode(msg, trace, ev) {
    const u = ev.update || {};
    if (u.trace) Object.assign(trace, u.trace);
    // A failed check that is about to be retried is not a thought; the retry header
    // and the Verification toggle carry it.
    if (ev.node === "verify" && u.verify_feedback) {
      $(".label", msg.think).textContent = NEXT_LABEL.retrieve;
      return;
    }
    const d = describe(ev.node, u, trace);
    if (ev.node === "adjudicate" && (u.verify_attempts || 1) > 1) {
      const h = document.createElement("li"); h.className = "head"; h.textContent = `retry ${u.verify_attempts - 1}`; msg.steps.appendChild(h);
    }
    const li = document.createElement("li"); if (d.warn) li.classList.add("warn");
    li.innerHTML = `<span class="n">${esc(ev.node)}</span><span class="d">${esc(d.text)}</span><span class="t">${ev.elapsed_ms >= 1000 ? (ev.elapsed_ms / 1000).toFixed(1) + "s" : ev.elapsed_ms + "ms"}</span>`;
    msg.steps.appendChild(li);
    $(".label", msg.think).textContent = TERMINAL[ev.node] || NEXT_LABEL[ev.node] || "Working";
    scrollDown();
  }
  function finishThinking(msg, ms, trace, result) {
    msg.think.classList.remove("running");
    const bits = [];
    const f = trace.diagnosis?.flags?.length; if (f) bits.push(`${f} flag${f === 1 ? "" : "s"}`);
    const e = trace.evidence?.length; if (e) bits.push(`${e} sources`);
    const v = trace.verification; if (v) bits.push(v.degraded ? "degraded" : `${v.attempts} pass${v.attempts === 1 ? "" : "es"}`);
    $(".label", msg.think).textContent = `Reasoned for ${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)}s${bits.length ? " · " + bits.join(" · ") : ""}`;
    msg.think.open = false;
  }
  function describe(node, u, t) {
    switch (node) {
      case "intake": { const ex = t.intake?.extracted || {}, k = Object.keys(ex), rej = t.intake?.rejected || [];
        return { text: k.length ? `read ${k.map((x) => x.replace(/_/g, " ")).join(", ")}` : "nothing new stated", warn: rej.length > 0 }; }
      case "normalize": { const k = Object.keys(t.inferred || {}); return { text: k.length ? `inferred ${k.join(", ")} — marked as inference, a stated value would replace it` : "nothing to infer" }; }
      case "geo_enrich": { const g = t.geo; if (!g) return { text: "no location given" };
        const where = g.place ? g.place.split(",").slice(0, 2).join(",") : `${fmt(g.lat, 4)}, ${fmt(g.lon, 4)}`;
        return { text: g.result === "no data" ? `located ${where} (${g.how}) — soil and climate APIs returned nothing` : `located ${where} (${g.how}) — pulled SoilGrids and NASA POWER`, warn: g.result === "no data" }; }
      case "intent": return { text: `this is ${(u.intent || "new_info").replace("_", " ")}` };
      case "slot_check": { const s = t.slot_check; return { text: u.question ? `${s?.asked?.replace(/_/g, " ")} would change the answer most — asking for it` : "enough is known to proceed" }; }
      case "diagnose": { const d = t.diagnosis || {}, f = d.flags || [], p = (d.patterns || []).map((x) => x.name.replace(/_/g, " "));
        return { text: f.length ? `${f.join(", ")}${p.length ? " · pattern: " + p.join(", ") : ""}` : "no flags raised" }; }
      case "root_cause": return { text: `${(t.root_causes || []).length} causal chains lead to the flagged metrics` };
      case "leverage": { const l = t.leverage || []; return { text: l.length ? `${l.slice(0, 3).map((x) => x.node).join(", ")} reach the most flagged metrics` : "no leverage points" }; }
      case "candidates": { const c = t.candidates || [], e = t.excluded || [], d = c.filter((x) => x.downgraded);
        return { text: `${c.length} practices scored · ${d.length ? d.map((x) => x.practice_id).join(", ") + " downgraded" : "none downgraded"} · ${e.length} excluded` }; }
      case "combine": { const c = (t.combinations || [])[0]; return { text: c ? `${c.combo.join(" + ")}${c.synergy ? " — " + c.synergy : ""}` : "no pairs" }; }
      case "sequence": return { text: (t.sequence || []).map((x) => x.practice_id).join(" → ") || "no sequence" };
      case "plan_queries": return { text: "support, risk and path queries — never the raw message" };
      case "retrieve": { const q = t.queries || [], e = t.evidence || [], w = Math.max(0, ...q.map((x) => x.filter_level));
        return { text: `${q.length} queries → ${e.length} chunks kept${w ? ` (filter relaxed to level ${w})` : ""}`, warn: w > 0 }; }
      case "adjudicate": return { text: `one reasoning call over the dossier and evidence` };
      case "verify": { const fb = u.verify_feedback, v = t.verification;
        if (fb) { const n = (fb.match(/^- /gm) || []).length; const which = [...new Set((fb.match(/\bV\d+\b/g) || []))].join(", "); return { text: `${n} check${n === 1 ? "" : "s"} failed (${which}) — sending feedback and retrying`, warn: true }; }
        if (v?.degraded) return { text: `retries spent — stripped ${v.stripped_numbers || 0} numbers, dropped ${v.dropped_steps || 0} steps rather than show them unverified`, warn: true };
        return { text: "every source exists, every number matches a claim, every step is anchored" }; }
      case "render": return { text: "assembled from verified output only" };
      default: return { text: TERMINAL[node] || "" };
    }
  }

  // ── answer parsing and reveal ──────────────────────────────────
  const HEADERS = ["ANSWER", "SITE", "DIAGNOSIS", "CAUSES", "WHAT CHANGED", "EVIDENCE", "RECOMMENDED", "TRADE-OFFS", "DOWNGRADED", "EXCLUDED", "REASONING", "REVIEW", "CONFIDENCE", "SOURCES", "NOTES"];
  function parse(text) {
    const secs = []; let cur = null;
    for (const line of (text || "").split("\n")) {
      const h = HEADERS.find((x) => line === x || line.startsWith(x + " "));
      if (h) { cur = { h, inline: line.slice(h.length).trim(), body: [] }; secs.push(cur); }
      else if (cur) cur.body.push(line);
    }
    return secs;
  }
  const chipsFromPipe = (s, cls) => s.split(" | ").filter(Boolean).map((p) => {
    const m = p.match(/^(.*?) \((\w+)\)$/);
    return m ? `<span class="chip">${esc(m[1])} <span class="chip src ${esc(m[2])}" style="border:0;padding:0 0 0 4px;background:none">${esc(m[2])}</span></span>` : `<span class="chip ${cls}">${esc(p)}</span>`;
  }).join("");

  function sectionEl(s) {
    const d = document.createElement("div");
    switch (s.h) {
      case "ANSWER": d.className = "lead"; d.textContent = [s.inline, ...s.body].join(" ").trim(); break;
      case "SITE": d.className = "diag"; d.innerHTML = `<span class="lbl">site</span>${chipsFromPipe(s.inline, "")}`; break;
      case "DIAGNOSIS": {
        const [flags, patterns] = s.inline.split(" | patterns: ");
        d.className = "diag";
        d.innerHTML = `<span class="lbl">diagnosis</span>` + flags.split(" | ").map((f) => {
          const m = f.match(/^(\S+) \[(.+)\]$/); return m ? `<span class="chip flag" title="rule ${esc(m[2])}">${esc(m[1])}</span>` : `<span class="chip">${esc(f)}</span>`;
        }).join("") + (patterns ? patterns.split(", ").map((p) => `<span class="chip pattern">${esc(p.replace(/ \[.*\]$/, "").replace(/_/g, " "))}</span>`).join("") : "");
        break; }
      case "RECOMMENDED": {
        d.innerHTML = `<h3>Recommended</h3>`; let card = null;
        for (const raw of s.body) {
          if (raw.trim().startsWith("Order:")) { d.insertAdjacentHTML("beforeend", `<p class="order">${esc(raw.trim())}</p>`); continue; }
          const m = raw.match(/^\[(\d+)\] (.*)$/);
          if (m) { card = document.createElement("div"); card.className = "rec"; card.innerHTML = `<div class="rec-t"><span class="rec-n">${m[1]}</span>${esc(m[2])}</div>`; d.appendChild(card); continue; }
          const r = raw.match(/^\s{4}([A-Za-z ]+?):\s(.*)$/);
          if (r && card) {
            const k = r[1].toLowerCase(); const v = r[2].replace(/\[(S\d+(?:, S\d+)*|no source|P\d+)\]/g, '<span class="cite">[$1]</span>');
            card.insertAdjacentHTML("beforeend", `<div class="row ${k}"><b>${esc(r[1])}</b><span>${v.replace(/&(?!amp;|lt;|gt;)/g, "&amp;")}</span></div>`);
          }
        }
        if (!card) d.innerHTML += `<p class="lead">${esc(s.body.join(" ").trim())}</p>`;
        break; }
      case "DOWNGRADED": case "EXCLUDED": {
        d.className = "list " + (s.h === "DOWNGRADED" ? "down" : "excl"); d.innerHTML = `<h3>${s.h === "DOWNGRADED" ? "Downgraded — the trade-offs" : "Excluded"}</h3>`;
        let p = null;
        for (const raw of s.body) { if (!raw.trim()) continue;
          if (raw.startsWith("        would need:")) { if (p) p.insertAdjacentHTML("beforeend", `<span class="why">${esc(raw.trim())}</span>`); }
          else { p = document.createElement("p"); p.textContent = raw.trim(); d.appendChild(p); } }
        break; }
      case "CAUSES": {
        d.innerHTML = `<h3>Why this is happening</h3>` + s.body.filter((l) => l.trim()).map((l) => {
          const t = l.trim();
          if (t.startsWith("You reported:")) return `<p class="lead"><b>${esc(t)}</b></p>`;
          const m = t.match(/^\((.+?)\) (.*?)(?: \[(.+)\])?$/);
          if (!m) return `<div class="reason">${esc(t)}</div>`;
          const kind = m[1]; const cls = kind.startsWith("observed") ? "obs" : kind.startsWith("possible") ? "maybe" : "";
          return `<div class="reason ${cls}"><span class="k">${esc(kind)}</span>${esc(m[2]).replace(/ -&gt; /g, ' <span class="arr">→</span> ')}${m[3] ? `<span class="a">[${esc(m[3])}]</span>` : ""}</div>`;
        }).join(""); break; }
      case "EVIDENCE": d.className = "sources"; d.innerHTML = `<h3>Evidence this rests on</h3>` + s.body.filter((l) => l.trim()).map((l) => {
        const m = l.trim().match(/^\[(S\d+)\] (.*?): (.*)$/); return m ? `<p><span class="lbl">${m[1]}</span><b>${esc(m[2])}</b> — <span class="snip">${esc(m[3])}</span></p>` : `<p>${esc(l.trim())}</p>`; }).join(""); break;
      case "TRADE-OFFS": {
        d.className = "list down"; d.innerHTML = `<h3>Trade-offs</h3>`; let p = null;
        for (const raw of s.body) { if (!raw.trim()) continue; const t = raw.trim();
          if (raw.startsWith("        would need:")) { if (p) p.insertAdjacentHTML("beforeend", `<span class="why">${esc(t)}</span>`); continue; }
          const m = t.match(/^\((tradeoff|synergy|conflict)\) (.*?)(?: \[(.+)\])?$/);
          if (m) { d.insertAdjacentHTML("beforeend", `<div class="reason"><span class="k">${m[1]}</span>${esc(m[2])}${m[3] ? `<span class="a">[${esc(m[3])}]</span>` : ""}</div>`); p = null; continue; }
          p = document.createElement("p"); p.textContent = t; d.appendChild(p); }
        break; }
      case "REASONING": {
        d.innerHTML = `<h3>Reasoning chain</h3>` + s.body.filter((l) => l.trim()).map((l) => {
          const m = l.trim().match(/^\((\w+)\) (.*?)(?: \[(.+)\])?$/);
          return m ? `<div class="reason"><span class="k">${esc(m[1])}</span>${esc(m[2])}${m[3] ? `<span class="a">[${esc(m[3])}]</span>` : ""}</div>` : `<div class="reason">${esc(l.trim())}</div>`;
        }).join(""); break; }
      case "REVIEW": d.className = "review"; d.textContent = s.inline; break;
      case "CONFIDENCE": return null;
      case "SOURCES": d.className = "sources"; d.innerHTML = `<h3>Sources</h3>` + s.body.filter((l) => l.trim()).map((l) => {
        const m = l.trim().match(/^\[(S\d+)\] (.*)$/); return m ? `<p><span class="lbl">${m[1]}</span>${esc(m[2])}</p>` : `<p>${esc(l.trim())}</p>`; }).join(""); break;
      case "WHAT CHANGED": d.className = "changed"; d.innerHTML = `<h3>What changed</h3>` + s.body.filter((l) => l.trim()).map((l) => `<p>${esc(l.trim())}</p>`).join(""); break;
      case "NOTES": d.className = "notes"; d.innerHTML = s.body.filter((l) => l.trim()).map((l) => `<p>${esc(l.trim())}</p>`).join(""); break;
      default: d.className = "plain"; d.textContent = [s.inline, ...s.body].join("\n").trim();
    }
    return d;
  }

  async function typeInto(el, text, perWord = 14) {
    el.classList.add("cursor"); el.textContent = "";
    for (const w of String(text).split(/(\s+)/)) { el.textContent += w; if (w.trim()) await sleep(perWord); }
    el.classList.remove("cursor");
  }

  function locationEl(g) {
    if (!g || g.lat == null) return null;
    const d = document.createElement("div"); d.className = "loc";
    const via = [g.soil && Object.keys(g.soil).length ? "SoilGrids" : null,
                 g.climate && Object.keys(g.climate).length ? "NASA POWER" : null].filter(Boolean);
    d.innerHTML = `<span class="pin">located</span>` +
      (g.place ? `<span>${esc(g.place)}</span>` : "") +
      `<span class="coords">${fmt(g.lat, 4)}, ${fmt(g.lon, 4)}</span>` +
      `<span class="how">${esc(g.how || "")}</span>` +
      (via.length ? `<span class="via">${via.join(" + ")}</span>` : `<span class="how">no soil or climate data returned</span>`);
    return d;
  }

  async function revealAnswer(msg, result, trace) {
    const box = msg.answer; box.innerHTML = "";
    if (!result) { box.innerHTML = `<p class="m-error">No result returned.</p>`; return; }
    const loc = locationEl(trace.geo); if (loc) { box.appendChild(loc); await sleep(120); }

    if (result.question && !trace.diagnosis) {
      const [q, why] = String(result.answer || result.question).split("\n\nWhy this matters: ");
      const qe = document.createElement("p"); qe.className = "q"; box.appendChild(qe); await typeInto(qe, q, 22);
      if (why) { const w = document.createElement("p"); w.className = "q-why"; box.appendChild(w); await typeInto(w, "Why this matters: " + why, 8); }
      return;
    }
    const secs = parse(result.answer);
    if (!secs.length) { const p = document.createElement("p"); p.className = "plain"; box.appendChild(p); await typeInto(p, result.answer || "", 6); return; }

    for (const s of secs) {
      const el = sectionEl(s); if (!el) continue;
      if (s.h === "RECOMMENDED") {
        const cards = $$(".rec", el); cards.forEach((c) => (c.hidden = true)); box.appendChild(el);
        for (const c of cards) { c.hidden = false; scrollDown(); await sleep(260); }
      } else { box.appendChild(el); scrollDown(); await sleep(s.h === "SITE" || s.h === "DIAGNOSIS" ? 160 : 110); }
    }
  }

  // ── footer: confidence + inline detail toggles ─────────────────
  function addFooter(msg, result, trace) {
    if (!result) return;
    const c = result.confidence;
    const toggles = [
      ["evidence", "Evidence", trace.evidence?.length], ["reasoning", "Reasoning", trace.candidates?.length],
      ["profile", "Profile", trace.profile && Object.keys(trace.profile).length], ["verification", "Verification", null],
      ["graph", "Graph", null],
    ].filter(([k]) => (k === "verification" ? trace.verification : k === "graph" ? (trace.root_causes?.length || trace.candidates?.length) : k === "profile" ? trace.profile : trace[k === "evidence" ? "evidence" : "candidates"]));
    if (!c && !toggles.length) return;
    msg.foot.hidden = false;
    msg.foot.innerHTML = (c ? `<span class="conf"><span>confidence</span><span class="band ${c.band}">${c.band} ${fmt(c.value)}</span><span class="bars" title="evidence · context · agreement · data quality">${
      ["evidence", "context", "agreement", "data_quality"].map((k) => `<i style="--w:${Math.round((c.breakdown?.[k] || 0) * 100)}%" title="${k} ${fmt(c.breakdown?.[k])}"></i>`).join("")}</span></span>` : "") +
      `<span class="spacer"></span>` + toggles.map(([k, l, n]) => `<button class="tgl" data-d="${k}">${l}${n ? `<span class="cnt">${n}</span>` : ""}</button>`).join("");
    msg.foot.addEventListener("click", (e) => {
      const b = e.target.closest(".tgl"); if (!b) return;
      const k = b.dataset.d; b.classList.toggle("on");
      let panel = $(`.detail[data-d="${k}"]`, msg.details);
      if (panel) { panel.remove(); return; }
      panel = document.createElement("div"); panel.className = "detail"; panel.dataset.d = k;
      panel.innerHTML = RENDER[k](trace, result); msg.details.appendChild(panel);
    });
  }

  const kv = (o) => `<table class="tbl"><tbody>${Object.entries(o).map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td class="mono">${esc(typeof v === "number" ? +v.toFixed(3) : v)}</td></tr>`).join("")}</tbody></table>`;

  const RENDER = {
    profile(t) {
      const p = t.profile || {}; let h = `<div class="sub">site profile · value, source, uncertainty</div><table class="tbl"><tbody>${
        Object.entries(p).map(([k, v]) => `<tr><td class="mono">${esc(k)}</td><td class="mono">${esc(typeof v.value === "number" ? +v.value.toFixed(3) : v.value)}</td><td><span class="chip src ${esc(v.source)}">${esc(v.source)}</span></td><td class="num">${v.uncertainty != null ? fmt(v.uncertainty) : ""}</td></tr>`).join("")}</tbody></table>`;
      if (t.inferred && Object.keys(t.inferred).length) h += `<p class="note">Inferred, not stated: ${Object.entries(t.inferred).map(([k, v]) => `${esc(k)} = ${esc(v)}`).join(", ")}.</p>`;
      if (t.intake?.rejected?.length) h += `<p class="note warn">${t.intake.rejected.map(esc).join("<br>")}</p>`;
      if (t.slot_check) h += `<div class="sub">which question to ask · value of information</div><table class="tbl"><thead><tr><th>slot</th><th>flip score</th><th>outcomes</th></tr></thead><tbody>${
        (t.slot_check.ranking || []).map((r) => `<tr class="${r.slot === t.slot_check.asked ? "" : "dim"}"><td class="mono">${esc(r.slot)}</td><td class="num">${fmt(r.flip_score)}</td><td class="num">${r.distinct_outcomes}</td></tr>`).join("")}</tbody></table>`;
      if (t.geo) { h += `<div class="sub">geo enrichment</div>` + kv({ lat: t.geo.lat, lon: t.geo.lon, ...(t.geo.soil || {}), ...(t.geo.climate || {}) }); }
      return h;
    },
    reasoning(t) {
      const d = t.diagnosis || {}; let h = `<div class="sub">rules fired</div><div class="chips">${(d.fired_rules || []).map((r) => `<span class="chip">${esc(r)}</span>`).join("")}</div>`;
      if ((d.patterns || []).length) h += `<div class="sub">compound patterns</div>` + d.patterns.map((p) => `<div class="path"><span class="chip pattern">${esc(p.name)}</span> ${(p.loop || []).map(esc).join(' <span class="arr">→</span> ')}</div>`).join("");
      if ((t.root_causes || []).length) h += `<div class="sub">root causes, upstream</div>` + t.root_causes.slice(0, 6).map((c) => `<div class="path"><span class="w">${esc(c.id)}</span> ${c.path.map(esc).join(' <span class="arr">→</span> ')}</div>`).join("");
      if ((t.leverage || []).length) h += `<div class="sub">leverage</div><table class="tbl"><thead><tr><th>node</th><th>score</th><th>reaches</th><th>on loop</th></tr></thead><tbody>${t.leverage.slice(0, 5).map((l) => `<tr><td class="mono">${esc(l.node)}</td><td class="num">${fmt(l.score)}</td><td class="mono">${(l.reaches || []).map(esc).join(", ")}</td><td>${l.on_loop ? "yes" : ""}</td></tr>`).join("")}</tbody></table>`;
      if ((t.candidates || []).length) h += `<div class="sub">candidates · ranking scores, not effect sizes</div>` + t.candidates.map((c) => { const r = (c.risks_applied || [])[0];
        return `<div class="cand"><div class="h"><b>${esc(c.name || c.practice_id)}</b><span class="score">${fmt(c.suitability)}</span></div><div class="bar ${c.downgraded ? "down" : ""}"><i style="--w:${Math.round(c.suitability * 100)}%"></i></div><div class="m">coverage ${fmt(c.coverage)} · benefit ${c.benefit != null ? (c.benefit >= 0 ? "+" : "") + fmt(c.benefit) : "—"} · penalty ${fmt(c.penalty)}</div>${r ? `<div class="r">↓ ${esc(r.risk)} (−${fmt(r.penalty)})</div>` : ""}</div>`; }).join("");
      if ((t.excluded || []).length) h += `<div class="sub">excluded</div>` + t.excluded.map((e) => `<div class="cand excluded"><div class="h"><b>${esc(e.name || e.practice_id)}</b></div><div class="m">${esc(e.reason)}</div></div>`).join("");
      if ((t.combinations || []).length) h += `<div class="sub">best pairs</div>` + t.combinations.map((c) => `<div class="path">${c.combo.map(esc).join(' <span class="arr">+</span> ')} <span class="w">${fmt(c.score)}</span>${c.synergy ? `<br><span class="w">${esc(c.synergy)}</span>` : ""}</div>`).join("");
      if ((t.sequence || []).length) h += `<div class="sub">sequence</div><table class="tbl"><thead><tr><th>#</th><th>practice</th><th>horizon</th><th>needs first</th></tr></thead><tbody>${t.sequence.map((s) => `<tr><td class="num">${s.order}</td><td>${esc(s.name)}${s.added_as_prerequisite ? ' <span class="chip">prerequisite</span>' : ""}</td><td class="mono">${esc(s.time_horizon)}</td><td class="mono">${(s.requires_first || []).map(esc).join(", ")}</td></tr>`).join("")}</tbody></table>`;
      return h;
    },
    evidence(t) {
      const qs = t.queries || [], ev = t.evidence || [];
      if (t.concept) return `<div class="sub">concept lookup</div><div class="q"><div class="h">${esc(t.concept.query)}</div><div class="m">filter level ${t.concept.filter_level} · kept ${t.concept.kept.length}</div></div>`;
      return `<div class="sub">queries · built from the diagnosis, never the raw message</div>` + qs.map((q) => `<div class="q"><div class="h"><span class="purpose ${esc(q.purpose)}">${esc(q.purpose)}</span><span>${esc(q.q)}</span></div><div class="m">${q.practice_id ? esc(q.practice_id) + " · " : ""}filter level ${q.filter_level} · ${q.n_candidates} candidates · kept ${(q.kept || []).length}</div></div>`).join("") +
        `<div class="sub">evidence block · as cited</div>` + ev.map((e) => `<div class="evi"><div class="h"><span class="lbl">${esc(e.label)}</span><span>${esc(e.doc_title)}${e.page != null ? `, p${e.page}` : ""}</span><span class="meta">${esc(e.evidence_level || "")} · ${fmt(e.score, 3)}</span></div></div>`).join("");
    },
    verification(t) {
      const v = t.verification || {}; const order = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10"];
      const label = { V1: "structure", V2: "sources exist", V3: "numbers ↔ claims", V4: "direction", V5: "climate fit", V6: "mechanism + risk query", V7: "mechanism supported", V8: "no filler", V9: "steps anchored", V10: "≥3 variables" };
      let h = `<div class="sub">checks · ${v.attempts} attempt${v.attempts === 1 ? "" : "s"}${v.degraded ? " · degraded" : ""}</div><div class="chips">${order.map((k) => v.checks?.[k] ? `<span class="chip ${v.checks[k]}">${k} ${esc(label[k])}</span>` : "").join("")}</div>`;
      if (v.degraded) h += `<p class="note warn">Retries exhausted. Removed rather than shown unverified: ${v.stripped_numbers || 0} numbers, ${v.dropped_steps || 0} reasoning steps, ${v.dropped_estimates || 0} estimates, ${v.dropped_recommendations || 0} recommendations.</p>`;
      if ((v.unresolved || []).length) h += `<div class="sub">unresolved</div>` + v.unresolved.map((u) => `<p class="note">${esc(u)}</p>`).join("");
      if (t.confidence) h += `<div class="sub">confidence = evidence × context × agreement × data quality</div><table class="tbl"><tbody>${Object.entries(t.confidence.breakdown || {}).map(([k, x]) => `<tr><td class="mono">${esc(k)}</td><td class="num">${fmt(x)}</td><td><div class="bar"><i style="--w:${Math.round(x * 100)}%"></i></div></td></tr>`).join("")}</tbody></table>`;
      if (t.timing_ms) h += `<div class="sub">timing</div>` + kv(Object.fromEntries(Object.entries(t.timing_ms).map(([k, x]) => [k, `${x} ms`])));
      return h;
    },
    graph(t) {
      const edges = new Map(), depth = new Map(), seen = new Map(), practices = new Set(); let order = 0;
      const touch = (n, d) => { depth.set(n, Math.max(depth.get(n) ?? 0, d)); if (!seen.has(n)) seen.set(n, order++); };
      const add = (path, cls, w) => { path.forEach((n, i) => touch(n, i)); for (let i = 0; i < path.length - 1; i++) { const k = path[i] + "→" + path[i + 1]; if (!edges.has(k)) edges.set(k, { a: path[i], b: path[i + 1], cls, w }); } };
      (t.root_causes || []).slice(0, 6).forEach((c) => add(c.path, "cause", null));
      (t.candidates || []).slice(0, 4).forEach((c) => (c.paths || []).slice(0, 3).forEach((p) => { practices.add(p.path[0]); add(p.path, p.weight < 0 ? "neg" : "pos", p.weight); }));
      if (!edges.size) return `<p class="note">No paths were traversed for this turn.</p>`;
      const cols = new Map(); for (const [n, d] of depth) { if (!cols.has(d)) cols.set(d, []); cols.get(d).push(n); }
      for (const l of cols.values()) l.sort((a, b) => seen.get(a) - seen.get(b));
      const maxD = Math.max(...cols.keys()), rows = Math.max(...Array.from(cols.values(), (l) => l.length));
      const W = 680, BW = 124, BH = 20, GY = 8, PAD = 12, H = PAD * 2 + rows * (BH + GY);
      const colX = (d) => PAD + d * ((W - 2 * PAD - BW) / Math.max(1, maxD)); const pos = new Map();
      for (const [d, l] of cols) l.forEach((n, i) => pos.set(n, { x: colX(d), y: PAD + i * (BH + GY) }));
      const E = Array.from(edges.values()).map((e) => { const a = pos.get(e.a), b = pos.get(e.b); const x1 = a.x + BW, y1 = a.y + BH / 2, x2 = b.x, y2 = b.y + BH / 2, cx = (x1 + x2) / 2;
        return `<path class="e ${e.cls}" d="M${x1},${y1} C${cx},${y1} ${cx},${y2} ${x2},${y2}"><title>${esc(e.a)} → ${esc(e.b)}${e.w != null ? ` (${fmt(e.w, 3)})` : ""}</title></path>`; }).join("");
      const N = Array.from(pos.entries()).map(([n, p]) => `<g class="${practices.has(n) ? "n-practice" : "n-metric"}" transform="translate(${p.x},${p.y})"><rect width="${BW}" height="${BH}" rx="3"/><text x="7" y="13.5">${esc(n.length > 19 ? n.slice(0, 18) + "…" : n)}<title>${esc(n)}</title></text></g>`).join("");
      return `<div class="legend"><span class="pos"><i></i>raises</span><span class="neg"><i></i>lowers</span><span class="cause"><i></i>root-cause chain</span></div><svg class="graph-svg" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">${E}${N}</svg><p class="note">Only edges whose condition holds for this site are drawn. Dashed edges are the trade-offs — why a practice gets downgraded rather than recommended.</p>`;
    },
  };
})();
