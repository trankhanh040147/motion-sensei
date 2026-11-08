"""Build dashboard.html: the practice video with clickable event markers, reviews and replays.

Usage: ms dashboard --run-dir DIR
Writes DIR/dashboard.html next to video.mp4 (open it directly in a browser; nothing is uploaded).
Works on partial runs: events without analysis/review are shown as pending. Re-run any time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aggregate import md_to_html
from common import Run, read_json

WORDING = {"jump_shot": "Jump shot", "set_shot": "Set shot", "layup": "Layup", "shot": "Shot", "dribble": "Dribble"}


def event_record(run: Run, e: dict, fps: float, off: float) -> dict:
    d = run.event_dir(e["id"])
    m = read_json(d / "metrics.json") if (d / "metrics.json").exists() else None
    rv = read_json(d / "review.json") if (d / "review.json").exists() else None
    kind = e["type"]
    if kind == "shot":
        kind = (rv or {}).get("shot_kind") or (m or {}).get("info", {}).get("shot_kind") or "shot"
    excluded = bool(rv and (rv.get("detection_ok") is False or rv.get("pose_ok") is False))
    verdicts = {c["key"]: c for c in (rv or {}).get("metric_checks", [])}
    metrics, issues = [], []
    for x in (m or {}).get("metrics", []):
        v = verdicts.get(x["key"], {})
        row = {"key": x["key"], "label": x["label"], "value": x["value"], "unit": x["unit"], "good": x["good"],
               "status": x["status"], "reliable": x["reliable"], "note": x.get("note"), "cue": x.get("cue"),
               "verdict": v.get("verdict"), "why": v.get("why")}
        metrics.append(row)
        if not excluded and x["status"] in ("flag", "warn") and x["reliable"] and v.get("verdict") != "rejected":
            issues.append({"key": x["key"], "label": x["label"], "status": x["status"], "cue": x.get("cue"),
                           "confirmed": v.get("verdict") == "confirmed"})
    if not m:
        state = "pending"
    elif excluded:
        state = "excluded"
    elif not rv:
        state = "unreviewed"
    elif any(i["confirmed"] for i in issues):
        state = "issue"
    elif issues:
        state = "maybe"
    else:
        state = "clean"
    t_main = e.get("release_frame", e["start_frame"])
    return {
        "id": e["id"], "type": e["type"], "kind": kind, "kind_label": WORDING.get(kind, kind),
        "state": state, "start": e["start_frame"] / fps, "end": e["end_frame"] / fps, "t": t_main / fps,
        "offset": off, "detection": e.get("detection"), "confidence": e.get("confidence"),
        "key_frames": {k: v["frame"] / fps for k, v in (m or {}).get("key_frames", {}).items()},
        "info": (m or {}).get("info", {}), "metrics": metrics, "issues": issues,
        "review": rv and {"summary": rv.get("summary"), "detection_ok": rv.get("detection_ok"),
                          "pose_ok": rv.get("pose_ok"), "observations": rv.get("observations", [])},
        "keyframes": f"events/{e['id']}/keyframes.jpg" if (d / "keyframes.jpg").exists() else None,
        "overlay": f"events/{e['id']}/overlay.mp4" if (d / "overlay.mp4").exists() else None,
    }


def build(run: Run) -> dict:
    m = run.load_manifest()
    ev = read_json(run.events)
    fps, off = m["fps"], float(m.get("time_offset_s", 0.0))
    events = [event_record(run, e, fps, off) for e in ev["events"]]
    summary = read_json(run.summary) if run.summary.exists() else None
    top = []
    for typ, t in (summary or {}).get("types", {}).items():
        for i in t.get("top_issues", []):
            top.append({"kind": typ, "kind_label": WORDING.get(typ, typ), "key": i["key"], "label": i["label"],
                        "count": len(i["events"]), "of": t["usable"], "cue": i.get("cue")})
    return {
        "title": Path(m["source"]).name, "duration": m["duration_s"], "fps": fps, "offset": off,
        "view": ev.get("view"), "target": ev.get("target"), "warnings": (summary or ev).get("warnings", []),
        "notes_html": md_to_html(run.coach_notes.read_text()) if run.coach_notes.exists() else None,
        "top_issues": top, "events": events,
    }


PAGE = r"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Practice Review</title>
<style>
:root{--bg:#f6f5f2;--panel:#fff;--fg:#1c1b19;--mut:#6d6a63;--line:#e4e1da;--accent:#c4501b;
--issue:#c0392b;--maybe:#c98a12;--clean:#2e8b57;--excl:#a7a39a;--pend:#5b7fb3;--chip:#efece6;--sel:#fff3ea}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#121211;--panel:#1b1b19;--fg:#ecebe6;--mut:#9d9a92;
--line:#2f2e2a;--chip:#26251f;--sel:#2b1f17;--excl:#5f5c55}}
:root[data-theme=dark]{--bg:#121211;--panel:#1b1b19;--fg:#ecebe6;--mut:#9d9a92;--line:#2f2e2a;--chip:#26251f;--sel:#2b1f17;--excl:#5f5c55}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,"Segoe UI",sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:18px;margin:0}.mut{color:var(--mut)}.counts{display:flex;gap:12px;flex-wrap:wrap;font-size:13px}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px;vertical-align:0}
.s-issue{background:var(--issue)}.s-maybe{background:var(--maybe)}.s-clean{background:var(--clean)}
.s-excluded{background:transparent;border:2px solid var(--excl)}.s-unreviewed,.s-pending{background:transparent;border:2px solid var(--pend)}
#timeline{position:relative;height:46px;margin:12px 20px 0;background:var(--panel);border:1px solid var(--line);border-radius:8px;cursor:pointer}
#timeline .mk{position:absolute;top:9px;width:10px;height:26px;margin-left:-5px;border-radius:3px}
#timeline .mk.sel{outline:2px solid var(--fg);outline-offset:1px}
#timeline .mk.s-excluded{top:15px;height:14px;opacity:.7}
#playhead{position:absolute;top:0;bottom:0;width:2px;background:var(--accent);pointer-events:none}
#ticks{display:flex;justify-content:space-between;margin:2px 20px 0;font-size:11px;color:var(--mut)}
main{display:grid;grid-template-columns:minmax(260px,400px) minmax(0,1fr) 300px;gap:16px;padding:14px 20px 24px;align-items:start}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px}
#player{padding:10px;position:sticky;top:10px}
video{width:100%;max-height:72vh;background:#000;border-radius:6px;display:block}
.ctrl{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;align-items:center}
button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--chip);color:var(--fg);border-radius:6px;padding:4px 9px;cursor:pointer}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}button:disabled{opacity:.4;cursor:default}
#clock{font-variant-numeric:tabular-nums;margin-left:auto;font-size:13px}
#detail{padding:16px;min-height:300px}#detail h2{margin:0 0 4px;font-size:18px}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}
.chip{background:var(--chip);border-radius:999px;padding:3px 10px;font-size:12px;cursor:pointer;border:1px solid transparent}
.chip:hover{border-color:var(--accent)}
.kf{width:100%;border-radius:6px;margin:6px 0;display:block}
.box{border-left:3px solid var(--line);padding:4px 10px;margin:8px 0}.box.issue{border-color:var(--issue)}.box.maybe{border-color:var(--maybe)}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:5px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--mut);font-size:12px}
.st-flag{color:var(--issue);font-weight:600}.st-warn{color:var(--maybe);font-weight:600}.st-good{color:var(--clean)}.st-na{color:var(--mut)}
.vd{font-size:11px;padding:1px 6px;border-radius:4px;background:var(--chip);white-space:nowrap}
.vd-confirmed{background:var(--issue);color:#fff}.vd-rejected{text-decoration:line-through}
details summary{cursor:pointer;color:var(--mut);margin:8px 0}
#list{max-height:calc(100vh - 140px);overflow:auto;position:sticky;top:10px}
.filters{display:flex;gap:4px;flex-wrap:wrap;padding:10px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--panel)}
.row{display:flex;gap:8px;padding:8px 10px;border-bottom:1px solid var(--line);cursor:pointer;align-items:flex-start}
.row:hover{background:var(--chip)}.row.sel{background:var(--sel)}
.row .t{font-variant-numeric:tabular-nums;color:var(--mut);font-size:12px;width:52px;flex:none}
.row .what{flex:1;min-width:0}.row .sub{font-size:12px;color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.notes h2,.notes h3{font-size:15px;margin:14px 0 4px}.notes p,.notes li{margin:4px 0}
.warn{background:var(--chip);border-radius:6px;padding:8px 10px;margin:8px 0;font-size:13px}
kbd{font:11px ui-monospace,monospace;border:1px solid var(--line);border-radius:4px;padding:0 4px}
#ovwrap{display:none;margin-top:8px}#ovwrap.show{display:block}
@media (max-width:1100px){main{grid-template-columns:minmax(240px,360px) minmax(0,1fr)}#list{grid-column:1/-1;position:static;max-height:none}}
@media (max-width:700px){main{grid-template-columns:1fr;padding:12px 16px}#player{position:static}header,#timeline,#ticks{margin-left:16px;margin-right:16px;padding-left:0;padding-right:0}header{padding:12px 0}}
</style></head><body>
<header><h1>Practice Review</h1><span class="mut" id="meta"></span><div class="counts" id="counts"></div></header>
<div id="timeline" title="Click a marker to jump to that event"><div id="playhead"></div></div>
<div id="ticks"></div>
<main>
 <section class="panel" id="player">
  <video id="v" src="video.mp4" preload="metadata" playsinline></video>
  <div class="ctrl">
   <button id="bPlay" title="Space">Play</button>
   <button id="bPrev" title="J / ←">Prev</button><button id="bNext" title="K / →">Next</button>
   <button id="bLoop" title="L: loop the selected event">Loop</button>
   <span id="clock">0:00.0</span>
  </div>
  <div class="ctrl" id="speeds"></div>
  <div class="ctrl mut" style="font-size:12px"><kbd>,</kbd><kbd>.</kbd> frame step · <kbd>1</kbd>–<kbd>4</kbd> speed</div>
  <div id="ovwrap"><div class="mut" style="font-size:12px;margin:6px 0 4px">Skeleton replay</div><video id="ov" muted loop playsinline></video></div>
 </section>
 <section class="panel" id="detail"></section>
 <aside class="panel" id="list"><div class="filters" id="filters"></div><div id="rows"></div></aside>
</main>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s), v = $("#v"), ov = $("#ov");
const STATE = {issue:"Issue confirmed", maybe:"Possible issue", clean:"Looks good", excluded:"Not a valid shot", unreviewed:"Not reviewed yet", pending:"Not analyzed yet"};
let sel = null, loop = false, filter = "all";
const fmt = s => { s += D.offset; const m = Math.floor(s / 60); return m + ":" + (s - m * 60).toFixed(1).padStart(4, "0"); };
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const vis = () => D.events.filter(e => filter === "all" || e.state === filter || (filter.startsWith("issue:") && e.issues.some(i => i.key === filter.slice(6))));

$("#meta").textContent = `${D.title} · ${fmt(D.duration - D.offset)} · camera: ${D.view || "?"}` + (D.target ? ` · player: groups ${D.target}` : "");
const counts = {}; D.events.forEach(e => counts[e.state] = (counts[e.state] || 0) + 1);
$("#counts").innerHTML = Object.keys(STATE).filter(k => counts[k]).map(k => `<span><span class="dot s-${k}"></span>${STATE[k]}: ${counts[k]}</span>`).join("");

function drawTimeline() {
  const tl = $("#timeline"); tl.querySelectorAll(".mk").forEach(n => n.remove());
  D.events.forEach(e => {
    const m = document.createElement("div");
    m.className = `mk s-${e.state}` + (sel === e ? " sel" : "");
    m.style.left = (100 * e.t / D.duration) + "%";
    m.title = `${e.id} · ${e.kind_label} · ${fmt(e.t)} · ${STATE[e.state]}`;
    m.onclick = ev => { ev.stopPropagation(); select(e, true); };
    tl.appendChild(m);
  });
  const n = innerWidth < 700 ? 4 : 8; $("#ticks").innerHTML = Array.from({length: n + 1}, (_, i) => `<span>${fmt(D.duration * i / n)}</span>`).join("");
}
$("#timeline").onclick = ev => { const r = ev.currentTarget.getBoundingClientRect(); v.currentTime = D.duration * (ev.clientX - r.left) / r.width; };

function drawFilters() {
  const opts = [["all", "All"], ["issue", "Issues"], ["maybe", "Possible"], ["clean", "Good"], ["excluded", "Invalid"], ["unreviewed", "Unreviewed"]]
    .filter(([k]) => k === "all" || counts[k]);
  $("#filters").innerHTML = opts.map(([k, l]) => `<button data-f="${k}" class="${filter === k ? "on" : ""}">${l}</button>`).join("")
    + (filter.startsWith("issue:") ? `<button class="on" data-f="all">× ${esc(filter.slice(6))}</button>` : "");
  $("#filters").querySelectorAll("button").forEach(b => b.onclick = () => { filter = b.dataset.f; drawFilters(); drawRows(); });
}
function drawRows() {
  $("#rows").innerHTML = vis().map(e => `<div class="row ${sel === e ? "sel" : ""}" data-id="${e.id}">
    <span class="t">${fmt(e.t)}</span><span class="dot s-${e.state}" style="margin-top:5px"></span>
    <span class="what"><b>${esc(e.kind_label)}</b> <span class="mut">${e.id}</span>
    <div class="sub">${esc(e.issues.length ? e.issues.map(i => i.label).join(" · ") : (e.review?.summary || STATE[e.state]))}</div></span></div>`).join("")
    || `<div class="row mut">No events</div>`;
  $("#rows").querySelectorAll(".row[data-id]").forEach(r => r.onclick = () => select(D.events.find(e => e.id === r.dataset.id), true));
}

function summaryView() {
  const w = D.warnings.map(x => `<div class="warn">${esc(x)}</div>`).join("");
  const top = D.top_issues.length ? `<h3>Recurring issues</h3><table><tr><th>Issue</th><th>Type</th><th>Seen</th><th>Cue</th></tr>${
    D.top_issues.map(i => `<tr class="chipline" data-k="${i.key}" style="cursor:pointer"><td>${esc(i.label)}</td><td>${esc(i.kind_label)}</td><td>${i.count}/${i.of}</td><td>${esc(i.cue || "")}</td></tr>`).join("")}</table>
    <p class="mut" style="font-size:12px">Click an issue to list the shots that show it.</p>` : "";
  $("#detail").innerHTML = `<h2>Session summary</h2><p class="mut">Click a marker on the timeline or a row on the right to jump to that moment.</p>
    ${w}${D.notes_html ? `<div class="notes">${D.notes_html}</div>` : `<p class="mut">Coach notes are not written yet.</p>`}${top}`;
  $("#detail").querySelectorAll("tr[data-k]").forEach(t => t.onclick = () => { filter = "issue:" + t.dataset.k; drawFilters(); drawRows(); const f = vis()[0]; if (f) select(f, true); });
}

function detailView(e) {
  const kf = Object.entries(e.key_frames).map(([k, t]) => `<span class="chip" data-t="${t}">${k.replace(/_/g, " ")} · ${fmt(t)}</span>`).join("");
  const iss = e.issues.map(i => `<div class="box ${i.confirmed ? "issue" : "maybe"}"><b>${esc(i.label)}</b> ${i.confirmed ? '<span class="vd vd-confirmed">confirmed</span>' : '<span class="vd">not confirmed</span>'}<div>${esc(i.cue || "")}</div></div>`).join("");
  const obs = (e.review?.observations || []).map(o => `<li>${esc(o.text)} <span class="mut">(${esc(o.frame || "")}${o.severity ? ", " + esc(o.severity) : ""})</span></li>`).join("");
  const rows = e.metrics.map(x => {
    const st = x.status === "n/a" ? "na" : x.status;
    const val = x.value === null ? "–" : `${x.value} ${x.unit}`;
    const vd = x.verdict ? `<span class="vd vd-${x.verdict}" title="${esc(x.why || "")}">${x.verdict}</span>` : "";
    return `<tr><td>${esc(x.label)}${x.note ? `<div class="mut" style="font-size:11px">${esc(x.note)}</div>` : ""}</td><td>${val}</td><td class="mut">${x.good[0]}–${x.good[1]}</td><td class="st-${st}">${x.status}${x.reliable ? "" : " ≈"}</td><td>${vd}</td></tr>`;
  }).join("");
  const why = e.metrics.filter(x => x.why).map(x => `<li><b>${esc(x.label)}</b> (${x.verdict}): ${esc(x.why)}</li>`).join("");
  $("#detail").innerHTML = `<div class="mut" style="font-size:12px"><a href="#" id="toSummary">← Session summary</a></div>
    <h2>${esc(e.kind_label)} · ${fmt(e.t)}</h2>
    <div class="mut"><span class="dot s-${e.state}"></span>${STATE[e.state]} · ${e.id}${e.info.shooting_hand ? " · " + e.info.shooting_hand + " hand" : ""} · detected by ${esc(e.detection)}</div>
    <div class="chips">${kf}</div>
    ${e.review?.summary ? `<p><b>Review:</b> ${esc(e.review.summary)}</p>` : ""}
    ${e.state === "excluded" ? `<div class="warn">Reviewer marked this as ${e.review.detection_ok === false ? "not a real shot" : "skeleton on the wrong person / badly off"}; it is left out of the summary.</div>` : ""}
    ${iss}${obs ? `<h3 style="font-size:14px">What the reviewer saw</h3><ul>${obs}</ul>` : ""}
    ${e.keyframes ? `<img class="kf" src="${e.keyframes}" alt="${e.id} key frames">` : ""}
    ${rows ? `<details ${e.issues.length ? "" : "open"}><summary>All measurements</summary><table><tr><th>Metric</th><th>Value</th><th>Target</th><th></th><th>Review</th></tr>${rows}</table>
      ${why ? `<ul class="mut" style="font-size:12px">${why}</ul>` : ""}<p class="mut" style="font-size:12px">≈ = approximate for this camera angle or release frame</p></details>` : ""}`;
  $("#toSummary").onclick = ev => { ev.preventDefault(); sel = null; loop = false; refresh(); summaryView(); };
  $("#detail").querySelectorAll(".chip[data-t]").forEach(c => c.onclick = () => { v.pause(); v.currentTime = +c.dataset.t; });
}

function refresh() {
  drawTimeline(); drawRows();
  $("#bLoop").classList.toggle("on", loop);
  const w = $("#ovwrap");
  if (sel && sel.overlay) { if (!ov.src.endsWith(sel.overlay)) { ov.src = sel.overlay; } w.classList.add("show"); ov.play().catch(() => {}); }
  else { w.classList.remove("show"); ov.pause(); }
}
function select(e, play) {
  sel = e; detailView(e); refresh();
  v.currentTime = Math.max(0, e.start);
  if (play) v.play().catch(() => {});
  const row = document.querySelector(`.row[data-id="${e.id}"]`), box = $("#list");
  if (row && box.scrollHeight > box.clientHeight + 4 && getComputedStyle(box).position === "sticky")
    box.scrollTop = row.offsetTop - box.clientHeight / 2;
}
function step(dir) {
  const list = vis(); if (!list.length) return;
  let i = sel ? list.indexOf(sel) : -1;
  if (i < 0) { const now = v.currentTime; i = dir > 0 ? list.findIndex(e => e.t > now) - 1 : list.findIndex(e => e.t >= now); }
  select(list[Math.min(list.length - 1, Math.max(0, i + dir))], true);
}

$("#bPlay").onclick = () => v.paused ? v.play() : v.pause();
$("#bPrev").onclick = () => step(-1); $("#bNext").onclick = () => step(1);
$("#bLoop").onclick = () => { loop = !loop && !!sel; refresh(); };
const SPEEDS = [0.25, 0.5, 1, 2];
$("#speeds").innerHTML = SPEEDS.map(s => `<button data-s="${s}" class="${s === 1 ? "on" : ""}">${s}×</button>`).join("");
const setSpeed = s => { v.playbackRate = ov.playbackRate = s; $("#speeds").querySelectorAll("button").forEach(b => b.classList.toggle("on", +b.dataset.s === s)); };
$("#speeds").querySelectorAll("button").forEach(b => b.onclick = () => setSpeed(+b.dataset.s));
v.onplay = v.onpause = () => $("#bPlay").textContent = v.paused ? "Play" : "Pause";
v.ontimeupdate = () => {
  $("#clock").textContent = fmt(v.currentTime);
  $("#playhead").style.left = (100 * v.currentTime / D.duration) + "%";
  if (loop && sel && v.currentTime >= sel.end) v.currentTime = sel.start;
};
document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT") return;
  const k = ev.key;
  if (k === " ") { ev.preventDefault(); $("#bPlay").click(); }
  else if (k === "ArrowRight" || k === "k") step(1);
  else if (k === "ArrowLeft" || k === "j") step(-1);
  else if (k === "l") $("#bLoop").click();
  else if (k === "," || k === ".") { v.pause(); v.currentTime += (k === "." ? 1 : -1) / D.fps; }
  else if ("1234".includes(k) && k.length === 1) setSpeed(SPEEDS[+k - 1]);
});

drawFilters(); drawTimeline(); drawRows(); summaryView();
const first = location.hash.slice(1) && D.events.find(e => e.id === location.hash.slice(1));
if (first) select(first, false);
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    run = Run(ap.parse_args().run_dir)
    data = build(run)
    out = run.dir / "dashboard.html"
    out.write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    states = {}
    for e in data["events"]:
        states[e["state"]] = states.get(e["state"], 0) + 1
    print(json.dumps({"dashboard": str(out), "events": len(data["events"]), "states": states}, indent=2))


if __name__ == "__main__":
    main()
