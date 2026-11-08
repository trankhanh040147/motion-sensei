"""Phase 4 (lead): merge per-event metrics + analyst reviews into summary.json and report.html.

Usage: ms aggregate --run-dir DIR
Re-run after writing coach_notes.md to embed the final coaching notes in report.html.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze import LAYUP_SKIP
from common import Run, clean, read_json, write_json

SEV = {"flag": 2, "warn": 1}


def load_event(run: Run, eid: str):
    d = run.event_dir(eid)
    if not (d / "DONE.json").exists():
        return None, None
    m = read_json(d / "metrics.json")
    rv = read_json(d / "review.json") if (d / "review.json").exists() else None
    return m, rv


def rejected_keys(rv) -> set:
    if not rv:
        return set()
    if rv.get("pose_ok") is False or rv.get("detection_ok") is False:
        return {"*"}
    return {c["key"] for c in rv.get("metric_checks", []) if c.get("verdict") == "rejected"}


def summarize(run: Run) -> dict:
    ev_doc = read_json(run.events)
    per_type = defaultdict(list)
    missing, unreviewed, excluded = [], [], []
    for e in ev_doc["events"]:
        m, rv = load_event(run, e["id"])
        if m is None:
            missing.append(e["id"])
            continue
        if rv is None:
            unreviewed.append(e["id"])
        rej = rejected_keys(rv)
        if "*" in rej:
            excluded.append(e["id"])
        typ = e["type"]
        if typ == "shot":
            typ = (rv or {}).get("shot_kind") or m["info"].get("shot_kind", "jump_shot")
            if typ == "layup":
                rej = rej | LAYUP_SKIP
        per_type[typ].append((e, m, rv, rej))

    out = {"schema": "motion-sensei/summary@1", "view": ev_doc["view"], "warnings": list(ev_doc["warnings"]),
           "missing_analysis": missing, "unreviewed": unreviewed, "excluded_by_review": excluded, "types": {}}
    if missing:
        out["warnings"].append(f"events not analyzed: {missing}")
    if unreviewed:
        out["warnings"].append(f"events without visual review: {unreviewed}")

    for typ, rows in per_type.items():
        stats, issues = {}, defaultdict(lambda: {"score": 0, "events": [], "cue": None, "label": None})
        usable = [r for r in rows if "*" not in r[3]]
        keys = [mm["key"] for mm in rows[0][1]["metrics"]] if rows else []
        for k in keys:
            vals, statuses = [], []
            for e, m, rv, rej in usable:
                mm = next(x for x in m["metrics"] if x["key"] == k)
                if k in rej or mm["value"] is None:
                    continue
                vals.append(mm["value"])
                statuses.append(mm["status"])
                if mm["status"] in SEV and mm["reliable"]:
                    it = issues[k]
                    it["score"] += SEV[mm["status"]]
                    it["events"].append(e["id"])
                    it["cue"] = it["cue"] or mm.get("cue")
                    it["label"] = mm["label"]
            if vals:
                a = np.array(vals, float)
                stats[k] = {"label": next(x for x in rows[0][1]["metrics"] if x["key"] == k)["label"],
                            "n": len(vals), "mean": clean(a.mean()), "std": clean(a.std()),
                            "min": clean(a.min()), "max": clean(a.max()),
                            "good_rate": round(statuses.count("good") / len(statuses), 2)}
        obs = []
        for e, m, rv, rej in usable:
            for o in (rv or {}).get("observations", []):
                obs.append({"event": e["id"], **o})
        ranked = sorted(({"key": k, **v, "share": round(len(v["events"]) / max(len(usable), 1), 2)}
                         for k, v in issues.items()), key=lambda x: -x["score"])
        out["types"][typ] = {"count": len(rows), "usable": len(usable), "stats": stats,
                             "top_issues": ranked[:5], "visual_observations": obs,
                             "events": [e["id"] for e, *_ in rows]}
        if typ in ("jump_shot", "set_shot", "layup"):
            info = [m["info"] for _, m, _, _ in usable]
            out["types"][typ]["shooting_hand"] = max(set(i["shooting_hand"] for i in info),
                                                     key=[i["shooting_hand"] for i in info].count) if info else None
            cons = {}
            for k in ("release_height_torso", "elbow_set_deg", "dip_to_release_s", "release_vs_apex_s"):
                s = stats.get(k)
                if s and s["n"] >= 3 and s["mean"]:
                    cons[k] = {"std": s["std"], "cv": clean(abs(s["std"] / s["mean"]))}
            out["types"][typ]["consistency"] = cons
    return out


def md_to_html(md: str) -> str:
    lines, outp, in_ul = md.splitlines(), [], False
    for ln in lines:
        t = html.escape(ln.rstrip())
        t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
        if ln.startswith(("- ", "* ")):
            if not in_ul:
                outp.append("<ul>")
                in_ul = True
            outp.append(f"<li>{t[2:]}</li>")
            continue
        if in_ul:
            outp.append("</ul>")
            in_ul = False
        if m := re.match(r"^(#{1,3}) (.*)", t):
            n = len(m.group(1)) + 1
            outp.append(f"<h{n}>{m.group(2)}</h{n}>")
        elif t:
            outp.append(f"<p>{t}</p>")
    if in_ul:
        outp.append("</ul>")
    return "\n".join(outp)


CSS = """
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dd;--good:#1f8a4c;--warn:#b7791f;--flag:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe6;--mut:#a3a29c;--card:#1e1e1c;--line:#33332f}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:24px 16px}
main{max-width:1100px;margin:auto}h1{margin:0 0 4px}.mut{color:var(--mut)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}
.good{color:var(--good)}.warn{color:var(--warn)}.flag{color:var(--flag);font-weight:600}.n\\/a{color:var(--mut)}
img,video{max-width:100%;border-radius:6px}.wrap{overflow-x:auto}
"""


def render_html(run: Run, s: dict) -> str:
    P = []
    m = run.load_manifest()
    P.append(f"<h1>Basketball form report</h1><p class=mut>{html.escape(Path(m['source']).name)} · "
             f"{m['duration_s']:.1f}s · camera view: <b>{s['view']}</b></p>")
    if s["warnings"]:
        P.append("<div class=card><b>Data quality</b><ul>" + "".join(f"<li>{html.escape(w)}</li>" for w in s["warnings"]) + "</ul></div>")
    if run.coach_notes.exists():
        P.append("<div class=card>" + md_to_html(run.coach_notes.read_text()) + "</div>")
    for typ, t in s["types"].items():
        P.append(f"<h2>{typ.replace('_', ' ').title()}s: {t['count']} detected ({t['usable']} usable)</h2>")
        if t["top_issues"]:
            P.append("<div class=card><b>Recurring issues (from measurements)</b><table><tr><th>Issue</th><th>Seen in</th><th>Cue</th></tr>" +
                     "".join(f"<tr><td>{html.escape(i['label'])}</td><td>{len(i['events'])}/{t['usable']}</td>"
                             f"<td>{html.escape(i['cue'] or '')}</td></tr>" for i in t["top_issues"]) + "</table></div>")
        P.append("<div class='card wrap'><b>Averages</b><table><tr><th>Metric</th><th>Mean</th><th>Range</th><th>In target</th></tr>" +
                 "".join(f"<tr><td>{html.escape(v['label'])}</td><td>{v['mean']}</td><td>{v['min']} – {v['max']}</td>"
                         f"<td>{int(v['good_rate'] * 100)}%</td></tr>" for v in t["stats"].values()) + "</table></div>")
        for eid in t["events"]:
            d = run.event_dir(eid)
            mt = read_json(d / "metrics.json")
            rv = read_json(d / "review.json") if (d / "review.json").exists() else None
            P.append(f"<div class=card><h3>{eid} <span class=mut>at {mt['key_frames'][next(iter(mt['key_frames']))]['t_s']}s</span></h3>")
            if (d / "keyframes.jpg").exists():
                P.append(f"<div class=wrap><img src='events/{eid}/keyframes.jpg' alt='{eid} key frames'></div>")
            if rv and rv.get("summary"):
                P.append(f"<p><b>Review:</b> {html.escape(rv['summary'])}</p>")
            P.append("<div class=wrap><table><tr><th>Metric</th><th>Value</th><th>Target</th><th></th></tr>" +
                     "".join(f"<tr><td>{html.escape(x['label'])}</td><td>{x['value'] if x['value'] is not None else '–'} {x['unit']}</td>"
                             f"<td>{x['good'][0]} – {x['good'][1]}</td><td class='{x['status']}'>{x['status']}"
                             f"{'' if x['reliable'] else ' (approx.)'}</td></tr>" for x in mt["metrics"]) + "</table></div>")
            if (d / "overlay.mp4").exists():
                P.append(f"<details><summary>Skeleton clip</summary><video controls muted src='events/{eid}/overlay.mp4'></video></details>")
            P.append("</div>")
    return (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Basketball Form Report</title><style>{CSS}</style></head><body><main>{''.join(P)}</main></body></html>")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Run(args.run_dir)
    s = summarize(run)
    write_json(run.summary, s)
    run.report_html.write_text(render_html(run, s))
    brief = {t: {"count": v["count"], "top_issues": [(i["label"], f"{len(i['events'])}/{v['usable']}") for i in v["top_issues"]]}
             for t, v in s["types"].items()}
    print(json.dumps({"summary": str(run.summary), "report": str(run.report_html), "warnings": s["warnings"], **brief},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
