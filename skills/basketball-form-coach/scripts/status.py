"""Show pipeline state and the next runnable tasks (what can be dispatched in parallel now).

Usage: ms status --run-dir DIR
"""
from __future__ import annotations

import argparse
import json

from common import Run, read_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    run = Run(ap.parse_args().run_dir)
    if not run.manifest.exists():
        print(json.dumps({"phase": "prepare", "next": ["ms prepare <video> --run-dir " + str(run.dir)]}))
        return
    m = run.load_manifest()
    todo = [c["id"] for c in m["chunks"] if not (run.chunk_dir(c["id"]) / "DONE.json").exists()]
    if todo:
        prog = {}
        for i in todo:
            pf, tf = run.chunk_dir(i) / "PROGRESS.json", run.chunk_dir(i) / "frames.jsonl.tmp"
            if pf.exists():
                prog[i] = read_json(pf)
            elif tf.exists():
                prog[i] = {"done": sum(1 for _ in open(tf)), "running": False}
        print(json.dumps({"phase": "extract", "pending_chunks": todo, "parallel": True,
                          "max_concurrent": m.get("max_concurrent_extract"), "progress": prog,
                          "note": "re-running a pending chunk resumes from its partial file",
                          "next": [f"ms extract --run-dir {run.dir} --chunk {i}" for i in todo]}, indent=2))
        return
    if not run.events.exists():
        print(json.dumps({"phase": "segment", "next": [f"ms segment --run-dir {run.dir}"]}))
        return
    ev = read_json(run.events)
    pending, review = [], []
    for t in ev["analyst_tasks"]:
        if any(not (run.event_dir(e) / "DONE.json").exists() for e in t["events"]):
            pending.append(t["command"])
        review += [e for e in t["events"] if not (run.event_dir(e) / "review.json").exists()]
    if pending:
        print(json.dumps({"phase": "analyze", "parallel": True, "next": pending,
                          "then": "analyst writes events/<id>/review.json"}, indent=2))
        return
    if review:
        print(json.dumps({"phase": "review", "parallel": True, "events_missing_review": review}, indent=2))
        return
    nxt = [f"ms aggregate --run-dir {run.dir}"]
    if not run.coach_notes.exists():
        nxt.append(f"write {run.coach_notes} then re-run aggregate")
    print(json.dumps({"phase": "aggregate" if not run.summary.exists() else "report",
                      "next": nxt, "report": str(run.report_html) if run.report_html.exists() else None}, indent=2))


if __name__ == "__main__":
    main()
