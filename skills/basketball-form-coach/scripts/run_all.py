"""Single-agent fallback: run every scripted phase, using local processes for the parallel phases.

Usage: ms run-all <video> [--run-dir DIR] [--target B,D] [--start S --end S] [--device mps|cpu]
Runs prepare, extract, people, segment, analyze, aggregate, dashboard. If 2+ people share the frame and
no --target is given, it stops after `people` so the player can be picked from people.jpg; re-run with
--run-dir and --target (finished steps are skipped).
Visual review (review.json) and coach_notes.md are still the agent's job afterwards.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ms(*args) -> str:
    r = subprocess.run([sys.executable, str(HERE / f"{args[0]}.py"), *args[1:]],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{args[0]} failed:\n{r.stderr[-3000:]}")
    return r.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--run-dir")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--target")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--device")
    a = ap.parse_args()

    if a.run_dir and (Path(a.run_dir) / "manifest.json").exists():
        rd = str(Path(a.run_dir).resolve())
        n = len(json.loads((Path(rd) / "manifest.json").read_text())["chunks"])
    else:
        prep = ["prepare", a.video] + (["--run-dir", a.run_dir] if a.run_dir else [])
        prep += (["--workers", str(a.workers)] if a.workers else []) + (["--start", str(a.start)] if a.start else [])
        prep += ["--end", str(a.end)] if a.end else []
        info = json.loads(ms(*prep))
        rd, n = info["run_dir"], len(info["parallel_tasks"])
    dev = ["--device", a.device] if a.device else []
    with ThreadPoolExecutor(n) as pool:  # extra workers queue on the GPU slot lock
        list(pool.map(lambda i: ms("extract", "--run-dir", rd, "--chunk", str(i), *dev), range(n)))
    if not (Path(rd) / "people.json").exists():
        ms("people", "--run-dir", rd)
    seg = json.loads(ms("segment", "--run-dir", rd, *(["--target", a.target] if a.target else [])))
    if not a.target and any("2+ people" in w for w in seg["warnings"]):
        print(json.dumps({"stopped": "several people in frame: pick the player first",
                          "gallery": f"{rd}/people.jpg",
                          "next": f"ms run-all {a.video} --run-dir {rd} --target <letters>"}, indent=2))
        return
    tasks = [t.split("--events ")[1] for t in seg["parallel_tasks"]]
    with ThreadPoolExecutor(max(1, min(4, len(tasks)))) as pool:
        list(pool.map(lambda ev: ms("analyze", "--run-dir", rd, "--events", ev), tasks))
    print(ms("aggregate", "--run-dir", rd))
    print(ms("dashboard", "--run-dir", rd))
    print(json.dumps({"run_dir": rd, "segment": seg}, indent=2))


if __name__ == "__main__":
    main()
