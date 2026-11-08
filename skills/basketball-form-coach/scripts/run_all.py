"""Single-agent fallback: run every scripted phase, using local processes for the parallel phases.

Usage: ms run-all <video> [--run-dir DIR] [--workers N] [--device mps|cpu]
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
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--device")
    a = ap.parse_args()

    prep = ["prepare", a.video, "--workers", str(a.workers)] + (["--run-dir", a.run_dir] if a.run_dir else [])
    info = json.loads(ms(*prep))
    rd = info["run_dir"]
    dev = ["--device", a.device] if a.device else []
    n = len(info["parallel_tasks"])
    with ThreadPoolExecutor(n) as pool:
        list(pool.map(lambda i: ms("extract", "--run-dir", rd, "--chunk", str(i), *dev), range(n)))
    seg = json.loads(ms("segment", "--run-dir", rd))
    tasks = [t.split("--events ")[1] for t in seg["parallel_tasks"]]
    with ThreadPoolExecutor(max(1, min(4, len(tasks)))) as pool:
        list(pool.map(lambda ev: ms("analyze", "--run-dir", rd, "--events", ev), tasks))
    print(ms("aggregate", "--run-dir", rd))
    print(json.dumps({"run_dir": rd, "segment": seg}, indent=2))


if __name__ == "__main__":
    main()
