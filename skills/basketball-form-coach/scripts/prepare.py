"""Phase 0 (lead): normalize the video and write the chunk plan for parallel extraction.

Usage: ms prepare <video> [--run-dir DIR] [--workers N] [--start S] [--end S] [--max-height 720] [--max-fps 60]
--workers defaults to how many extract processes this machine's GPU/RAM can run at once.
--start/--end (seconds) analyze only part of a long video; reported times stay in source-video time.
Prints the run dir and the parallel extract tasks.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path

from common import REPO_ROOT, Run, gpu_slots, pick_device, write_json


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets,duration:stream_side_data=rotation:format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    s = info["streams"][0]
    num, den = (s.get("avg_frame_rate") or s["r_frame_rate"]).split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    dur = float(s.get("duration") or info["format"]["duration"])
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        rot = int(sd.get("rotation", 0) or 0)
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": fps, "duration_s": dur,
            "frames": int(s.get("nb_read_packets") or round(dur * fps)), "rotation": rot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--run-dir")
    ap.add_argument("--workers", type=int, help="extract chunks; default = accelerator slots on this machine")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the source video")
    ap.add_argument("--end", type=float, help="seconds into the source video")
    ap.add_argument("--max-height", type=int, default=720)
    ap.add_argument("--max-fps", type=float, default=60.0)
    ap.add_argument("--overlap-s", type=float, default=1.0)
    args = ap.parse_args()

    src = Path(args.video).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"video not found: {src}")
    run = Run(args.run_dir or REPO_ROOT / "runs" / f"{src.stem}-{time.strftime('%Y%m%d-%H%M%S')}")
    run.dir.mkdir(parents=True, exist_ok=True)

    src_info = probe(src)
    slots = gpu_slots(pick_device())
    workers = args.workers or slots
    trim = ["-ss", f"{args.start:.3f}"] if args.start > 0 else []
    if args.end:
        trim += ["-to", f"{args.end:.3f}"]
    fps = min(src_info["fps"], args.max_fps)
    # ffmpeg auto-rotates; force CFR + 1s GOP so chunk workers can seek precisely.
    vf = f"scale=-2:'min({args.max_height},ih)',fps={fps:.3f}"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *trim, "-i", str(src), "-an", "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-g", str(max(1, round(fps))),
         "-pix_fmt", "yuv420p", str(run.video)],
        check=True,
    )
    norm = probe(run.video)
    total = norm["frames"]
    n = max(1, min(workers, math.ceil(norm["duration_s"] / 5)))  # >=5s per chunk
    core = math.ceil(total / n)
    ov = round(args.overlap_s * norm["fps"])
    chunks = []
    for i in range(n):
        cs, ce = i * core, min(total, (i + 1) * core)
        chunks.append({"id": i, "core_start": cs, "core_end": ce,
                       "start_frame": max(0, cs - ov), "end_frame": min(total, ce + ov)})

    manifest = {
        "schema": "motion-sensei/manifest@1",
        "sport": "basketball",
        "source": str(src),
        "source_info": src_info,
        "video": str(run.video),
        "fps": norm["fps"], "width": norm["width"], "height": norm["height"],
        "frames": total, "duration_s": norm["duration_s"],
        "time_offset_s": args.start, "max_concurrent_extract": slots,
        "chunks": chunks,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_json(run.manifest, manifest)
    print(json.dumps({
        "run_dir": str(run.dir),
        "duration_s": round(norm["duration_s"], 2), "fps": round(norm["fps"], 2),
        "resolution": f"{norm['width']}x{norm['height']}",
        "max_concurrent_extract": slots,
        "note": "extract workers beyond max_concurrent_extract wait for a free GPU slot; they are safe to launch but add no speed",
        "parallel_tasks": [f"ms extract --run-dir {run.dir} --chunk {c['id']}" for c in chunks],
        "next_after_all_done": f"ms segment --run-dir {run.dir}",
    }, indent=2))


if __name__ == "__main__":
    main()
