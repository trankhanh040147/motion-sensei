"""Phase 1 (parallel, one worker per chunk): pose + ball detection for a frame range.

Usage: ms extract --run-dir DIR --chunk I [--device mps|cpu|cuda:0] [--batch 4]
       ms extract --run-dir DIR --all          (sequential, for single-agent runs)
Writes chunks/<I>/frames.jsonl, PROGRESS.json while running, DONE.json at the end.

- Resumable: a killed/timed-out worker continues from frames.jsonl.tmp on the next run.
- Self-throttling: waits for a machine-wide accelerator slot, so launching more workers
  than the GPU can take only queues them instead of thrashing.
- A finished chunk is a no-op; --force starts the chunk from scratch.
"""
from __future__ import annotations

import argparse
import json
import time

import cv2

from common import (BALL_CLASS, BALL_MODEL, POSE_MODEL, Run, accelerator_slot, model_path, pick_device,
                    write_json)

MAX_PERSONS = 5
PROGRESS_EVERY_S = 10


def athlete_crop(frame, pr):
    """Square crop around the largest person, extended upward for a ball in flight.

    The ball is often < 15 px when the camera is far away; detecting on this crop
    instead of the full frame roughly doubles the ball hit rate.
    """
    H, W = frame.shape[:2]
    if pr.boxes is None or not len(pr.boxes):
        return frame, (0, 0)
    b = pr.boxes.xyxy.cpu().numpy()
    x0, y0, x1, y1 = b[((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax()]
    h = y1 - y0
    side = int(min(max(W, H), 2.4 * h))
    cx = (x0 + x1) / 2
    left = int(min(max(0, cx - side / 2), max(0, W - side)))
    top = int(min(max(0, y0 - 1.0 * h), max(0, H - side)))
    return frame[top:top + side, left:left + side], (left, top)


def detect_batch(pose_model, ball_model, frames, device):
    pose_res = pose_model.predict(frames, device=device, conf=0.25, verbose=False)
    crops = [athlete_crop(fr, pr) for fr, pr in zip(frames, pose_res)]
    ball_res = ball_model.predict([c for c, _ in crops], device=device, conf=0.12, classes=[BALL_CLASS],
                                  imgsz=640, verbose=False)
    out = []
    for pr, br, (_, (ox, oy)) in zip(pose_res, ball_res, crops):
        persons = []
        if pr.boxes is not None and len(pr.boxes):
            boxes = pr.boxes.xyxy.cpu().numpy()
            confs = pr.boxes.conf.cpu().numpy()
            kps = pr.keypoints.data.cpu().numpy()  # (N,17,3)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            for j in areas.argsort()[::-1][:MAX_PERSONS]:
                persons.append({"box": [round(float(v), 1) for v in boxes[j]],
                                "conf": round(float(confs[j]), 3),
                                "kp": [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                                       for x, y, c in kps[j]]})
        balls = []
        if br.boxes is not None and len(br.boxes):
            boxes = br.boxes.xyxy.cpu().numpy()
            confs = br.boxes.conf.cpu().numpy()
            for j in confs.argsort()[::-1][:3]:
                bx = boxes[j] + [ox, oy, ox, oy]
                balls.append({"box": [round(float(v), 1) for v in bx], "conf": round(float(confs[j]), 3)})
        out.append({"persons": persons, "balls": balls})
    return out


def recover_partial(tmp, start_frame):
    """Keep the contiguous, fully written prefix of a previous attempt."""
    recs = []
    if not tmp.exists():
        return recs
    expect = start_frame
    for line in tmp.read_bytes().split(b"\n"):
        try:
            rec = json.loads(line)
        except ValueError:
            break
        if rec.get("f") != expect:
            break
        recs.append(rec)
        expect += 1
    with open(tmp, "w") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return recs


def run_chunk(run: Run, cid: int, device: str, batch: int, force: bool = False) -> dict:
    m = run.load_manifest()
    chunk = m["chunks"][cid]
    cdir = run.chunk_dir(cid)
    done = cdir / "DONE.json"
    if done.exists() and not force:
        return json.loads(done.read_text())
    cdir.mkdir(parents=True, exist_ok=True)
    tmp = cdir / "frames.jsonl.tmp"
    if force:
        tmp.unlink(missing_ok=True)
        done.unlink(missing_ok=True)

    prev = recover_partial(tmp, chunk["start_frame"])
    expected = chunk["end_frame"] - chunk["start_frame"]
    n = len(prev)
    n_person = sum(bool(r["persons"]) for r in prev)
    n_ball = sum(bool(r["balls"]) for r in prev)
    first = chunk["start_frame"] + n

    with accelerator_slot(device) as slot:
        t0 = time.time()
        if first < chunk["end_frame"]:
            from ultralytics import YOLO

            pose_model = YOLO(model_path(POSE_MODEL))
            ball_model = YOLO(model_path(BALL_MODEL))
            cap = cv2.VideoCapture(m["video"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            f_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if f_idx != first:
                raise RuntimeError(f"seek mismatch: wanted {first} got {f_idx}")
            last_report, n_session = 0.0, 0
            with open(tmp, "a") as fh:
                buf, idxs = [], []
                while f_idx < chunk["end_frame"]:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    buf.append(frame)
                    idxs.append(f_idx)
                    f_idx += 1
                    if len(buf) == batch or f_idx >= chunk["end_frame"]:
                        for fi, rec in zip(idxs, detect_batch(pose_model, ball_model, buf, device)):
                            rec["f"] = fi
                            n += 1
                            n_session += 1
                            n_person += bool(rec["persons"])
                            n_ball += bool(rec["balls"])
                            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
                        fh.flush()
                        buf, idxs = [], []
                        el = time.time() - t0
                        if el - last_report >= PROGRESS_EVERY_S:
                            last_report = el
                            fps = n_session / max(el, 1e-6)
                            write_json(cdir / "PROGRESS.json", {
                                "chunk": cid, "done": n, "total": expected, "pct": round(100 * n / expected, 1),
                                "fps": round(fps, 1), "eta_s": round((expected - n) / max(fps, 1e-6)),
                                "slot": slot, "updated": time.strftime("%H:%M:%S")})
            cap.release()
        elapsed = time.time() - t0

    complete = n >= expected - 1
    stats = {"chunk": cid, "frames": n, "expected_frames": expected, "resumed_from": len(prev),
             "person_rate": round(n_person / max(n, 1), 3), "ball_rate": round(n_ball / max(n, 1), 3),
             "device": device, "elapsed_s": round(elapsed, 1), "complete": complete}
    if complete:
        tmp.replace(cdir / "frames.jsonl")
        write_json(done, stats)
        (cdir / "PROGRESS.json").unlink(missing_ok=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--chunk", type=int)
    g.add_argument("--all", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run = Run(args.run_dir)
    device = pick_device(args.device)
    ids = range(len(run.load_manifest()["chunks"])) if args.all else [args.chunk]
    for cid in ids:
        print(json.dumps(run_chunk(run, cid, device, args.batch, args.force)), flush=True)


if __name__ == "__main__":
    main()
