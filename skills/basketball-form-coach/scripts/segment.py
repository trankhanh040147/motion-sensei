"""Phase 2 (lead, fast): merge chunk outputs, track the athlete + ball, find shot / dribble events.

Usage: ms segment --run-dir DIR [--batch-size 3]
Writes timeline.npz and events.json (with analyst task batches for the next fan-out).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import (KP_MIN_CONF, L_ANK, L_HIP, L_SHO, L_WRI, NOSE, R_ANK, R_HIP, R_SHO, R_WRI, Run,
                    clean, read_json, write_json)


# ---------- merge ----------

def load_frames(run: Run, m: dict) -> list[dict | None]:
    frames: list[dict | None] = [None] * m["frames"]
    missing = []
    for c in m["chunks"]:
        cdir = run.chunk_dir(c["id"])
        if not (cdir / "DONE.json").exists():
            missing.append(c["id"])
            continue
        with open(cdir / "frames.jsonl") as fh:
            for line in fh:
                rec = json.loads(line)
                if c["core_start"] <= rec["f"] < c["core_end"]:
                    frames[rec["f"]] = rec
    if missing:
        raise SystemExit(f"chunks not extracted yet: {missing} -> run `ms extract --chunk <id>` first")
    return frames


def box_center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def track_athlete(frames) -> np.ndarray:
    """Pick the same person frame to frame (solo practice: usually the largest, most continuous)."""
    F = len(frames)
    kp = np.full((F, 17, 3), np.nan)
    prev = None
    for f, rec in enumerate(frames):
        if not rec or not rec["persons"]:
            continue
        best, best_s = None, -1e9
        for p in rec["persons"]:
            b = p["box"]
            h = max(b[3] - b[1], 1)
            s = h / 100.0 + p["conf"]
            if prev is not None:
                s -= 3.0 * np.linalg.norm(box_center(b) - prev[0]) / prev[1]
            if s > best_s:
                best, best_s = p, s
        b = best["box"]
        prev = (box_center(b), max(b[3] - b[1], 1))
        kp[f] = np.array(best["kp"], float)
    low = kp[:, :, 2] < KP_MIN_CONF
    kp[low, 0] = np.nan
    kp[low, 1] = np.nan
    return kp


def interp_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    x = x.copy()
    isn = np.isnan(x)
    if isn.all() or not isn.any():
        return x
    idx = np.arange(len(x))
    filled = np.interp(idx, idx[~isn], x[~isn])
    # only fill short interior gaps
    f = 0
    while f < len(x):
        if isn[f]:
            g = f
            while g < len(x) and isn[g]:
                g += 1
            if f > 0 and g < len(x) and g - f <= max_gap:
                x[f:g] = filled[f:g]
            f = g
        else:
            f += 1
    return x


def smooth(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    k = np.ones(win)
    valid = ~np.isnan(x)
    num = np.convolve(np.where(valid, x, 0.0), k, mode="same")
    den = np.convolve(valid.astype(float), k, mode="same")
    out = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    out[~valid] = np.nan
    return out


def clean_kp(kp: np.ndarray, fps: float) -> np.ndarray:
    gap = max(2, round(0.1 * fps))
    win = 3 if fps <= 35 else 5
    out = kp.copy()
    for j in range(17):
        for d in range(2):
            out[:, j, d] = smooth(interp_gaps(kp[:, j, d], gap), win)
    return out


def mid(kp, a, b):
    return np.nanmean(np.stack([kp[:, a, :2], kp[:, b, :2]]), axis=0)


def track_ball(frames, kp, torso_px: float, fps: float) -> np.ndarray:
    """Ball center + radius per frame; rejects detections whose size is implausible."""
    F = len(frames)
    ball = np.full((F, 3), np.nan)
    wr = np.stack([kp[:, L_WRI, :2], kp[:, R_WRI, :2]], axis=1)
    prev = None
    for f, rec in enumerate(frames):
        if not rec or not rec["balls"]:
            continue
        best, best_s = None, -1e9
        for bd in rec["balls"]:
            b = bd["box"]
            r = ((b[2] - b[0]) + (b[3] - b[1])) / 4
            if not (0.08 * torso_px <= r <= 0.6 * torso_px):
                continue
            c = box_center(b)
            s = bd["conf"]
            if prev is not None and f - prev[1] <= fps:
                s -= np.linalg.norm(c - prev[0]) / (torso_px * 2)
            elif not np.isnan(wr[f]).all():
                s -= np.nanmin(np.linalg.norm(wr[f] - c, axis=1)) / (torso_px * 4)
            if s > best_s:
                best, best_s = (c, r), s
        if best is not None:
            ball[f] = [best[0][0], best[0][1], best[1]]
            prev = (best[0], f)
    gap = max(1, round(0.07 * fps))
    for d in range(3):
        ball[:, d] = interp_gaps(ball[:, d], gap)
    return ball


# ---------- events ----------

def runs_of(mask: np.ndarray, merge_gap: int, min_len: int):
    idx = np.flatnonzero(mask)
    if not idx.size:
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > merge_gap:
            out.append((s, p))
            s = i
        p = i
    out.append((s, p))
    return [(a, b) for a, b in out if b - a + 1 >= min_len]


def detect_shots(kp, ball, torso, fps):
    F = len(kp)
    nose_y = kp[:, NOSE, 1].copy()
    sho_y = mid(kp, L_SHO, R_SHO)[:, 1]
    nose_y = np.where(np.isnan(nose_y), sho_y - 0.35 * torso, nose_y)
    wrist_high = np.nanmax(np.stack([(nose_y - kp[:, L_WRI, 1]), (nose_y - kp[:, R_WRI, 1])]), axis=0) / torso
    wrist_high = np.nan_to_num(wrist_high, nan=-9)
    cand = wrist_high > 0.15
    shots = []
    for a, b in runs_of(cand, round(0.25 * fps), max(2, round(0.12 * fps))):
        release, evidence = None, "wrist_peak"
        lo, hi = max(1, a - round(0.2 * fps)), min(F - 1, b + round(0.3 * fps))
        has_ball = ~np.isnan(ball[:, 0])
        if has_ball[lo:hi].mean() > 0.3:
            held = False
            for f in range(lo, hi):
                if not has_ball[f]:
                    continue
                d = np.nanmin([np.linalg.norm(ball[f, :2] - kp[f, w, :2]) for w in (L_WRI, R_WRI)]) / torso
                if np.isnan(d):
                    continue
                if d < 0.55:
                    held = True
                elif held and d > 0.7 and ball[f, 1] < nose_y[f] and has_ball[f - 1] and ball[f, 1] < ball[f - 1, 1]:
                    release, evidence = f - 1, "ball_leaves_hand"
                    break
        if release is None:
            release = a + int(np.argmax(wrist_high[a:b + 1]))
        if shots and release - shots[-1]["release"] < round(1.2 * fps):
            continue
        shots.append({"release": int(release), "evidence": evidence,
                      "peak_wrist_over_head": round(float(wrist_high[a:b + 1].max()), 2)})
    return shots


def detect_dribbles(kp, ball, torso, fps, blocked: np.ndarray):
    hip_y = mid(kp, L_HIP, R_HIP)[:, 1]
    by = ball[:, 1]
    low = (by > hip_y - 0.1 * torso) & ~blocked
    bounces = []
    for f in range(2, len(by) - 2):
        if not low[f] or np.isnan(by[f - 2:f + 3]).any():
            continue
        if by[f] == np.max(by[f - 2:f + 3]) and by[f] - np.nanmin(by[max(0, f - round(0.4 * fps)):f + 1]) > 0.35 * torso:
            if not bounces or f - bounces[-1] > round(0.15 * fps):
                bounces.append(f)
    seqs, cur = [], []
    for f in bounces:
        if cur and f - cur[-1] > round(1.2 * fps):
            seqs.append(cur)
            cur = []
        cur.append(f)
    if cur:
        seqs.append(cur)
    return [s for s in seqs if len(s) >= 4]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=3, help="events per analyst task")
    args = ap.parse_args()

    run = Run(args.run_dir)
    m = run.load_manifest()
    fps = m["fps"]
    off = float(m.get("time_offset_s", 0.0))
    frames = load_frames(run, m)
    kp = clean_kp(track_athlete(frames), fps)
    F = len(kp)

    sho_mid, hip_mid = mid(kp, L_SHO, R_SHO), mid(kp, L_HIP, R_HIP)
    torso_series = np.linalg.norm(sho_mid - hip_mid, axis=1)
    if np.isnan(torso_series).all():
        raise SystemExit("no athlete detected in the video (check framing / lighting)")
    torso = float(np.nanmedian(torso_series))
    sho_w = float(np.nanmedian(np.linalg.norm(kp[:, L_SHO, :2] - kp[:, R_SHO, :2], axis=1)))
    ratio = sho_w / torso
    view = "side" if ratio < 0.38 else ("front" if ratio > 0.5 else "oblique")

    ball = track_ball(frames, kp, torso, fps)
    person_rate = float((~np.isnan(kp[:, L_HIP, 0]) | ~np.isnan(kp[:, R_HIP, 0])).mean())
    ball_rate = float((~np.isnan(ball[:, 0])).mean())

    events = []
    pre, post = round(1.5 * fps), round(1.0 * fps)
    blocked = np.zeros(F, bool)
    for i, s in enumerate(detect_shots(kp, ball, torso, fps)):
        st, en = max(0, s["release"] - pre), min(F - 1, s["release"] + post)
        blocked[st:en + 1] = True
        events.append({"id": f"shot_{i + 1:02d}", "type": "shot", "start_frame": st, "end_frame": en,
                       "release_frame": s["release"], "t_release_s": round(s["release"] / fps + off, 2),
                       "detection": s["evidence"],
                       "confidence": "high" if s["evidence"] == "ball_leaves_hand" else "medium",
                       "peak_wrist_over_head_torso": s["peak_wrist_over_head"]})
    for i, seq in enumerate(detect_dribbles(kp, ball, torso, fps, blocked)):
        st, en = max(0, seq[0] - round(0.3 * fps)), min(F - 1, seq[-1] + round(0.3 * fps))
        events.append({"id": f"dribble_{i + 1:02d}", "type": "dribble", "start_frame": st, "end_frame": en,
                       "bounce_frames": [int(x) for x in seq], "t_start_s": round(st / fps + off, 2),
                       "detection": "ball_bounces", "confidence": "medium"})
    events.sort(key=lambda e: e["start_frame"])

    np.savez_compressed(run.timeline, kp=kp, ball=ball, fps=fps, torso_px=torso, view=view, offset_s=off)
    ids = [e["id"] for e in events]
    batches = [ids[i:i + args.batch_size] for i in range(0, len(ids), args.batch_size)]
    warnings = []
    if person_rate < 0.6:
        warnings.append(f"athlete visible in only {person_rate:.0%} of frames")
    if ball_rate < 0.25:
        warnings.append(f"ball detected in only {ball_rate:.0%} of frames: release/arc metrics will be approximate")
    if view == "oblique":
        warnings.append("camera is oblique: angle metrics are less reliable; film from the side or front")
    if not events:
        warnings.append("no shot or dribble events found")

    out = {
        "schema": "motion-sensei/events@1",
        "view": view, "shoulder_torso_ratio": round(ratio, 3), "torso_px": round(torso, 1),
        "athlete_visible_rate": round(person_rate, 3), "ball_visible_rate": round(ball_rate, 3),
        "warnings": warnings, "events": events,
        "analyst_tasks": [{"task": k, "events": b,
                           "command": f"ms analyze --run-dir {run.dir} --events {','.join(b)}"}
                          for k, b in enumerate(batches)],
    }
    write_json(run.events, out)
    print(json.dumps({"view": view, "n_shots": sum(e["type"] == "shot" for e in events),
                      "n_dribble_sequences": sum(e["type"] == "dribble" for e in events),
                      "warnings": warnings,
                      "parallel_tasks": [t["command"] for t in out["analyst_tasks"]],
                      "next_after_all_done": f"ms aggregate --run-dir {run.dir}"},
                     indent=2, default=clean))


if __name__ == "__main__":
    main()
