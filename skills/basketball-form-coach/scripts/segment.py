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


def target_mask(run: Run, letters: str) -> dict[int, set[int]]:
    """Frame -> indices of detected persons whose clothing group is one of the chosen letters."""
    path = run.dir / "appearance.npz"
    if not path.exists():
        raise SystemExit("--target needs `ms people --run-dir <R>` first")
    groups = [g["id"] for g in read_json(run.dir / "people.json")["groups"]]
    want = {x.strip().upper() for x in letters.split(",") if x.strip()}
    unknown = want - set(groups)
    if unknown:
        raise SystemExit(f"unknown groups {sorted(unknown)}; available: {groups}")
    a = np.load(path)
    ok = np.isin(a["group"], [groups.index(x) for x in want])
    allowed: dict[int, set[int]] = {}
    for f, j in zip(a["frame"][ok], a["person"][ok]):
        allowed.setdefault(int(f), set()).add(int(j))
    return allowed


def track_athlete(frames, allowed: dict[int, set[int]] | None = None) -> np.ndarray:
    """Pick the same person frame to frame.

    Without a target: the largest, most continuous person (fine for a truly solo video).
    With a target: only persons in the chosen clothing groups; frames without them stay empty,
    so the track never jumps to someone else.
    """
    F = len(frames)
    kp = np.full((F, 17, 3), np.nan)
    prev = None
    h_lo, h_hi = 0.0, 1e9
    if allowed is not None:
        # passers-by close to the camera can share the shirt colour but are far bigger in frame
        hs = [frames[f]["persons"][j]["box"][3] - frames[f]["persons"][j]["box"][1]
              for f, js in allowed.items() if frames[f] for j in js]
        if hs:
            med = float(np.median(hs))
            h_lo, h_hi = 0.5 * med, 1.8 * med
    for f, rec in enumerate(frames):
        if not rec or not rec["persons"]:
            continue
        cands = rec["persons"] if allowed is None else [
            p for j, p in enumerate(rec["persons"])
            if j in allowed.get(f, ()) and h_lo <= p["box"][3] - p["box"][1] <= h_hi]
        if not cands:
            continue
        best, best_s = None, -1e9
        for p in cands:
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


def vet_shot(kp, ball, torso, fps, s) -> str | None:
    """Reason to drop a shot candidate, or None. Cheap checks that save analyst review time."""
    r, F = s["release"], len(kp)
    win = range(max(0, r - 3), min(F, r + 4))
    hips = [mid(kp[f:f + 1], L_HIP, R_HIP)[0] for f in win]
    shos = [mid(kp[f:f + 1], L_SHO, R_SHO)[0] for f in win]
    if all(np.isnan(h).any() for h in hips):
        return "player not visible at release"
    tilts = [np.degrees(np.arctan2(abs(sh[0] - h[0]), h[1] - sh[1])) for h, sh in zip(hips, shos)
             if not (np.isnan(h).any() or np.isnan(sh).any())]
    if tilts and np.median(tilts) > 40:
        return "bent over at 'release' (picking up or dribbling low)"
    if s["evidence"] == "wrist_peak":
        nose = kp[:, NOSE, 1]
        body_x = mid(kp[r:r + 1], L_SHO, R_SHO)[0][0]
        after = [f for f in range(r + 1, min(F, r + round(0.5 * fps)))
                 if not np.isnan(ball[f, 1]) and not np.isnan(body_x) and abs(ball[f, 0] - body_x) < 1.5 * torso]
        if len(after) >= 3:
            ref = np.nanmedian(np.where(np.isnan(nose[after]), np.nan, nose[after]))
            if not np.isnan(ref) and min(ball[f, 1] for f in after) > ref:
                return "ball never rose above the head after 'release' (catch / hold)"
    return None


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
        shots.append({"release": int(release), "evidence": evidence,
                      "peak_wrist_over_head": round(float(wrist_high[a:b + 1].max()), 2)})

    # Ball-flight trigger: catches shots where the shooting wrist is hidden (camera behind
    # the player) but the ball is clearly seen rising well above the head.
    sho = mid(kp, L_SHO, R_SHO)
    sho_x = sho[:, 0].copy()  # hold last known body x through pose dropouts (jumps blur)
    for i in range(1, len(sho_x)):
        if np.isnan(sho_x[i]):
            sho_x[i] = sho_x[i - 1]
    ball_high = np.nan_to_num((nose_y - ball[:, 1]) / torso, nan=-9)
    for a, b in runs_of(ball_high > 0.5, round(0.4 * fps), 2):
        if ball_high[a:b + 1].max() < 0.8:
            continue
        # must find the ball still in the hands at head height just before the flight,
        # otherwise it is a rebound / ball off the wall, not this player's shot
        release = None
        for f in range(a, max(0, a - round(0.5 * fps)), -1):
            if np.isnan(ball[f, 0]):
                continue
            d_wr = np.nanmin([np.linalg.norm(ball[f, :2] - kp[f, w, :2]) for w in (L_WRI, R_WRI)]) / torso
            d_sh = np.linalg.norm(ball[f, :2] - sho[f]) / torso
            if (d_wr < 0.9 or d_sh < 1.0) and ball_high[f] > -0.4:
                release = f
                break
        if release is None:
            # ball first seen already in flight (hands hidden / pose lost mid-jump):
            # accept only if it rises straight above the body, not beside it
            vis = [f for f in range(a, b + 1) if not np.isnan(ball[f, 0])][:4]
            if len(vis) < 2 or np.isnan(sho_x[vis[0]]):
                continue
            dx = abs(ball[vis[0], 0] - sho_x[vis[0]]) / torso
            rising = ball[vis[-1], 1] < ball[vis[0], 1]
            if dx > 0.8 or not rising:
                continue
            release = max(0, vis[0] - 2)
            estimated = True
        else:
            estimated = False
        shots.append({"release": int(release), "evidence": "ball_flight", "release_estimated": estimated,
                      "peak_wrist_over_head": round(float(wrist_high[a:b + 1].max()), 2)})

    rank = {"ball_leaves_hand": 0, "ball_flight": 1, "wrist_peak": 2}
    merged = []
    for s in sorted(shots, key=lambda s: s["release"]):
        if merged and s["release"] - merged[-1]["release"] < round(1.2 * fps):
            if rank[s["evidence"]] < rank[merged[-1]["evidence"]]:
                merged[-1] = s
            continue
        merged.append(s)
    return merged


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
    ap.add_argument("--target", help="clothing groups of the player from `ms people`, e.g. B,C,E")
    args = ap.parse_args()

    run = Run(args.run_dir)
    m = run.load_manifest()
    fps = m["fps"]
    off = float(m.get("time_offset_s", 0.0))
    frames = load_frames(run, m)
    allowed = target_mask(run, args.target) if args.target else None
    kp = clean_kp(track_athlete(frames, allowed), fps)
    F = len(kp)

    sho_mid, hip_mid = mid(kp, L_SHO, R_SHO), mid(kp, L_HIP, R_HIP)
    torso_series = np.linalg.norm(sho_mid - hip_mid, axis=1)
    if np.isnan(torso_series).all():
        raise SystemExit("no athlete detected in the video (check framing / lighting)")
    torso = float(np.nanmedian(torso_series))
    sho_w = float(np.nanmedian(np.linalg.norm(kp[:, L_SHO, :2] - kp[:, R_SHO, :2], axis=1)))
    ratio = sho_w / torso
    view = "side" if ratio < 0.38 else ("front" if ratio > 0.5 else "oblique")
    # front vs back look alike by shoulder width; the face tells them apart
    body = (kp[:, L_SHO, 2] > 0.5) & (kp[:, R_SHO, 2] > 0.5)
    face_rate = float((kp[body, NOSE, 2] > 0.5).mean()) if body.any() else 1.0
    if view in ("front", "oblique") and face_rate < 0.35:
        view = "back"

    ball = track_ball(frames, kp, torso, fps)
    person_rate = float((~np.isnan(kp[:, L_HIP, 0]) | ~np.isnan(kp[:, R_HIP, 0])).mean())
    ball_rate = float((~np.isnan(ball[:, 0])).mean())

    events = []
    pre, post = round(1.5 * fps), round(1.0 * fps)
    blocked = np.zeros(F, bool)
    candidates, rejected = detect_shots(kp, ball, torso, fps), []
    kept = []
    for s in candidates:
        why = vet_shot(kp, ball, torso, fps, s)
        if why:
            rejected.append({"t_s": round(s["release"] / fps + off, 2), "evidence": s["evidence"], "reason": why})
        else:
            kept.append(s)
    for i, s in enumerate(kept):
        st, en = max(0, s["release"] - pre), min(F - 1, s["release"] + post)
        blocked[st:en + 1] = True
        events.append({"id": f"shot_{i + 1:02d}", "type": "shot", "start_frame": st, "end_frame": en,
                       "release_frame": s["release"], "t_release_s": round(s["release"] / fps + off, 2),
                       "detection": s["evidence"], "release_estimated": s.get("release_estimated", False),
                       "confidence": "medium" if s["evidence"] == "wrist_peak" else "high",
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
    big = [sum(1 for q in r["persons"] if (q["box"][3] - q["box"][1]) > 0.5 * max(x["box"][3] - x["box"][1] for x in r["persons"]))
           for r in frames if r and r["persons"]]
    multi_rate = float(np.mean([n >= 2 for n in big])) if big else 0.0
    if multi_rate > 0.15 and not args.target:
        warnings.append(f"2+ people in {multi_rate:.0%} of frames and no --target: the skeleton may follow the wrong "
                        "person; run `ms people`, pick the player's groups, re-run segment with --target")
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
        "rejected_candidates": rejected,
        "target": args.target.upper() if args.target else None, "multi_person_rate": round(multi_rate, 3),
        "view": view, "face_visible_rate": round(face_rate, 3), "shoulder_torso_ratio": round(ratio, 3), "torso_px": round(torso, 1),
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
