"""Phase 3 (parallel, one worker per event batch): metrics + annotated key frames per event.

Usage: ms analyze --run-dir DIR --events shot_01,shot_02 [--no-clip]
       ms analyze --run-dir DIR --all
Writes events/<id>/{metrics.json, keyframes.jpg, overlay.mp4, DONE.json}.
The analyst agent then reads those and writes events/<id>/review.json (see references/review-schema.md).
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess

import cv2
import numpy as np

from common import (L_ANK, L_ELB, L_HIP, L_KNE, L_SHO, L_WRI, NOSE, R_ANK, R_ELB, R_HIP, R_KNE, R_SHO,
                    R_WRI, SKELETON, SKILL_DIR, Run, angle_deg, clean, nanmean, read_json,
                    tilt_from_vertical_deg, write_json)

RULES = read_json(SKILL_DIR / "references" / "rules.json")
SIDE = {"L": (L_SHO, L_ELB, L_WRI), "R": (R_SHO, R_ELB, R_WRI)}
# jump-shot mechanics that do not apply to a layup
LAYUP_SKIP = {"knee_flex_min_deg", "elbow_set_deg", "dip_to_release_s", "release_vs_apex_s",
              "landing_drift_torso", "follow_through_s", "elbow_alignment_ratio", "launch_angle_deg"}


class Ctx:
    def __init__(self, run: Run):
        t = np.load(run.timeline, allow_pickle=False)
        self.kp, self.ball = t["kp"], t["ball"]
        self.fps, self.torso, self.view = float(t["fps"]), float(t["torso_px"]), str(t["view"])
        self.F = len(self.kp)
        self.off = float(t["offset_s"]) if "offset_s" in t.files else 0.0

    def t(self, f):
        return round(f / self.fps + self.off, 2)

    def p(self, f, j):
        return self.kp[f, j, :2]

    def mid(self, f, a, b):
        return np.nanmean(np.stack([self.kp[f, a, :2], self.kp[f, b, :2]]), axis=0)

    def knee(self, f):
        return nanmean([angle_deg(self.p(f, L_HIP), self.p(f, L_KNE), self.p(f, L_ANK)),
                        angle_deg(self.p(f, R_HIP), self.p(f, R_KNE), self.p(f, R_ANK))])

    def nose_y(self, f):
        y = self.kp[f, NOSE, 1]
        return y if not np.isnan(y) else self.mid(f, L_SHO, R_SHO)[1] - 0.35 * self.torso

    def s(self, sec):
        return max(1, round(sec * self.fps))

    def clamp(self, f):
        return int(min(max(f, 0), self.F - 1))


def metric(group, key, value, view, extra=None):
    r = RULES[group][key]
    m = {"key": key, "label": r["label"], "value": clean(value), "unit": r["unit"],
         "good": r["good"], "reliable": view in r["views"]}
    v = m["value"]
    if v is None:
        m["status"] = "n/a"
    elif r["good"][0] <= v <= r["good"][1]:
        m["status"] = "good"
    elif r["warn"][0] <= v <= r["warn"][1]:
        m["status"] = "warn"
    else:
        m["status"] = "flag"
    if m["status"] in ("warn", "flag"):
        m["cue"] = r.get("low_cue") if v < r["good"][0] else r.get("high_cue")
    if not m["reliable"] and m["status"] != "n/a":
        m["note"] = f"measured from a {view} view; treat as approximate"
    if extra:
        m.update(extra)
    return m


def argext(fn, frames, mode):
    vals = np.array([fn(f) for f in frames], float)
    if np.isnan(vals).all():
        return None
    i = int(np.nanargmin(vals) if mode == "min" else np.nanargmax(vals))
    return frames[i]


# ---------- shot ----------

def analyze_shot(c: Ctx, ev: dict) -> tuple[list, dict, dict]:
    r = ev["release_frame"]
    st, en = ev["start_frame"], ev["end_frame"]
    # shooting hand: wrist nearest the ball just before release, else the higher wrist
    side = None
    b = c.ball[c.clamp(r - 1), :2]
    if not np.isnan(b).any():
        dl, dr = np.linalg.norm(c.p(r - 1, L_WRI) - b), np.linalg.norm(c.p(r - 1, R_WRI) - b)
        if not (np.isnan(dl) and np.isnan(dr)):
            side = "L" if np.nan_to_num(dl, nan=1e9) < np.nan_to_num(dr, nan=1e9) else "R"
    if side is None:
        side = "L" if np.nan_to_num(c.kp[r, L_WRI, 1], nan=1e9) < np.nan_to_num(c.kp[r, R_WRI, 1], nan=1e9) else "R"
    S_SHO, S_ELB, S_WRI = SIDE[side]

    def elbow(f):
        return angle_deg(c.p(f, S_SHO), c.p(f, S_ELB), c.p(f, S_WRI))

    def hip_y(f):
        return c.mid(f, L_HIP, R_HIP)[1]

    def ankle_y(f):
        return nanmean([c.kp[f, L_ANK, 1], c.kp[f, R_ANK, 1]])

    load = argext(c.knee, list(range(c.clamp(r - c.s(1.2)), r + 1)), "min") or c.clamp(r - c.s(0.4))
    up = [f for f in range(load, r + 1) if c.kp[f, S_WRI, 1] < c.kp[f, S_SHO, 1]]
    setp = argext(elbow, up, "min") if up else None
    setp = setp if setp is not None else c.clamp(r - c.s(0.15))
    after = list(range(load, c.clamp(r + c.s(0.5)) + 1))
    apex = argext(hip_y, after, "min") or r
    base = [ankle_y(f) for f in range(st, max(st + 1, load - c.s(0.2)))]
    base_ank = float(np.nanmedian(base)) if not np.isnan(base).all() else math.nan
    lift = (base_ank - np.nanmin([ankle_y(f) for f in after])) / c.torso if not math.isnan(base_ank) else math.nan
    jumped = bool(not math.isnan(lift) and lift > 0.12)
    follow = c.clamp(r + c.s(0.33))
    # layup: travelling into the shot and/or one-foot take-off
    pre = c.clamp(r - c.s(0.5))
    travel = abs(c.mid(r, L_HIP, R_HIP)[0] - c.mid(pre, L_HIP, R_HIP)[0]) / c.torso
    lifts = [(np.nan_to_num(base_ank, nan=0) - np.nanmin([c.kp[f, a, 1] for f in after])) / c.torso
             for a in (L_ANK, R_ANK)] if not math.isnan(base_ank) else [math.nan, math.nan]
    one_foot = bool(jumped and min(lifts) < 0.4 * max(lifts))
    kind = "layup" if (travel > 0.9 or one_foot) else ("jump_shot" if jumped else "set_shot")
    land = c.clamp(r + c.s(0.6))

    elbow_rel = np.nanmax([elbow(c.clamp(f)) for f in range(r - 1, r + 3)])
    rel_h = (c.nose_y(r) - c.kp[r, S_WRI, 1]) / c.torso
    hold = 0
    for f in range(r, c.clamp(r + c.s(2.0)) + 1):
        if c.kp[f, S_WRI, 1] < c.nose_y(f) and np.nan_to_num(elbow(f), nan=0) > 135:
            hold += 1
        else:
            break
    drift = (abs(c.mid(land, L_HIP, R_HIP)[0] - c.mid(load, L_HIP, R_HIP)[0]) / c.torso) if jumped else math.nan

    align = math.nan
    if c.view == "front":
        sw = np.linalg.norm(c.p(setp, L_SHO) - c.p(setp, R_SHO))
        align = abs(c.kp[setp, S_ELB, 0] - c.kp[setp, S_WRI, 0]) / sw if sw > 1 else math.nan

    launch, n_pts = math.nan, 0
    pts = [(f, c.ball[f, 0], c.ball[f, 1]) for f in range(r + 1, c.clamp(r + c.s(0.2)) + 1)
           if not np.isnan(c.ball[f, 0])]
    if len(pts) >= 3:
        fs, xs, ys = (np.array(z, float) for z in zip(*pts))
        vx, vy = np.polyfit(fs, xs, 1)[0], -np.polyfit(fs, ys, 1)[0]
        if vy > 0:
            launch, n_pts = math.degrees(math.atan2(vy, abs(vx))), len(pts)

    v = c.view
    ms = [
        metric("shot", "knee_flex_min_deg", c.knee(load), v),
        metric("shot", "elbow_set_deg", elbow(setp), v),
        metric("shot", "elbow_release_deg", elbow_rel, v),
        metric("shot", "release_height_torso", rel_h, v),
        metric("shot", "dip_to_release_s", (r - load) / c.fps, v),
        metric("shot", "release_vs_apex_s", (r - apex) / c.fps if jumped else math.nan, v,
               None if jumped else {"note": "no jump detected (set shot / free throw)"}),
        metric("shot", "trunk_lean_release_deg",
               tilt_from_vertical_deg(c.mid(r, L_HIP, R_HIP), c.mid(r, L_SHO, R_SHO)), v),
        metric("shot", "landing_drift_torso", drift, v),
        metric("shot", "follow_through_s", hold / c.fps, v),
        metric("shot", "elbow_alignment_ratio", align, v,
               None if v == "front" else {"note": "needs a front view"}),
        metric("shot", "launch_angle_deg", launch, v,
               {"ball_points": n_pts} if n_pts else {"note": "ball not tracked after release"}),
    ]
    if kind == "layup":
        for mm in ms:
            if mm["key"] in LAYUP_SKIP:
                mm.update(status="n/a", note="not used for layups")
                mm.pop("cue", None)
    info = {"shot_kind": kind, "travel_torso": clean(travel), "one_foot_takeoff": one_foot,
            "shooting_hand": "left" if side == "L" else "right", "jumped": jumped,
            "ankle_lift_torso": clean(lift), "view": v}
    keys = {"load": load, "set_point": setp, "release": r, "follow_through": follow}
    return ms, info, keys


# ---------- dribble ----------

def analyze_dribble(c: Ctx, ev: dict) -> tuple[list, dict, dict]:
    bf = ev["bounce_frames"]
    peaks, hands = [], []
    for a, b in zip(bf[:-1], bf[1:]):
        seg = c.ball[a:b + 1, 1]
        if np.isnan(seg).all():
            continue
        pf = a + int(np.nanargmin(seg))
        hip = c.mid(pf, L_HIP, R_HIP)[1]
        peaks.append((hip - c.ball[pf, 1]) / c.torso)
        dl = np.linalg.norm(c.p(pf, L_WRI) - c.ball[pf, :2])
        dr = np.linalg.norm(c.p(pf, R_WRI) - c.ball[pf, :2])
        if not (np.isnan(dl) and np.isnan(dr)):
            hands.append("L" if np.nan_to_num(dl, nan=1e9) < np.nan_to_num(dr, nan=1e9) else "R")
    span = range(ev["start_frame"], ev["end_frame"] + 1)
    freq = (len(bf) - 1) / ((bf[-1] - bf[0]) / c.fps) if bf[-1] > bf[0] else math.nan
    v = c.view
    ms = [
        metric("dribble", "frequency_hz", freq, v),
        metric("dribble", "ball_peak_over_hip_torso", float(np.median(peaks)) if peaks else math.nan, v),
        metric("dribble", "knee_flex_mean_deg", nanmean([c.knee(f) for f in span]), v),
        metric("dribble", "trunk_lean_mean_deg",
               nanmean([tilt_from_vertical_deg(c.mid(f, L_HIP, R_HIP), c.mid(f, L_SHO, R_SHO)) for f in span]), v),
    ]
    switches = sum(1 for a, b in zip(hands, hands[1:]) if a != b)
    info = {"bounces": len(bf), "hand_split": {"left": hands.count("L"), "right": hands.count("R")},
            "hand_switches": switches, "duration_s": round((ev["end_frame"] - ev["start_frame"]) / c.fps, 2),
            "view": v}
    mid_b = bf[len(bf) // 2]
    keys = {"start": ev["start_frame"], "mid": mid_b, "peak_example": c.clamp(mid_b + c.s(0.12)),
            "end": ev["end_frame"]}
    return ms, info, keys


# ---------- visuals ----------

def draw(frame, c: Ctx, f: int, label: str):
    img = frame.copy()
    k = c.kp[f]
    for a, b in SKELETON:
        if not (np.isnan(k[a, :2]).any() or np.isnan(k[b, :2]).any()):
            cv2.line(img, tuple(map(int, k[a, :2])), tuple(map(int, k[b, :2])), (60, 220, 255), 2, cv2.LINE_AA)
    for j in range(17):
        if not np.isnan(k[j, :2]).any():
            cv2.circle(img, tuple(map(int, k[j, :2])), 3, (20, 90, 255), -1, cv2.LINE_AA)
    bl = c.ball[f]
    if not np.isnan(bl).any():
        cv2.circle(img, (int(bl[0]), int(bl[1])), int(max(bl[2], 4)), (0, 165, 255), 2, cv2.LINE_AA)
    if label:
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def crop_box(c: Ctx, st, en, W, H):
    xs = c.kp[st:en + 1, :, 0]
    ys = c.kp[st:en + 1, :, 1]
    if np.isnan(xs).all():
        return 0, 0, W, H
    x0, x1 = np.nanpercentile(xs, 1), np.nanpercentile(xs, 99)
    y0, y1 = np.nanpercentile(ys, 1), np.nanpercentile(ys, 99)
    pad = 0.6 * c.torso
    x0, y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad * 1.3))
    x1, y1 = int(min(W, x1 + pad)), int(min(H, y1 + pad * 0.5))
    return x0, y0, x1, y1


def render(run: Run, c: Ctx, ev: dict, keys: dict, edir, clip: bool):
    cap = cv2.VideoCapture(str(run.video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st, en = ev["start_frame"], ev["end_frame"]
    x0, y0, x1, y1 = crop_box(c, st, en, W, H)
    want = {}
    for name, f in keys.items():
        want.setdefault(f, []).append(name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, st)
    tiles, writer, raw = {}, None, edir / "overlay_raw.mp4"
    for f in range(st, en + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if clip:
            if writer is None:
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), c.fps, (x1 - x0, y1 - y0))
            writer.write(draw(frame, c, f, f"{ev['id']}  t={c.t(f):.2f}s")[y0:y1, x0:x1])
        for name in want.get(f, []):
            tiles[name] = draw(frame, c, f, f"{name} ({c.t(f):.2f}s)")[y0:y1, x0:x1]
    cap.release()
    if writer is not None:
        writer.release()
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(edir / "overlay.mp4")], check=True)
        raw.unlink(missing_ok=True)
    th = 420
    row = [cv2.resize(tiles[n], (max(1, int(tiles[n].shape[1] * th / tiles[n].shape[0])), th))
           for n in keys if n in tiles]
    if row:
        cv2.imwrite(str(edir / "keyframes.jpg"), np.hstack(row), [cv2.IMWRITE_JPEG_QUALITY, 85])


def analyze_event(run: Run, c: Ctx, ev: dict, clip: bool) -> dict:
    edir = run.event_dir(ev["id"])
    edir.mkdir(parents=True, exist_ok=True)
    fn = analyze_shot if ev["type"] == "shot" else analyze_dribble
    ms, info, keys = fn(c, ev)
    render(run, c, ev, keys, edir, clip)
    data = {"schema": "motion-sensei/metrics@1", "event": ev, "info": info,
            "key_frames": {k: {"frame": int(f), "t_s": c.t(f)} for k, f in keys.items()},
            "metrics": ms,
            "flags": [m["key"] for m in ms if m["status"] == "flag"],
            "warns": [m["key"] for m in ms if m["status"] == "warn"],
            "files": {"keyframes": str(edir / "keyframes.jpg"),
                      "overlay": str(edir / "overlay.mp4") if clip else None,
                      "review": str(edir / "review.json")}}
    write_json(edir / "metrics.json", data)
    write_json(edir / "DONE.json", {"event": ev["id"], "flags": data["flags"], "warns": data["warns"]})
    return {"event": ev["id"], "flags": data["flags"], "warns": data["warns"], "keyframes": data["files"]["keyframes"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--events", help="comma-separated event ids")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--no-clip", action="store_true")
    args = ap.parse_args()

    run = Run(args.run_dir)
    evs = {e["id"]: e for e in read_json(run.events)["events"]}
    ids = list(evs) if args.all else [x.strip() for x in args.events.split(",") if x.strip()]
    unknown = [i for i in ids if i not in evs]
    if unknown:
        raise SystemExit(f"unknown events: {unknown}")
    c = Ctx(run)
    for i in ids:
        print(json.dumps(analyze_event(run, c, evs[i], not args.no_clip)))


if __name__ == "__main__":
    main()
