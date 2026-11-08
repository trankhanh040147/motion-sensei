"""Who is in the video? Cluster every detected person by clothing colour so the right player can be targeted.

Usage: ms people --run-dir DIR [--max-groups 8]
Run after extract. Writes appearance.npz, people.json and people.jpg (a gallery: one row per person group
A, B, C... with sample crops across the video). Then: ms segment --run-dir DIR --target <letters>.
Groups are kept pure rather than complete: one player often spans several groups (lighting, facing
direction), so pick every row that shows the player, e.g. --target B,C,E.
CPU only (re-reads the normalized video; no model inference).
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from common import KP_MIN_CONF, L_HIP, L_KNE, L_SHO, R_HIP, R_KNE, R_SHO, Run, write_json
from segment import load_frames

MERGE_DIST = 10.0   # weighted Lab units: groups closer than this are the same clothes
# Shirt colour (chroma a/b) separates people best; brightness shifts with lighting and pose, and the
# hip-knee "shorts" region picks up skin, so both are down-weighted.
FEAT_W = np.array([0.5, 1.5, 1.5, 0.1, 0.3, 0.3], np.float32)
MIN_SHARE = 0.03    # groups below 3% of detections are noise


def region_mean(lab, pts, box, frac):
    """Mean Lab colour inside a keypoint quad, or a fallback band of the box."""
    H, W = lab.shape[:2]
    if pts is not None:
        poly = np.round(pts).astype(np.int32)
        x0, y0 = np.clip(poly.min(0), 0, [W - 1, H - 1])
        x1, y1 = np.clip(poly.max(0) + 1, 1, [W, H])
        if x1 - x0 >= 3 and y1 - y0 >= 3:
            mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(mask, [poly - [x0, y0]], 1)
            if mask.sum() >= 6:
                return lab[y0:y1, x0:x1][mask.astype(bool)].mean(0)
    bx0, by0, bx1, by1 = box
    h, w = by1 - by0, bx1 - bx0
    y0, y1 = int(by0 + frac[0] * h), int(by0 + frac[1] * h)
    x0, x1 = int(bx0 + 0.3 * w), int(bx1 - 0.3 * w)
    y0, y1, x0, x1 = max(0, y0), min(H, y1), max(0, x0), min(W, x1)
    if y1 - y0 < 2 or x1 - x0 < 2:
        return np.full(3, np.nan)
    return lab[y0:y1, x0:x1].reshape(-1, 3).mean(0)


def descriptor(lab, p):
    kp = np.array(p["kp"], float)
    ok = kp[:, 2] >= KP_MIN_CONF

    def quad(a, b, c, d):
        return kp[[a, b, c, d], :2] if ok[[a, b, c, d]].all() else None

    shirt = region_mean(lab, quad(L_SHO, R_SHO, R_HIP, L_HIP), p["box"], (0.2, 0.45))
    shorts = region_mean(lab, quad(L_HIP, R_HIP, R_KNE, L_KNE), p["box"], (0.5, 0.65))
    return np.concatenate([shirt, shorts]) * FEAT_W


def kmeans(x, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), 1)
        new = np.array([x[lab == i].mean(0) if (lab == i).any() else c[i] for i in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return c


def group(desc, max_groups):
    sub = desc[np.random.default_rng(1).choice(len(desc), min(len(desc), 20000), replace=False)]
    cents = list(kmeans(sub, min(max_groups, len(sub))))
    merged = True
    while merged and len(cents) > 1:  # merge clusters wearing the same clothes
        merged = False
        d = np.array([[np.linalg.norm(a - b) for b in cents] for a in cents]) + np.eye(len(cents)) * 1e9
        i, j = np.unravel_index(d.argmin(), d.shape)
        if d[i, j] < MERGE_DIST:
            cents[i] = (cents[i] + cents[j]) / 2
            cents.pop(j)
            merged = True
    cents = np.array(cents)
    lab = np.argmin(((desc[:, None] - cents[None]) ** 2).sum(-1), 1)
    keep = [i for i in range(len(cents)) if (lab == i).mean() >= MIN_SHARE]
    keep.sort(key=lambda i: -(lab == i).sum())
    cents = cents[keep]
    lab = np.argmin(((desc[:, None] - cents[None]) ** 2).sum(-1), 1)
    dist = np.linalg.norm(desc - cents[lab], axis=1)
    return cents, lab, dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-groups", type=int, default=8)
    args = ap.parse_args()
    run = Run(args.run_dir)
    m = run.load_manifest()
    frames = load_frames(run, m)
    fps = m["fps"]

    rows, descs = [], []  # rows: (frame, person index, box height)
    cap = cv2.VideoCapture(m["video"])
    for f in range(len(frames)):
        ok, img = cap.read()
        if not ok:
            break
        rec = frames[f]
        if not rec or not rec["persons"]:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        for j, p in enumerate(rec["persons"]):
            d = descriptor(lab, p)
            if not np.isnan(d).any():
                rows.append((f, j, p["box"][3] - p["box"][1]))
                descs.append(d)
    cap.release()
    if not descs:
        raise SystemExit("no people detected")
    rows, descs = np.array(rows, float), np.array(descs, np.float32)
    cents, lab, dist = group(descs, args.max_groups)
    np.savez_compressed(run.dir / "appearance.npz", frame=rows[:, 0].astype(int), person=rows[:, 1].astype(int),
                        desc=descs, group=lab, dist=dist)

    letters = "ABCDEFGHIJ"
    groups = []
    for g in range(len(cents)):
        idx = np.flatnonzero(lab == g)
        spread = float(np.percentile(dist[idx], 90))
        # samples: the most typical detection in each fifth of the video where the group appears
        order = idx[np.argsort(rows[idx, 0])]
        samples = []
        for part in np.array_split(order, 5):
            if len(part):
                best = part[np.argmin(dist[part] - 0.002 * rows[part, 2])]  # typical + large
                samples.append(int(best))
        frames_seen = len(set(rows[idx, 0].astype(int)))
        groups.append({"id": letters[g], "share": round(len(idx) / len(descs), 3),
                       "frames": frames_seen, "seconds_visible": round(frames_seen / fps, 1),
                       "centroid": [round(float(v), 1) for v in cents[g]], "spread_p90": round(spread, 1),
                       "samples": [{"frame": int(rows[s, 0]), "person": int(rows[s, 1]),
                                    "t_s": round(rows[s, 0] / fps + m.get("time_offset_s", 0.0), 1)} for s in samples]})

    # gallery image
    cap = cv2.VideoCapture(m["video"])
    th = 180
    strips = []
    for g in groups:
        tiles = []
        for s in g["samples"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s["frame"])
            ok, img = cap.read()
            if not ok:
                continue
            x0, y0, x1, y1 = map(int, frames[s["frame"]]["persons"][s["person"]]["box"])
            pad = int(0.15 * (y1 - y0))
            crop = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
            tiles.append(cv2.resize(crop, (max(1, int(crop.shape[1] * th / crop.shape[0])), th)))
        label = np.full((th, 150, 3), 255, np.uint8)
        cv2.putText(label, g["id"], (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(label, f"{g['seconds_visible']:.0f}s seen", (12, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
        cv2.putText(label, f"{g['share'] * 100:.0f}% of people", (12, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1, cv2.LINE_AA)
        strips.append(np.hstack([label] + tiles))
    cap.release()
    W = max(s.shape[1] for s in strips)
    sheet = np.vstack([np.pad(s, ((0, 6), (0, W - s.shape[1]), (0, 0)), constant_values=255) for s in strips])
    cv2.imwrite(str(run.dir / "people.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    write_json(run.dir / "people.json", {"schema": "motion-sensei/people@1", "groups": groups})
    print(json.dumps({"gallery": str(run.dir / "people.jpg"),
                      "groups": [{k: g[k] for k in ("id", "share", "seconds_visible")} for g in groups],
                      "next": f"ms segment --run-dir {run.dir} --target <letters of every row showing the player, e.g. B,C>"}, indent=2))


if __name__ == "__main__":
    main()
