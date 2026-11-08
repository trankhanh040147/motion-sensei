"""Shared helpers: run-dir layout, keypoint indices, geometry, JSON IO."""
from __future__ import annotations

import fcntl
import json
import math
import os
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN slices are expected

SKILL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_DIR.parent.parent
MODELS_DIR = Path(os.environ.get("MS_MODELS_DIR", REPO_ROOT / "models"))
POSE_MODEL = os.environ.get("MS_POSE_MODEL", "yolo11s-pose.pt")
BALL_MODEL = os.environ.get("MS_BALL_MODEL", "yolo11s.pt")
BALL_CLASS = 32  # COCO "sports ball"

# COCO-17 keypoints
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
SKELETON = [
    (L_SHO, R_SHO), (L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB), (R_ELB, R_WRI),
    (L_SHO, L_HIP), (R_SHO, R_HIP), (L_HIP, R_HIP), (L_HIP, L_KNE), (L_KNE, L_ANK),
    (R_HIP, R_KNE), (R_KNE, R_ANK), (NOSE, L_SHO), (NOSE, R_SHO),
]
KP_MIN_CONF = 0.35


def model_path(name: str) -> str:
    p = MODELS_DIR / name
    return str(p) if p.exists() else name


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def total_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError):
        return 8.0


def gpu_slots(device: str) -> int:
    """How many extract processes may share the accelerator at once.

    Measured on a 16 GB M-series Mac: 1 worker ~16 fps, 2 ~22 fps total, 3 ~4 fps total
    (MPS contention + swap at ~2.7 GB per worker). So MPS caps at 2 and also by RAM.
    """
    if "MS_GPU_SLOTS" in os.environ:
        return max(1, int(os.environ["MS_GPU_SLOTS"]))
    ram_cap = max(1, int(total_ram_gb() // 7))
    if device == "mps":
        return min(2, ram_cap)
    if device.startswith("cuda"):
        return min(4, ram_cap)
    return max(1, min((os.cpu_count() or 2) // 4, ram_cap))


@contextmanager
def accelerator_slot(device: str):
    """Machine-wide semaphore (flock files) so parallel agents cannot oversubscribe the GPU.

    Locks are released by the OS if the process dies, so killed workers never leak a slot.
    """
    n = gpu_slots(device)
    lock_dir = Path(os.environ.get("MS_LOCK_DIR", Path.home() / ".cache" / "motion-sensei"))
    lock_dir.mkdir(parents=True, exist_ok=True)
    announced, t0 = False, time.time()
    while True:
        for i in range(n):
            fh = open(lock_dir / f"{device.split(':')[0]}-slot-{i}.lock", "w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            try:
                if announced:
                    print(f"got {device} slot {i} after {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
                yield i
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)
                fh.close()
            return
        if not announced:
            print(f"waiting for a free {device} slot ({n} max; set MS_GPU_SLOTS to change)",
                  file=sys.stderr, flush=True)
            announced = True
        time.sleep(3)


class Run:
    """Layout of one analysis run. Every phase writes only inside its own slot."""

    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir).resolve()

    manifest = property(lambda s: s.dir / "manifest.json")
    video = property(lambda s: s.dir / "video.mp4")
    timeline = property(lambda s: s.dir / "timeline.npz")
    events = property(lambda s: s.dir / "events.json")
    summary = property(lambda s: s.dir / "summary.json")
    report_html = property(lambda s: s.dir / "report.html")
    coach_notes = property(lambda s: s.dir / "coach_notes.md")

    def chunk_dir(self, cid: int) -> Path:
        return self.dir / "chunks" / f"{cid:03d}"

    def event_dir(self, eid: str) -> Path:
        return self.dir / "events" / eid

    def load_manifest(self) -> dict:
        return read_json(self.manifest)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return None if math.isnan(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def clean(v):
    """Convert NaN/numpy scalars into JSON-friendly values."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(float(v)) else round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def angle_deg(a, b, c) -> float:
    """Angle ABC in degrees for 2D points; NaN if any point is missing."""
    a, b, c = np.asarray(a, float), np.asarray(b, float), np.asarray(c, float)
    if np.isnan(a).any() or np.isnan(b).any() or np.isnan(c).any():
        return float("nan")
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def tilt_from_vertical_deg(lower, upper) -> float:
    """Angle between the lower->upper vector and image vertical (up)."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if np.isnan(lower).any() or np.isnan(upper).any():
        return float("nan")
    d = upper - lower
    return float(np.degrees(np.arctan2(abs(d[0]), -d[1]))) if np.linalg.norm(d) > 1e-6 else float("nan")


def nanmean(xs) -> float:
    arr = np.asarray([x for x in xs if x is not None], float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")
