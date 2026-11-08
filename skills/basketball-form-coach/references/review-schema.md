# review.json (one per event, written by the analyst)

Path: `<run>/events/<id>/review.json`. `ms aggregate` reads it; invalid JSON breaks the report, so write valid JSON only.

```json
{
  "event": "shot_03",
  "detection_ok": true,
  "shot_kind": "jump_shot",
  "pose_ok": true,
  "metric_checks": [
    {"key": "elbow_set_deg", "verdict": "confirmed", "why": "set point tile shows forearm ~45° behind vertical, ball behind head"},
    {"key": "launch_angle_deg", "verdict": "rejected", "why": "ball circle is on the backboard, not the ball"}
  ],
  "observations": [
    {"text": "guide hand thumb pushes the ball at release", "frame": "release", "severity": "medium"}
  ],
  "summary": "Good leg drive, but the ball goes behind the head at the set point and the arm does not fully extend."
}
```

Fields

- `detection_ok`: false if the event is not a real shot/dribble (e.g. a pass, picking up the ball, stretching). Aggregate then drops the event.
- `shot_kind` (shots only, optional): `jump_shot` | `set_shot` | `layup`. Set it when `metrics.json.info.shot_kind` is wrong (e.g. a one-foot layup classified as a jump shot); aggregate groups by your value.
- `pose_ok`: false if the skeleton is on the wrong person or badly off in 2+ key frames. Aggregate then drops all its metrics.
- `metric_checks`: one entry for **every** metric with status `flag` or `warn`. `verdict` is one of `confirmed`, `rejected`, `uncertain`. Rejected metrics are excluded from the summary.
- `observations`: 0–3 visual-only points. `frame` is a key-frame name from `metrics.json.key_frames`. `severity`: `high` | `medium` | `low`.
- `summary`: 1–2 sentences, plain language.
