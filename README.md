# motion-sensei

AI-powered motion analysis and coaching, packaged as agent skills.

## Skills

| Skill | What it does |
|---|---|
| [basketball-form-coach](skills/basketball-form-coach/SKILL.md) | Solo practice video → per-shot / per-dribble measurements, annotated key frames, skeleton clips, and a coaching report. Designed for agent teams: extraction and per-event review run in parallel. |

## How it works

```
video ──prepare──► N chunks ──extract (parallel, YOLO pose + ball)──► segment ──► events
                                                                                  │
          report.html ◄── coach notes ◄── aggregate ◄── analyze + visual review (parallel per event batch)
```

- Measurements (joint angles, timing, release height, ball launch angle) come from scripts, never from the LLM.
- Agents check the measurements against the key frames, add what only eyes can see, and write the coaching.
- Phases talk only through files in the run directory, so each parallel worker needs no shared chat context.

## Install

Requires macOS or Linux with Homebrew (ffmpeg and uv get installed if missing).

```bash
./bin/install-skill.sh
```

This links the skill into `~/.agents/skills` and `~/.claude/skills`, creates a Python 3.12 env in `.venv`, and downloads the YOLO weights into `models/`.

## Run without an agent

```bash
skills/basketball-form-coach/scripts/ms run-all path/to/practice.mp4
```

The output lands in `runs/<video>-<timestamp>/`. Open `dashboard.html` there to replay every detected shot from the timeline, with its review and measurements. The visual review and coach notes need an agent (see SKILL.md).

## Models

| Purpose | Default | Override |
|---|---|---|
| Pose | `yolo11s-pose.pt` | `MS_POSE_MODEL` |
| Ball | `yolo11s.pt` on a 640px crop around the player (COCO "sports ball") | `MS_BALL_MODEL=yolo11m.pt` (more accurate, ~3× slower) |

On an M-series Mac (MPS) one worker processes about 25–30 frames/s.
