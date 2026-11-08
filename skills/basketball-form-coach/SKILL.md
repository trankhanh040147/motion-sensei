---
name: basketball-form-coach
description: Analyze a solo basketball practice video (jump shots, free throws, set shots, stationary/moving dribbling) and give form corrections. Pose + ball tracking measure joint angles, timing, release and arc per shot; agents visually verify key frames and write a coaching report. Built for agent teams - extraction and per-shot analysis fan out in parallel. Use when the user sends a basketball video and asks to analyze form, fix shooting technique, "phân tích động tác", "sửa tư thế ném rổ", or review dribbling.
---

# Basketball Form Coach

Turns one practice video into: per-shot measurements, annotated key frames, a skeleton clip per event, and a ranked list of form corrections with drills.

Measurements come from scripts (YOLO pose + ball detection). Agents never invent numbers: they **verify** the measurements against the key-frame images, add what only eyes can see, and write the coaching.

All commands go through `scripts/ms` (path relative to this skill folder; it resolves its own repo and venv). First use on a machine: `scripts/ms setup` (installs uv/ffmpeg if missing, Python 3.12 env, model weights). It is idempotent.

## Pipeline and file contract

Every phase reads and writes files in one run dir. No phase needs another agent's chat context, so any phase marked **parallel** can be split across team members.

| Phase | Who | Command | Output | Parallel |
|---|---|---|---|---|
| 0 Prepare | lead | `ms prepare <video> [--workers N]` | `manifest.json`, `video.mp4` | - |
| 1 Extract | extractor ×N | `ms extract --run-dir R --chunk i` | `chunks/i/frames.jsonl`, `DONE.json` | **yes, per chunk** |
| 2a People | lead | `ms people --run-dir R` | `people.jpg` gallery of clothing groups A, B, C… | - (≈1 min CPU) |
| 2 Segment | lead | `ms segment --run-dir R [--target B,D]` | `timeline.npz`, `events.json` | - (seconds) |
| 3 Analyze + review | analyst ×M | `ms analyze --run-dir R --events a,b,c` then visual review | `events/<id>/metrics.json`, `keyframes.jpg`, `overlay.mp4`, **`review.json`** | **yes, per event batch** |
| 4 Report | lead | `ms aggregate`, write `coach_notes.md`, `ms aggregate` again, `ms dashboard` | `summary.json`, `report.html`, `dashboard.html` | - |

`ms status --run-dir R` prints the current phase, per-chunk progress (done / total / fps / ETA) and the exact commands that can run now. Use it instead of guessing; it is also how a resumed or new lead picks up a half-finished run.

Safety properties the lead can rely on:
- **Idempotent + resumable.** A finished chunk/event is a no-op. A killed or timed-out extract resumes from its partial file on the next run of the same command. `--force` is the only way to discard progress; never use it unless the user asks.
- **Self-throttling GPU.** Extract takes a machine-wide accelerator slot (file lock). The slot count comes from device and RAM (MPS: 2 on 16 GB, 1 on 8 GB; override `MS_GPU_SLOTS`). Extra workers wait instead of thrashing: measured on a 16 GB M-series Mac, 3 concurrent MPS workers ran 5× slower in total than 2.

Each command prints JSON. `prepare` and `segment` print `parallel_tasks`: one command per worker. Dispatch exactly those.

## Lead playbook (agent team)

1. **Prepare.** `ms prepare <video>`. Leave `--workers` at its default (= `max_concurrent_extract`, the GPU slots of this machine). Throughput is roughly `max_concurrent_extract × 10 fps` on Apple silicon, so estimate `frames / that` and tell the user the ETA. For videos longer than ~5 minutes, ask whether to analyze everything or a part (`--start S --end S`, seconds; report times stay in source-video time).
2. **Fan out extraction.** Exactly one extractor per `parallel_tasks` entry; do not add more. This is pure compute: give it to the cheapest/fastest member. Brief: *"Run `<command>` from `<skill>/scripts`. If your shell tool has a timeout shorter than the ETA, run it in the background and poll `ms status --run-dir <R>` every few minutes. On timeout or crash, run the same command again (it resumes). Never use `--force`, never kill other workers' processes. Report the final JSON line."* Wait for all of them (`ms status` shows `pending_chunks` and progress).
3. **Pick the player.** Run `ms segment --run-dir R` first. If it does not warn about 2+ people (`multi_person_rate` < 0.15 in `events.json`), the video is effectively solo: keep that result and go to step 4. Otherwise run `ms people --run-dir R`, look at `R/people.jpg` and pick **every** row that shows the player (groups stay pure, so one player usually spans several rows; leave out rows that mix two people). Use the user's description ("purple #77", "black tank top"); if they gave none, show them `people.jpg` and ask. Never guess between two plausible players.
   Then **re-segment**: `ms segment --run-dir R --target <letters>`. Read `warnings` (view, ball visibility, athlete visibility) and `rejected_candidates` in `events.json` (pick-ups, defending, catches dropped before review, each with a reason). If 0 events, stop and tell the user why (see `references/filming-guide.md`).
4. **Fan out analysis.** One analyst task per `parallel_tasks` entry (default 3 events each; change with `segment --batch-size`). Analysts need vision. Give each the brief in "Analyst brief" below. They write only inside their own `events/<id>/` dirs, so they cannot conflict. If the harness caps team size (DSH: 8 members), do not spawn past it: hand each member that finishes the next pending batch (finished extractors make good analysts). `ms status` lists events still missing a review.
5. **Aggregate.** `ms aggregate --run-dir R` → `summary.json`. Read it plus every `review.json`.
6. **Coach notes.** Write `R/coach_notes.md` (format below), then run `ms aggregate` again to embed it in `report.html`.
7. **Dashboard.** `ms dashboard --run-dir R` → `R/dashboard.html`: the video with a clickable timeline of every event (colour = confirmed issue / possible / good / invalid / not reviewed), per-event review, key frames, metrics with verdicts and a looping skeleton replay; 0.25×–2× speed and frame stepping. It opens straight from disk (no server, nothing uploaded) and works on partial runs, so it can be rebuilt any time. Reply to the user with the notes and the `dashboard.html` path (deep link to one event: `dashboard.html#shot_12`).

**Stopping a run.** Tell every worker to stop and not retry, then `pkill -f "basketball-form-coach/scripts/"`. Nothing is lost: re-running the same commands later resumes. Do not launch detached/`setsid` processes; they outlive the team and keep the GPU busy.

If the harness has no team (single agent), run `ms run-all <video>` (local processes for the parallel phases; it stops after `people` when several people are in frame, then re-run it with `--run-dir R --target <letters>`), then do the analyst review yourself for each event, then steps 5–7.

**Re-running from segment** (new target, new filters) keeps extraction: move `events/` and `events.json` aside (do not delete the old reviews), then run from step 3.

## Analyst brief (copy into each analyst task)

> You are reviewing basketball events `<ids>` in run `<R>`. Read `<skill>/references/shooting-form.md` (and `dribbling.md` for dribble events) and `<skill>/references/review-schema.md`.
> 1. Run `<skill>/scripts/ms analyze --run-dir <R> --events <ids>`.
> 2. For each event: open `events/<id>/keyframes.jpg` (load → set point → release → follow-through, or dribble frames) and read `events/<id>/metrics.json`.
> 3. Check that the skeleton sits on the athlete and the event really is a shot/dribble. Then check every `flag`/`warn` metric against the image: `confirmed`, `rejected` (skeleton or ball detection is wrong there) or `uncertain`. Add up to 3 observations the metrics cannot capture (guide-hand thumb push, eyes, ball position in hand, foot alignment).
> 4. Write `events/<id>/review.json` exactly per the schema. Never change metric values; never write outside `events/<id>/`.
> 5. Reply with one line per event: id, detection_ok, the confirmed issues.

## Coach notes format (`coach_notes.md`)

Write for the player, in the user's language, plain words, no metric keys:

```
## Tổng quan            (1–2 sentences: what the session shows, data-quality caveats)
## Ưu tiên sửa          (max 3 issues, most impactful first; each: what you see → why it matters → 1 cue → 1 drill; cite the shots, e.g. "shot_02, shot_05")
## Điểm tốt             (1–3 things to keep)
## Lần quay sau         (only if filming limited the analysis)
```

Rank by: issue confirmed in reviews × share of shots affected × impact (leg power and release timing > elbow alignment > follow-through cosmetics). Drills and cues: `references/shooting-form.md`, `references/dribbling.md`.

## Honesty rules

- Only report issues backed by a metric with `reliable: true` **and** not rejected in review, or by an analyst observation. A `reliable: false` metric may only be mentioned as approximate.
- Pass every `warnings` entry from `segment`/`aggregate` on to the user (e.g. ball seen in 12% of frames → no arc/launch-angle feedback).
- If a metric is `n/a`, say it was not measurable, never estimate it from the image.
- Camera view is decided per event (`metrics.json.info.view`). From behind (`back`), knee bend, elbow at set point and launch angle are unreliable and the shooting hand is `unknown`; do not report them.
- Values in torso units use each event's own torso length, but stay approximate (±15%) when the player moves toward or away from the camera.
- Thresholds in `references/rules.json` are coaching heuristics for amateurs, not biomechanics standards. Pain or injury questions → recommend a coach or physio.

## References

- `references/shooting-form.md` - what each shot metric means, cues and drills per fault
- `references/dribbling.md` - dribble metrics, cues, drills
- `references/review-schema.md` - `review.json` format
- `references/filming-guide.md` - how the player should film for best results (send this when data quality is poor)
- `references/rules.json` - target ranges used by the scripts (single source of truth)
