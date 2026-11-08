# Dribbling: metrics, faults, cues, drills

A dribble event is a run of at least 4 bounces with the ball below the hip and less than 1.2 s between bounces. Key frames: start, middle bounce, a frame just after it (ball near the top), end.

| Key | What it measures |
|---|---|
| `frequency_hz` | bounces per second |
| `ball_peak_over_hip_torso` | median highest ball point between bounces vs hip. 0 = at hip, positive = above |
| `knee_flex_mean_deg` | average knee angle during the sequence (athletic stance) |
| `trunk_lean_mean_deg` | average forward lean |

Info fields: `bounces`, `hand_split` (bounces where the ball top was nearest the left / right wrist), `hand_switches` (crossovers, between-the-legs or other hand changes).

## Faults → cue → drill

**Dribbling too high** (`ball_peak_over_hip_torso` high)
- Cue: "Pound it below the waist".
- Drills: pound dribble 30 s per hand at knee, waist, hip height; chair-height dribbles.

**Slow, floaty dribble** (`frequency_hz` low)
- Cue: "Push the ball into the floor".
- Drills: speed dribble count (max bounces in 10 s); two-ball same-time dribbles.

**Standing tall** (`knee_flex_mean_deg` high, `trunk_lean_mean_deg` low)
- Cue: "Sit in your stance, chest up".
- Drills: defensive-stance dribble walks; low-stance figure-8 around cones.

**Bent over the ball** (`trunk_lean_mean_deg` high)
- Cue: "Eyes up, chest up".
- Drills: call out a partner's raised fingers while dribbling; dribble with eyes on a target on the wall.

**One-hand dependent** (`hand_split` heavily one side)
- Cue: train the weak hand on purpose.
- Drills: weak-hand-only stationary series (pound, front V, side V); full-court weak-hand lay-ups.

## What only the image shows

- Eyes / head down looking at the ball
- Slapping the ball with the palm instead of finger pads
- Ball crossing high and wide in front of the body (easy steal)
- Off arm not protecting the ball
