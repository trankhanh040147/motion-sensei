# Shooting form: metrics, faults, cues, drills

Key frames in `keyframes.jpg`, left to right:

1. **load**: deepest knee bend before the shot
2. **set point**: smallest elbow angle with the wrist above the shoulder (ball loaded)
3. **release**: last frame the ball is in the hand (or highest wrist if the ball is not tracked)
4. **follow_through**: about 0.33 s after release

Target ranges live in `rules.json`. "torso" = shoulder-to-hip length of the player in the image, so values do not depend on distance to the camera.

## Metrics

| Key | What it measures | Best view |
|---|---|---|
| `knee_flex_min_deg` | hip–knee–ankle angle at the dip. 180 = straight leg | side |
| `elbow_set_deg` | shoulder–elbow–wrist at set point. ~90 = the "L" | side |
| `elbow_release_deg` | arm extension at release | any |
| `release_height_torso` | wrist height above the nose at release | any |
| `dip_to_release_s` | time from deepest dip to release: rhythm / one motion | any |
| `release_vs_apex_s` | release time minus top-of-jump time. Negative = released on the way up | any, jump shots only |
| `trunk_lean_release_deg` | torso tilt from vertical at release | any (side = forward/back, front = sideways) |
| `landing_drift_torso` | hip travel from dip to landing | side (forward/back), front (sideways) |
| `follow_through_s` | how long the arm stays up and extended after release | any |
| `elbow_alignment_ratio` | elbow vs wrist horizontal offset at set point, per shoulder width | front only |
| `launch_angle_deg` | ball direction in the first 0.2 s after release | side, ball tracked |

From behind the player (`view: back`) the geometry matches a front view for elbow alignment, drift, lean, release height and timing, but knee bend, elbow at set point and launch angle cannot be read, and left/right hands are not reliable.

## Faults → cue → drill

**Not enough legs** (`knee_flex_min_deg` high, short rim misses)
- Cue: "Sit, then shoot up". Power starts in the legs.
- Drills: form shooting close to the rim with deliberate dip; "1-2 step into shot"; wall sits before shooting sets.

**Dipping too deep / slow release** (`knee_flex_min_deg` low, `dip_to_release_s` high)
- Cue: "Quick dip, one motion".
- Drills: catch-and-shoot with a count ("catch-one-up"); shots against a closeout timer.

**Ball behind the head** (`elbow_set_deg` low)
- Cue: "See the rim under the ball", set point at forehead.
- Drills: one-hand form shooting from 1–2 m; mirror holds at set point.

**Pushing / short-arming** (`elbow_set_deg` high or `elbow_release_deg` low)
- Cue: "Reach into the cookie jar", finish with the elbow above the eyebrow.
- Drills: one-hand flick shots lying on the back (ball straight up); freeze follow-through 2 s.

**Low release** (`release_height_torso` low)
- Cue: "Shoot up, not out".
- Drills: shoot over a raised hand / broom held by a partner; "Mikan" shots focusing on high finish.

**Late release (on the way down)** (`release_vs_apex_s` high)
- Cue: "Release at the top".
- Drills: jump-and-release at the top with a soft ball; reduce jump height first, then add it back.

**Leaning / drifting** (`trunk_lean_release_deg`, `landing_drift_torso`)
- Cue: "Land on the same spot", "nose over toes".
- Drills: shoot from a line or tape cross and land on it; balance hold 1 s on landing.

**Elbow flare** (`elbow_alignment_ratio`, front view)
- Cue: "Elbow under the ball", align shooting foot/knee/elbow/wrist.
- Drills: wall shooting (shooting side close to the wall); one-hand form shooting along a floor line.

**Short follow-through** (`follow_through_s` low)
- Cue: "Hold the gooseneck until the ball hits the rim".
- Drills: freeze follow-through on every make in warm-up.

**Flat / very high arc** (`launch_angle_deg`)
- Cue: flat → "up and over the front rim"; high → "shoot through the rim, not at the ceiling".
- Drills: arc target (partner's hand above the rim line); swish-only sets from the free-throw line.

## What only the image shows (for observations)

- Guide hand pushing / thumb flick at release
- Ball on the palm instead of the finger pads
- Shooting-side foot pointed far away from the rim
- Head moving back or eyes dropping to the ball
- Thumbs-down follow-through (rotating wrist)
