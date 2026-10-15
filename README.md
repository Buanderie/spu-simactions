# spu-simactions

A sandbox for finding flight algorithms **before** they go into
[swarmpu](https://github.com/Buanderie/swarmpu). Nothing here ships. The point is to
measure a candidate against a real flight stack, plot it, and be able to say honestly how
well it works and where it breaks.

## The problem it was opened for

> reach a (moving or static) geographic point **from above, at a specific angle, at a
> specific speed**, as close to it as possible.

That is a terminal-guidance problem, not a waypoint problem. Flying *to* a point is
solved; arriving *through* it, on a specific line, at a specific angle and speed, is a
tangency condition, and the two need different mathematics.

## What is here

| | |
|---|---|
| `flyby/guidance.py` | the algorithm. Geometry only: state in, velocity command out. No MAVLink, no I/O. |
| `flyby/score.py` | the gate: where the pass line is crossed, and what the angle / speed / miss were. |
| `flyby/scenarios.py` | the experiment design, incl. a `Builder` so a new pursuit case is four lines. |
| `flyby/mavlink_link.py` | the same law flown by PX4 instead of a reference plant. |
| `flyby/experiment.py` | runs a scenario offline or live, identically. |
| `flyby/airframes.py` | per-aircraft capability (speeds, rates, bank, response) with provenance; builds the guidance and plant configs. |
| `flyby/stream.py` | appends telemetry as JSON lines while a run is in progress. |
| `scripts/bench.py` | randomised benchmark: what fraction of passes the law flies, per suite and per aircraft (`--fleet`). |
| `scripts/live3d.py` | live 3-D viewer for a stream file; follows a run or a streamed benchmark. |
| `viz.py` | figures, plus a self-contained **3-D playback** (orbit / play / scrub, no deps). |
| `scripts/rig.sh` | starts swarmpu-simuav's FDM + a PX4 SITL natively — no Docker. |
| `docs/FINDINGS.md` | what was measured, and the things that turned out to be traps. |

## Running it

```bash
# offline: the whole scenario suite against the reference plant
python3 scripts/offline_sweep.py --suite ground          # the real use case
python3 scripts/offline_sweep.py --suite level           # fly-through a point in the air
python3 scripts/report.py --suite ground --airframe quad --out runs/offline-ground-quad # figures + 3-D players + index.html
python3 scripts/lead_sweep.py                            # the one number tuned by search
python3 scripts/bench.py --suite final --n 300           # randomised benchmark
python3 scripts/bench.py --fleet --suite final --stream runs/bench-stream.jsonl &
python3 scripts/live3d.py runs/bench-stream.jsonl        # ...and watch it at http://127.0.0.1:8770/

# live: PX4 SITL flying its real stack against swarmpu-simuav's FDM
./scripts/rig.sh start
python3 scripts/fly_live.py --suite ground --scenario ground_on_slope --map
./scripts/rig.sh stop
```

Open `runs/offline-quad/index.html`, or any `*-3d.html` in a browser — they are single
files, nothing is fetched.

## The law, in one paragraph

Work in the **pass frame**: `d` is the required track direction, `r` is 90° clockwise from
it. The relative position is `s` (along-track, negative before the point) and `e`
(cross-track, positive right of the line). Every tick the law regenerates a **cubic
Hermite on absolute position** from the measured state `(p, v)` to the point's *predicted*
position at the arrival instant, with the *required* velocity as the terminal condition —
speed along `d`, no lateral rate, no climb. Reading it a short lead ahead is the lead
compensation, and because the profile is regenerated from the live state every tick, wind,
model error and a moving point all show up in the state and are absorbed. A ground strike
is the same law with a demanded **flight-path angle** (so the vertical channel is a
descent rate proportional to the *current* speed) and a **speed schedule** that ramps from
the approach speed down to the airframe's minimum by the gate.

Two things make it work that are easy to get wrong, both measured here:

* **Everything is absolute.** The requirement is about the aircraft; the error you measure
  is relative to a point that may be moving. Planning in one frame and commanding in the
  other needs a conversion at the boundary, and missing it is silent until the point moves.
* **The lead is the vehicle's lag and nothing else.** Too short and the aircraft arrives
  late with the residual rate still on it; too long and it arrives *early*, crossing the
  line before the gate with the residual rate on it. What the vehicle cannot fly is handled
  by stretching the profile, never by reading further ahead.

## The reference plant

The offline yardstick is what says how much of a miss is guidance and how much is the
vehicle, so it has to respond the way the vehicle does:

* **Multirotor:** a first-order lag per axis on the commanded velocity (τ 0.30 s horizontal,
  0.15 s vertical, measured on the Iris). A multirotor can push in any direction, so a lag
  on the velocity vector is the right model.
* **Fixed wing:** a wing changes course only by banking, so the horizontal channel is the PX4
  loop chain instead (`PlantConfig.turn_model`): an NPFG course loop (τ = ζP/2π = 0.67 s at
  simuav's 6 s period), a roll loop (`FW_R_TC` 0.4 s) clipped at the bank limit, course rate
  g·tan(bank)/V, and speed limited by the longitudinal acceleration and never below the
  minimum speed. A small correction settles in about a second and a large turn is
  turn-rate limited, as on the aircraft. The lead the law reads ahead is the course + roll
  response (~1.07 s), not a turn time.

`docs/FINDINGS.md` has the numbers, the failure modes, and the PX4 traps.