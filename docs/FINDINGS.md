# Findings

Working notes for the terminal-guidance sandbox. Everything below was measured on this
machine; the numbers are not estimates. Command used:

```
python3 scripts/offline_sweep.py --suite {level,ground} --airframe {quad,plane}
python3 scripts/bench.py --n 400                       # randomised success ratio (§7)
python3 scripts/vstep.py [--raise]                     # vertical channel isolation (§4)
python3 scripts/fly_live.py --suite ground --scenario ground_on_slope --quiet
python3 scripts/derive_profile.py --all --write        # aircraft profiles (§9)
```

## 0. Where this stands

Two live blockers are **closed**, and they turned out to be the same bug: PX4 clamps every
axis of a velocity setpoint at `MPC_*_VEL_MAX`, silently, inside
`PositionControl::_positionControl()`. The vertical one cost 125 m of altitude on a descent;
the horizontal one cost 21 m on a 300 m offset. Neither was visible to the offline suite,
whose reference plant knew none of PX4's limits, and neither produced an error anywhere.

| live, PX4 + FDM | miss | before it was |
|---|---|---|
| `ground_on_slope` | **0.03 m** | 202 m of altitude error |
| `offset_right_300` | **0.24 m** | 21.18 m, −35° track |

**What the law is good at** (§7, 800 randomised cases): fly-throughs — 98.8 % at the gate
tolerance, and over the passes it accepts, median miss **0.000 m**. Terminal descents onto a
*static* point — 77 %, median miss **0.011 m**.

**What it is not**: a point that *moves*. 0 % above 10 % of the approach speed, and 82 %
refused even below it (§7). And the fixed wing does not transfer: the same law scores 0.15 m
on the 1.2 s trainer and **119 m** on an 11.7 s aircraft (§7). Three open items, all
quantified, none mysterious.

## 1. What the law is

`flyby/guidance.py`, `TerminalProfileGuidance`. Given a point, a pass angle, an arrival
speed and optionally a demanded flight-path angle, it commands an **absolute NED velocity**
— no position reference, so it is independent of how the vehicle is flown and portable to
any autopilot that accepts velocity setpoints.

Three design decisions carry most of the accuracy:

* **Absolute positions, not relative rates.** The lateral and vertical axes are cubic
  Hermite profiles built in the vehicle's own frame between its measured state and the
  *predicted* contact point. Regenerated from the measured state every tick and read one
  vehicle-lag ahead (`u = tau_lead / T`). Building the profile on relative error and
  velocity instead — the obvious formulation — compounds drift: it arrives late, still
  carrying residual rate, and the error grows with range.
* **Along-track is a rate, not a profile.** A cubic along the track has to satisfy
  "arrive at 25 m/s" from 21 m/s, so it brakes, the aircraft slows, the next tick is less
  feasible still, and it stalls short forever. The along axis is a **speed schedule**: a
  ramp from `approach_speed` to `speed` with a **latched epoch** (`_sched_t0 =
  2|s| / (v_start + v_pass)`), so the ramp is a function of position-in-the-approach, not
  of elapsed time.
* **The minimum arrival speed is geometric, not just an airframe limit.** For a moving
  point, arriving with no cross-track rate while the point has cross-track velocity
  `|v_tgt · r|` needs lateral rate from somewhere, and the only source is speed ×
  `sin(cross-angle)`:

  ```
  V_arrival >= |v_tgt · r| / sin(cross_angle_budget)
  ```

  Same relation as a Zero-Effort-Velocity / rendezvous constraint. Without it the law
  accepts "arrive at 3 m/s" onto a point walking at 3 m/s and then reports 34° of track
  error at the gate — 34° is the least-bad answer, and the requirement was not achievable
  at any speed. With it, `ground_target_too_fast` is reported **infeasible** instead.

## 2. Measured accuracy (offline, reference plant)

Level suite, quad: flyable cases **0.02 m** mean miss, **0.08 m** worst, **1.05°** mean
absolute track error. Plane: **1.38 m** mean, **9.10 m** worst, **4.27°** mean.

Ground suite, `ground_on_slope` (900 m out, 328 m up, 30° pass, −20° flight path, 12 → 3 m/s):

| | quad | plane |
|---|---|---|
| miss | 0.00 m | 0.04 m |
| track error | +0.00° | +0.00° |
| speed error | +0.11 m/s | +1.17 m/s |
| flight-path error | +0.77° | −0.41° |

### Scoring a ground target: the gate is the arrival, not the line

A pass line through a point at altitude is a line in a plane, so "crossed the line" is
sharp. For a point **on the ground** the same plane extends infinitely upward and the
aircraft crosses it while still short and high. The first live ground run scored a
crossing at **202 m altitude** with everything else already correct to 0.04 m and 0.5°.

So `PassSpec.gate` selects the event: `"line"` for a pass through a point at altitude,
`"closest"` for a terminal descent, where the gate is minimum 3-D distance to the point —
which for a ground target is the touchdown, and is well defined for exactly the reason the
line crossing is not. `_closest_sample` refines the minimum with a parabolic fit through
its neighbours so the reported time is not quantised to the telemetry rate.

## 3. What the live PX4 run caught that the offline suites could not

The offline suites start the pass already flying the approach at speed. A live rig hands
over at the end of a transit. Three bugs only the live run exposed:

1. **Speed-schedule re-arm mid-approach.** The re-arm heuristic was "is the point
   comfortably ahead?". It re-armed **380 m into a 900 m approach**, restarted the ramp
   from 12 m/s with a 50 s horizon, flew the second half at full speed, passed **200 m
   above the gate**, and then flew on at 3 m/s. Offline never sees it because `s` never
   enters the region the heuristic was watching. Fixed: re-arm on the first tick, on the
   tick after a completed pass (`_was_post_pass` edge), and on target motion.
2. **`pass_duration` used the arrival speed as the closing rate.** A ground approach
   arriving at 3 m/s over 900 m budgeted 420 s; the correct rate is the *approach* speed
   (and, better, the ramp average `(v_start + v_pass)/2`).
3. **The live CSV was truncated** relative to the logged run — the artifact is written only
   after `run_pass` returns, so a killed run leaves the previous CSV in place and looks
   like a successful short run. Check the sample count against the logged wall time.

## 4. Closed: the live quad's vertical axis

**This was a PX4 parameter, not a guidance bug, and no amount of tuning the law could ever
have fixed it.** `PositionControl::_positionControl()` runs on every update and
unconditionally executes

```cpp
_vel_sp(2) = math::constrain(_vel_sp(2), -_lim_vel_up, _lim_vel_down);
```

so the **vertical velocity limit is applied to a pure velocity setpoint**, not only to a
position one, and silently. The stock iris values are `MPC_Z_VEL_MAX_DN` = 1.5 m/s and
`MPC_Z_VEL_MAX_UP` = 3.0 m/s — asymmetric, and applied to a setpoint that says nothing
about either.

`scripts/vstep.py` is the experiment that settled it. It reads the limits off the vehicle,
then steps a pure vertical velocity through a ladder **at two setpoint rates**, with the
FDM's ground truth polled alongside PX4's own estimate. Measured:

| commanded `vz` (NED, +down) | stock PX4 | ground truth | limits raised |
|---|---|---|---|
| +2.0 | +2.01 | +2.01 | +2.01 |
| +4.0 | **+1.50** | **+1.50** | +4.02 |
| +6.0 | **+1.50** | **+1.50** | +6.02 |
| −4.0 (climb) | **−3.00** | **−2.99** | −3.99 |

Three candidate explanations were separated by one run each:

* **Not the tick rate.** `|achieved(100 Hz) − achieved(20 Hz)| = 0.003 m/s`. A 5× faster
  setpoint stream changes nothing, which a loop-rate problem cannot do.
* **Not the altitude estimate.** PX4's vertical velocity and the FDM's ground truth agree
  to **0.004 m/s** mean over the ladder, so the aircraft really is where PX4 says it is.
* **Not the FDM.** The plant flies +6 m/s happily once the limit allows it.
* **It is the clamp**, on both axes, with the asymmetry above.

The previous table's constant 1.5 m/s sink was this number, to three digits.

The fix is in `flyby/airframes.py` (one place states each aircraft's capability) and
`OffboardController.apply_airframe_limits`, which pushes it to PX4 **and reads it back** —
a fixed wing gets `FW_T_CLMB_MAX`/`FW_T_SINK_MAX` instead, because `MPC_Z_VEL_*` is a
multirotor parameter and writing it to a plane is a silent no-op.

Live results after both fixes, on PX4 SITL against swarmpu-simuav's FDM:

| scenario | miss | track | speed | altitude | flight path |
|---|---|---|---|---|---|
| `ground_on_slope` (descent onto a ground point) | **0.03 m** | −0.82° | −0.01 m/s | −0.0 m | −3.3° |
| `aligned` (fly-through, no offset) | **0.02 m** | +0.26° | −0.01 m/s | −0.0 m | — |
| `offset_right_300` (300 m cross-track) | **0.24 m** | +3.49° | +0.02 m/s | +0.0 m | — |
| `offset_right_800` (800 m cross-track) | 102 m | −53° | +6.0 m/s | +0.0 m | — |

The first two rows are the live equivalents of offline cases that score 0.01–0.02 m, so the
reference plant is now telling the truth. The last row is the outstanding item.

### 4a. The same bug again, on the other axis — and it was hiding behind the first

With the vertical axis fixed, a **cross-track** fly-through was run live for the first
time: `offset_right_300`. It missed by **21 m with a −35° track error**, on both the iris
and the Mavic 3 profile, while the offline sweep scored the same scenario at 0.01 m. A
second silent clamp, in the same two lines of C++:

```cpp
_vel_sp.xy() = ControlMath::constrainXY(..., _lim_vel_horizontal);
_vel_sp(2)   = math::constrain(_vel_sp(2), -_lim_vel_up, _lim_vel_down);
```

`MPC_XY_VEL_MAX` is 12 m/s on a stock iris. Nulling a 300 m offset needs more than that, and
the reference plant allows 30 m/s, so the offline suite flew a manoeuvre the vehicle never
attempted. From the live telemetry: **commanded speed peaked at 14.76 m/s, achieved never
exceeded 12.06, and the aircraft sat at exactly 12.00 m/s for the whole manoeuvre — above
the limit on 46.7 % of ticks.** After raising `MPC_XY_VEL_MAX` to the airframe's real
ceiling, same scenario, same seed:

| | miss | track | speed |
|---|---|---|---|
| before | 21.18 m | −35.17° | +0.00 m/s |
| after | **0.24 m** | **+3.49°** | +0.02 m/s |

**The unified statement, and the thing to remember: PX4 clamps *every* axis of a velocity
setpoint at `MPC_*_VEL_MAX`, unconditionally, in `_positionControl()`, and says nothing.**
Vertical, this looked like a sinking aircraft. Horizontal, it looks like a guidance law that
cannot turn. Both are the same sentence in the same function, and both were invisible to an
offline suite whose reference plant knew none of these limits.

`GuidanceConfig` now carries `max_speed_mps` — the vehicle's ceiling, as distinct from
`max_speed_overshoot`, which is how much faster than the pass the law may go to buy lateral
correction. Conflating them is what let the law ask for a manoeuvre the autopilot refused.

**Still open:** `offset_right_800` (an 800 m offset) now runs the aircraft at exactly
18.0 m/s — pinned against its new ceiling — and still misses by 102 m with −53° of track
error, where the offline plant scores 0.03 m. So for large offsets the airframe's real speed
limit genuinely binds, and the law needs to say *infeasible* there rather than attempt it.
That is the next thing to fix, and it is a tuning question, not a mystery.

### 4b. Four more bugs, all silent

Every one of these produced a wrong or meaningless number with no error anywhere. They are
listed because the shape is the lesson: **a measurement that is not read back is a hope.**

1. **`set_param` never worked.** It sent the parameter name **one character per message**.
   Measured on this rig: one `PARAM_SET` carrying `MPC_Z_VEL_MAX_DN` padded to 16 bytes
   reads back 7.5 after being set to 7.5; the same parameter set one character at a time
   does not read back at all. It went unnoticed because nothing read it back — so
   `set_pass_speed` has also never actually set `FW_AIRSPD_TRIM`, and every fixed-wing pass
   speed on this rig was whatever PX4 booted with.
2. **PX4 persists parameters across rig restarts.** It writes `parameters.bson` on shutdown
   and reloads it on the next boot, so a limit changed by one experiment is still in place
   for the next. "Run against the stock autopilot" was silently measuring against whatever
   the previous run left behind — caught when a supposedly-stock run reported 8.0 m/s.
   `scripts/rig.sh` now clears parameters at start (`RESET_PARAMS=1`).
3. **Reading a parameter while in OFFBOARD triggers a failsafe.** PX4 needs setpoints or it
   drops out after about a second, so a blocking parameter read after `prepare()` is a
   failsafe with a long fuse: the aircraft flies beautifully right up until the read
   completes, then goes RTL, lands and disarms. `get_params` takes a `keepalive` callback.
4. **The ground scorer measured a stopped aircraft.** A terminal descent is *supposed* to
   end on the ground, so the minimum-distance sample is the aircraft sitting on the target
   — and 24 s later it is at 0.18 m/s with a track angle that wanders by tens of degrees.
   Scoring there reports *"0.2 m/s instead of 3 m/s, 67° off track"* for a descent that
   landed on its target to within 6 cm. Position is now scored at the touchdown and
   kinematics at the last sample still airborne (`score.TOUCHDOWN_M = 0.5 m`).

5. **`feasible` was being overwritten, not accumulated.** The profile-feasibility check
   read `feasible = (...)` where it should have read `feasible = feasible and (...)`. Every
   verdict set before that line — the deceleration bound, the geometric intercept floor — was
   silently discarded, while the *clamps* those checks justify still happened. Nothing
   crashed and nothing warned: the law simply stopped reporting infeasible for those
   conditions. `ground_target_too_fast`, whose entire purpose is to check that a
   geometrically impossible pass is REFUSED, was scoring as flown.

   An infeasibility check whose result is discarded is worse than no check, because the
   clamp still runs and looks like the check passing. Fixed, and fixing it exposed a second
   bug immediately: the deceleration bound had never been executed, so it had never been
   correct, and on its first live run it declared the textbook ground approach infeasible in
   its final tick — it compared the *nominal* 12 m/s approach speed against the arrival speed
   over a 0.35 s horizon, by which point the aircraft had already shed the difference. It
   now uses the measured along-track rate, which is the quantity the requirement is about.

   The general lesson is the one the whole of §4b illustrates: **a check that is never run,
   or whose result is never read, is indistinguishable from a check that passes.**

One hypothesis was tested and **rejected**, and is recorded because it is easy to repeat:
the live −3.1° flight-path residual was suspected to be the `fp_alt_trim_max_mps = 0.25`
absolute cap (4.8° of angle at a 3 m/s arrival). Making the cap a fraction of pass speed
changed nothing on the textbook cases (0.01 m, −0.12° at every setting) and made the
moving-target cases monotonically worse. The residual is real plant lag, not the trim. The
cap stayed absolute, with the reasoning in `guidance.py`: the bank limit is rightly
expressed as an angle because it is a *steady-state* limit, but the trim is a *closure*
term and needs the absolute rate.

## 5. PX4 / MAVLink facts worth keeping

* OFFBOARD UDP `127.0.0.1:14540`, `udpin`; setpoints above 2 Hz and a heartbeat with
  `MAV_STATE_ACTIVE` or PX4 rejects the first setpoints.
* `MAV_CMD_DO_SET_MODE` needs `param1=1` (`CUSTOM_MODE_ENABLED`), `param2` main mode,
  `param3` sub-mode.
* Take the setpoint type masks from the pymavlink constants — `VELOCITY_ONLY=3527`,
  `POSITION_AND_VELOCITY=3520`, `POSITION_ONLY=3576` — and **do not** set `FORCE_SET`.
* `POSITION_AND_VELOCITY` honours both fields; passing `x=y=0` pins the aircraft at the NED
  origin. Use pure velocity for the law and transit, position only for climb-out.
* `recv_match(blocking=True, timeout=0)` returns immediately without reading. Telemetry
  freezes unless the receive is genuinely nonblocking.
* Fixed-wing OFFBOARD does not navigate toward velocity/bearing setpoints on a stock
  build; `swarmpu-simuav/extdeps/patches/0001-fixed-wing-offboard-navigation.patch` adds
  it. No patched binary is built in this sandbox, so plane results are offline-only.
* **A parameter write is not done until it is read back.** One `PARAM_SET` with the whole
  16-byte name; never one message per character (§4b).
* **PX4 persists `parameters.bson` across restarts.** Any rig that boots PX4 in a reused
  working directory is measuring the previous run's parameters unless it clears them.
* **Anything that blocks in OFFBOARD is a failsafe.** `get_params` takes a `keepalive`.
* The vertical velocity limit is enforced on velocity setpoints, silently. Read it before
  planning a descent (§4).
* **So is the horizontal one**, at `MPC_XY_VEL_MAX`, in the same statement. Every axis of a
  velocity setpoint is clamped without a word (§4a).
* Fixed wing has no OFFBOARD airspeed control; `FW_AIRSPD_TRIM` is the nearest equivalent.
* `MPC_Z_VEL_*` is a multirotor parameter. Writing it to a fixed wing is a no-op, not an
  error.

## 6. Visualisation

`viz.py` emits static plots, a GIF and a **self-contained 3-D HTML player**
(`write_3d_player`) — a canvas projection with play/pause/scrub, no CDN and no build step.
`matplotlib`'s 3-D projection is broken in this environment (mixed installs) and `plotly`
is absent, so the player exists because nothing else does. All 18 generated players pass a
Node-based 40-frame render check.

## 7. Randomised benchmark: what fraction of passes does this law actually fly?

Everything measured before this section is a handful of hand-picked scenarios. That is the
right way to develop and the wrong way to report — a suite is chosen by the person who built
the law, so it over-represents the cases that author understood. `scripts/bench.py` samples
the geometry instead: random point, random start, random bearing, random speeds, seeded.
800 cases, iris reference plant, `python3 scripts/bench.py --n 400`.

**Two numbers, never averaged.** A pass can fail by being *refused* — the law reported
INFEASIBLE for a requirement no aircraft could meet, which is the system working — or by
being *missed*, the law said yes and then did not. Counting a refusal as a failure measures
nothing.

| family | strict | **gate** | loose | ballpark | refused | missed |
|---|---|---|---|---|---|---|
| level fly-through (400) | 98.2 % | **98.8 %** | 98.8 % | 99.0 % | 1.8 % | 0.8 % |
| terminal descent (400) | 40.2 % | **52.0 %** | 61.2 % | 76.5 % | 49.2 % | **1.2 %** |
| overall (800) | 69.2 % | **75.4 %** | 80.0 % | 87.8 % | 25.5 % | 1.0 % |

Tolerances: `strict` = 1 m / 1.5° / 0.20 m/s (+1.5 m, 1.5° flight path on a descent);
`gate` = 3 m / 3° / 0.50 m/s; `loose` = 10 m / 8° / 1.5 m/s; `ballpark` = 50 m / 25° / 5 m/s.

**And over the cases it accepts, it is essentially exact.** Statistics below are over
*accepted* cases only:

| | n | miss median | miss p95 | miss worst | track median | speed median |
|---|---|---|---|---|---|---|
| level | 393 | **0.000 m** | 0.003 m | 2.67 m | 0.017° | 0.000 m/s |
| descent | 203 | **0.011 m** | 0.031 m | 0.11 m | 0.037° | 0.000 m/s |

Descent altitude error: median **0.008 m**, worst 0.095 m. Flight-path error: median 0.91°,
worst 3.22°.

So the 52 % on descents is not "half of them are flown badly" — it is *half of them are
declined*. Of the 400 random descents, **197 were refused and only 5 actually missed**, and
the ones it did accept landed within 11 cm. That is the property worth having: the law
either flies the pass or says it cannot, and the benchmark's job is to say which and how
often.

### Does it work across the fleet? One number, and it is the turn response

> **Superseded for fixed wings by §12.** The τ values below for planes came from a wrong
> derivation and a plant that applied them as a lag on every command. The multirotor rows
> are unaffected.

`bench.py --fleet` holds the law, the seed and the case distribution fixed and changes
**only the aircraft**. 240 cases each (120 fly-throughs + 120 descents), all 14 profiles:

| airframe | kind | τ_horiz | min speed | refused | **strict** | gate | accepted mean miss | worst |
|---|---|---|---|---|---|---|---|---|
| matrix350rtk | quad | 0.30 s | 1.5 | 23.3 % | **70.0 %** | 76.3 % | **0.014 m** | 1.60 m |
| mavic2ent | quad | 0.30 s | 1.5 | 23.3 % | **70.0 %** | 76.3 % | 0.014 m | 1.60 m |
| mavic3 | quad | 0.30 s | 1.5 | 23.3 % | **70.0 %** | 76.3 % | 0.014 m | 1.60 m |
| mini4pro | quad | 0.30 s | 1.5 | 22.9 % | **70.0 %** | 76.3 % | 0.026 m | 2.23 m |
| iris | quad | 0.30 s | 1.5 | 29.2 % | 69.2 % | 75.0 % | 0.022 m | 2.73 m |
| trainer | plane | 1.20 s | 11.0 | 55.8 % | 49.6 % | 57.5 % | 0.136 m | 1.42 m |
| switchblade300 | plane | 7.08 s | 18.8 | 77.9 % | 1.3 % | 7.1 % | 12.36 m | 60.7 m |
| switchblade600 | plane | 7.44 s | 18.8 | 78.8 % | 0.8 % | 6.3 % | 12.70 m | 66.4 m |
| geran2 | plane | 7.49 s | 21.3 | 78.3 % | 0.4 % | 4.6 % | 18.92 m | 75.9 m |
| shahed136 | plane | 7.49 s | 21.3 | 78.3 % | 0.4 % | 4.6 % | 18.92 m | 75.9 m |
| bayraktar_tb2 | plane | 9.77 s | 31.3 | 73.8 % | 0.4 % | 1.3 % | 77.40 m | 265.9 m |
| tb3 | plane | 10.19 s | 32.5 | 75.4 % | 0.4 % | 1.3 % | 89.75 m | 231.3 m |
| akinci | plane | 11.65 s | 33.8 | 66.7 % | **0.0 %** | 1.3 % | **114.46 m** | 285.7 m |
| mq1_predator | plane | 11.80 s | 27.5 | 81.2 % | 0.4 % | 0.4 % | 83.92 m | 222.3 m |

(70 % rather than the 100 % of §7's level-only table: this run mixes in the 120 descents,
which is where moving-point refusals live.)

**A caveat that matters more than the table: the multirotor rows are identical by
construction, not by measurement.** `mavic3`, `mavic2ent` and `matrix350rtk` return
*bit-identical* numbers — 70.0 %, 0.014 m, worst 1.60 m — because the derived capability
collapses them onto the same point: τ = 0.30 s (carried over from the measured Iris as the
multirotor class value) and a speed envelope of 1.5 / 15 m/s, so they generate an identical
case set and the law sees an identical vehicle. `mini4pro` differs only because a 249 kg
quadrotor gets a higher thrust-to-weight ceiling.

So "the law transfers across multirotors" is **a statement about the capability model being
uniform across multirotors**, which is at least consistent with the live evidence (the Iris
and the Mavic 3 fly to 0.01–0.03 m on the same scenarios), but it is not evidence about
those three aircraft individually. Per-rotor time constants are not derivable from published
mass and wheelbase — inertia was the same problem (§11) — so distinguishing a Mini from a
Matrice needs them measured, and `scripts/vstep.py` already measures exactly this for
whatever profile it is pointed at. That is the next measurement worth making, and it is a
short one: point `vstep.py` at each profile and read the time constant off the vehicle.

**Read the table down the `τ_horiz` column and the rest is monotone.** Five
multirotors spanning a 6.5× mass range (249 g to 6.5 kg) all land within half a percentage
point of each other on **strict** and within 12 mm on mean miss — mass, thrust and size are
irrelevant. Then one fixed wing at 1.2 s is still fine (0.14 m), and everything past ~7 s
falls off a cliff to 114 m. The four aircraft with the slowest turn response are also the
four worst, in order, with no exceptions in 14 rows.

The mechanism is the read-ahead. `tau_lead` tracks the aircraft's turn response, which is
the right idea, but a lead that is correct *in absolute time* is not correct *relative to the
approach*: on a 900 m approach at 60 m/s there are ~15 s to run, and leading the profile by
11.65 s means reading most of the way to the gate on the first tick, on a profile that is
then rebuilt underneath the aircraft from a state it has not begun to approach.

So the honest statement is narrower than "the law does not transfer to the fixed wing": the
law transfers to every airframe whose turn response is under about 2 s, and does not
transfer to any faster one, and **1.2 s was not a discovered value — it is the class
default the trainer happened to agree with.** The fix is either a fraction of `t_go` rather
than an absolute time, or dropping the lead on the along-track axis, which is a rate command
and needs no read-ahead. Neither is tried yet; §10 notes this has to be solved before any
fixed-wing pass on PX4 is worth scoring at all.

### The case that actually matters: a final approach

Everything above mixes in hard INTERCEPTS — lateral deviation up to ±350 m over a 300 m
range, which at the short end is a 49° track error. That is not an approach, it is a
capture, and a law can look mediocre on it while being fine at the thing it is for.
`--suite final` samples the real thing: 600–3000 m stabilised leg, lateral deviation
*proportional to range* (±8 %, so a localizer intercept closes as the range closes), ±15 m
on the glideslope rather than ±40 m.

**The same law, on the same seed, same 14 aircraft:**

| airframe | τ_horiz | refused | **strict** | gate | accepted mean miss |
|---|---|---|---|---|---|
| matrix350rtk / mavic2ent / mavic3 / mini4pro | 0.30 s | 20.0 % | **75.8 %** | 82.1 % | **0.023 m** |
| iris | 0.30 s | 20.8 % | 74.2 % | 80.8 % | 0.030 m |
| trainer | 1.20 s | 29.2 % | **71.7 %** | 78.3 % | 0.099 m |
| switchblade300 | 7.08 s | 51.7 % | **50.8 %** | 62.5 % | 0.486 m |
| switchblade600 | 7.44 s | 55.0 % | 49.2 % | 61.7 % | 0.559 m |
| geran2 / shahed136 | 7.49 s | 56.7 % | **45.0 %** | 57.5 % | **0.760 m** |
| bayraktar_tb2 | 9.77 s | 58.3 % | 7.5 % | 20.0 % | 4.54 m |
| tb3 | 10.19 s | 59.2 % | 5.8 % | 16.7 % | 5.76 m |
| akinci | 11.65 s | 66.7 % | 5.0 % | 15.8 % | 6.61 m |
| mq1_predator | 11.80 s | 68.3 % | 5.8 % | 15.8 % | 5.38 m |

The Geran goes from **0.4 % strict / 18.9 m** on the mixed suite to **45.0 % / 0.76 m** on a
final approach. So most of the alarming fleet table above was the benchmark measuring the
wrong thing, and the honest summary of "how good is this law at air-to-ground" is the
second table. The ordering in τ still holds, so the turn-response problem is real — it is
just an order of magnitude less severe than the intercept suite made it look.

### Capping the read-ahead: helps, but only where the read-ahead is longer than the approach

`tau_lead = min(vehicle τ, frac · t_go)`, measured on the final-approach family, `frac=0.5`:

| airframe | τ_horiz | strict, uncapped | strict, capped | mean miss, uncapped → capped |
|---|---|---|---|---|
| iris | 0.30 s | 74.2 % | 73.3 % | 0.030 → 0.023 m |
| trainer | 1.20 s | 71.7 % | 71.7 % | 0.099 → **0.047 m** |
| switchblade300 | 7.08 s | **50.8 %** | 38.3 % | 0.486 → 0.759 m |
| geran2 | 7.49 s | **45.0 %** | 32.5 % | 0.760 → 0.980 m |
| bayraktar_tb2 | 9.77 s | 7.5 % | **17.5 %** | 4.54 → **1.95 m** |
| tb3 | 10.19 s | 5.8 % | **13.3 %** | 5.76 → **2.13 m** |
| akinci | 11.65 s | 5.0 % | **11.7 %** | 6.61 → **2.28 m** |
| mq1_predator | 11.80 s | 5.8 % | **10.8 %** | 5.38 → **2.21 m** |

**It splits by regime, and the split is the answer.** The cap roughly *doubles* the four
slowest aircraft (5.0 % → 11.7 % on the Akıncı, miss 2.9× better) and roughly *halves* the
two at 7–7.5 s (Geran 45.0 % → 32.5 %). That is exactly what "cap it when the lead is longer
than the approach" predicts: at τ ≈ 7 s on a 600–3000 m final approach the lead is already
shorter than the approach and the cap only removes feedforward the aircraft needed.

So on the mixed suite, where 0.5 looked like a clean 1.3–2.4× win, that was an average over
the two regimes weighted by how bad the slow ones were. **For the stated use case — a Geran
onto a final approach — the cap is the wrong change**, and this measurement is why: I was
about to make it the default off a result from the suite that models the task least well.

The general rule this suggests is to cap only when the uncapped lead exceeds the approach,
which is a comparison the law can make itself rather than a constant anyone has to pick.
Not implemented — that needs its own measurement.

### Does the ZEV quartic actually predict a miss? No. Measured.

§1 asks why not simply use ZEV/ZEM, since the terminal condition already *is* a generalised
ZEV rendezvous. That is a fair question and it deserved an experiment rather than an
argument, so `zev_rendezvous()` is implemented (`guidance.py`), wired to
`use_zev_feasibility` / `use_zev_tgo`, and `scripts/feasibility_ab.py` measures it against
the current test on the same 250 random final approaches.

Both tests answer "is this pass achievable", and both say `infeasible` a lot, which is worth
nothing on its own — a test that refuses everything refuses nothing. So the comparison is
whether the verdict tracks the outcome. Of 250 cases, 58 really missed:

| test | refused | caught the misses | false refusals |
|---|---|---|---|
| current (geometric floor) | 51 | **43 (74 %)** | 8 (16 % of its refusals) |
| ZEV quartic root test | 1 | **1 (2 %)** | 0 |

**The ZEV test loses, decisively**, and the reason is not that it is wrong. It models a
double integrator with no acceleration bound, no speed limit, no bank limit, no climb-rate
ceiling and no lag — and on this case distribution it answers "yes, achievable" for
essentially everything that is still 50 % of the way down the approach. The passes that
really miss miss for reasons ZEV has no term for.

Two implementation notes that are not obvious:

* **The test is degenerate at the gate.** It is a statement about the problem from the
  *current* state, and at the gate the current state is the terminal state: Δr → 0 makes
  a₂ → 0, so the condition `a₂ < 0` fails and *every* approach reports infeasible at its own
  gate, including the ones that flew to 0.01 m. Reading it there is not conservative, it is
  meaningless. It has to be read early — which is why `feasibility_ab.py` reads it at 50 %
  of time-to-go, and why any implementation has to make that choice explicitly.
* **ZEV's terminal constraint is stricter than a landing needs.** It fixes position AND
  velocity exactly. A terminal descent does not have to arrive with zero cross-track rate at
  the instant of touchdown; the aircraft is about to stop anyway.

What is left standing from §8 is the *explanation*, not the test: our law is in the ZEV
problem class, our cubic is the minimum-acceleration (min-snap m = 3) segment given zero
endpoint accelerations, and the speed schedule exists because the unconstrained optimum goes
non-monotone. Adopting ZEV's acceleration law would trade a velocity interface we can fly
and a feasibility verdict that catches 74 % of misses for one that catches 2 %.

### Where the ground family fails

| | n | refused | gate-success |
|---|---|---|---|
| static point | 268 | 25.7 % | **77.2 %** |
| walking < 10 % of approach speed | 17 | 82.4 % | 5.9 % |
| walking 10–20 % | 20 | **100 %** | 0 % |
| walking > 20 % | 95 | ~99 % | 0 % |

**The dominant open problem is a slowly-moving ground point.** At < 10 % of the approach
speed — a point ambling at 0.5 m/s while the aircraft does 12 — the law still refuses 82 %
of the time, and that cannot be the geometric floor doing its job: the floor is
`|v_target·n̂|/sin σ`, and 0.5/0.707 = 0.7 m/s is far below the iris's 1.5 m/s minimum.
So this is a real weakness in the intercept, not a correct refusal, and it is the first
thing to work on. §8 names the published test that should replace the geometric bound.

Secondary, and much smaller:

* **Range is the strongest predictor.** 23 % gate-success inside 400 m against 65 % beyond
  800 m — the opposite of intuition, because a long approach gives the profiles more time
  and a short one gives them almost none.
* **Starting far off the glide is bad in both directions**: 33 % starting 20 m *below*, 40 %
  starting 20 m+ *above*, against 62–74 % when within 20 m. A vertical profile has to dive
  and come back, and there is not enough vertical acceleration budget to absorb a large
  initial altitude error while holding a demanded flight-path angle.
* **Lateral offset barely matters** (47–55 % across 0–400 m), which is what the reference
  plant's bank-limited lateral acceleration is sized for.

### Three lessons about benchmarking, all from getting it wrong first

1. **Sample inside the airframe's envelope.** The first version drew speeds and angles
   independently and asked an iris-class quadrotor to start approaches at 29 m/s (ceiling
   15) and demand 9.9 m/s descents (ceiling 6). 88 % of the ground suite came back
   "infeasible" and the honest answer was a handful of cases. `sample_speeds` now clips to
   the capability and *reports* how many draws it clipped (4 % in the run above).
2. **Get the NED sign right.** The first ground sampler placed every aircraft 327 m
   *underground*, because NED z is down-positive and the point is at z = 0. All 100 %
   infeasible, which looks exactly like a broken law and is not one.
3. **Compute accuracy over accepted cases, and label the sample size.** The first version
   averaged the miss distance over *all* cases including refused ones, and printed the
   sentence "over the 203 cases that were flown" beside it. The two sets are different, and
   the effect was to report a **mean miss of 7.17 m** for the ground suite when the figure
   over the cases actually flown is **0.011 m** — a 650× error in the direction that makes
   the law look bad, produced entirely by counting the cases it had already rejected. The
   `--worst` figures had the same disease and were drawing refused cases as "worst", which
   implies it flew them badly when what it did was decline.

## 8. Literature

Positioning of this law, with verification status, because the claims below are used to
decide what to build next and a fabricated DOI would be worse than a gap.

**[V]** = verified against Crossref/OpenAlex/arXiv this session; **[V-meta]** = title,
venue and DOI verified but author list not; **[U]** = could not locate, do not print a
number next to it.

### The one result that matters most here

**de Paiva, Carvalho & Rodrigues, "UAV Optimal Guidance in Wind Fields Using ZEM/ZEV with
Generalized Performance Index", arXiv:2006.09614 (2020) [V]** is the same problem as this
one, stated in closed form. Its law is

    u*(t_go) = (6/t_go²)·ZEM − (2/t_go)·ZEV

with the optimal final time the positive root of a **depressed quartic**
`P(t_f) = C_I t_f⁴ − a₁ t_f² − 2a₂ t_f − 3a₃ = 0`, where `a₂ = 6 Δr·(v_rel⁰ + v_rel^f)`
and `a₃ = 6‖Δr‖²`. Setting `v_rel^f = V_arrival·r̂ − v_target` makes our terminal condition
a **generalised ZEV rendezvous** exactly. Two consequences:

* **Our along-track stalling has a published name.** Their Table I shows that when the
  final time exceeds the free-flight reference by more than 3×, the optimal position stays
  cubic but the **velocity reverses** — "a reversal movement in the x-axis will occur with
  a change in the sign of the velocity curve". Read one vehicle-lag ahead you never see the
  reversal, so the aircraft brakes toward a contact point it will never reach, the next tick
  is less feasible still, and it stalls short forever. That is precisely what we measured,
  and it confirms the speed schedule is the right object rather than a workaround.
* **The feasibility test should be a root-existence test, not a geometric inequality.** For
  the unpenalised case (`C_I = 0`) the quartic degenerates to a quadratic, and it has a
  positive root **iff** `a₂ < 0` and `a₂² ≥ 3a₁a₃` — i.e. iff
  `Δr·(v_rel⁰ + v_rel^f) < 0`. Read out: *the pass is feasible iff the current relative
  position and the sum of the initial and terminal relative velocities point at each
  other.* That is sharper than our test and it **names** the failure mode, which is exactly
  what section 7's moving-point cliff needs. This is the first thing to try.

### The feasibility formula: checked, and correct

The review flagged a likely component swap in our intercept floor. **Checked against the
code and it is not there**: `guidance.py` computes
`abs(tgt_vel · r) / sin(cross_budget)` where `r = geo.right_unit(angle)` is the *cross-track*
unit vector, giving `V ≥ |v_target·n̂|/sin σ` — the correct form. Recorded so nobody spends
a day "fixing" it.

Also noted, and consistent with our own notes: the ZEV *capture* condition is quadratic in
the velocity error and **involves the acceleration budget**, `|e_θ| ≤ sqrt(2·a_N·R_cross)`.
The linear form sometimes quoted cannot be a feasibility condition because it contains no
acceleration term. Not something we rely on.

### The cubic Hermite, confirmed and bounded

**Mellinger & Kumar, ICRA 2011, DOI 10.1109/ICRA.2011.5980409 [V].** A cubic Hermite with
matched endpoint positions *and* zero endpoint accelerations **is** the minimum-acceleration
segment — the `m = 3` case of the minimum-snap family. Since a velocity-tracking inner loop
has near-zero endpoint acceleration, our lateral and vertical profiles are minimum-
acceleration segments, which is a stronger justification than "it works".

But **a cubic is not time-optimal** (bang-bang is, under real bounds), and the unbounded
`C_I → ∞` limit is exactly the `K > 3` reversal branch above. Minimum-snap belongs on the
approach corridor, not the terminal gate.

**Zhao & Tsiotras, JGCD 2013, DOI 10.2514/1.57471 [V]** is the canonical reference for the
path + speed-profile decomposition, and it makes the curvature correction explicit:
`a_max,eff(s) = a_max − ṡ(s)²·κ(s)`. We do not currently budget for that.

### Rejected, with reasons

* **CLF-CBF-QP as the terminal layer.** CLF gives asymptotic set-invariance, not arrival at a
  point at a time — a category mismatch. Worse, the undesirable-equilibrium literature
  ([V-meta] *IEEE LCSYS* 2020, DOI 10.1109/LCSYS.2020.3004797; *Automatica* 2023,
  DOI 10.1016/j.automatica.2023.111359) documents QPs freezing at a wrong equilibrium, which
  is the same failure we already fight in a simpler form. The base reference is Ames, Xu,
  Grizzle & Tabuada, *IEEE TAC* 2017, DOI 10.1109/TAC.2016.2638961 [V] — Ghaffari and Xiao
  are on *other* CLF-CBF papers, not this one.
* **A learned terminal-guidance map.** Replaces an explicit INFEASIBLE verdict with a
  *sampled* region, which is the one asset section 7 shows the law actually has.
* **MPPI/MPC as the first move.** It is the right *family* for target-model error — our
  failure is model-error-dominated — but every MPC formulation needs a target state
  estimator emitting a velocity **with covariance**, and this rig has none: the benchmark
  plants the point as truth. Adopting MPPI before the estimator would replace a closed-form
  law that can prove infeasibility with one that cannot. Closest verified prior art:
  Çetin & Koyuncu, CoDIT 2023, DOI 10.1109/CODIT58514.2023.10284484 [V] (quadrotor
  interception with an impact-angle constraint).

### Citations corrected

| cited as | actually |
|---|---|
| Shakhat et al., rendezvous guidance for manoeuvring targets | **[U]** — no record in Crossref/OpenAlex/arXiv |
| Lygeros & Datta, optimality of ZEV | **[U]**; the citable result is de Paiva et al. above |
| Sussmann, time-optimal feedback guidance | **[U]** |
| De Wachter, time-optimal interpolation | **[U]** |
| Tordesillas & How, MPPI, IROS 2016 | **[U]** for a DOI (existence not in doubt) |
| Richter, Mellinger & Ware, time-optimal online replanning | Romero, Penicka & Scaramuzza, *RA-L* 2022, DOI 10.1109/LRA.2022.3185772 **[V]** |

## 10. Fixed wing on PX4: how far it actually got

Asked as "how well does a PX4 Geran, then?" — so this records what was tried, in order, and
where it stopped.

### Against the reference plant (always was measurable)

The `geran2` profile, 150 random cases per family, `bench.py --n 150 --airframe geran2`:

| family | strict | gate | loose | refused | accepted-case mean miss |
|---|---|---|---|---|---|
| level fly-through | 1.3 % | 6.7 % | 21.3 % | 53.3 % | **25.2 m** |
| terminal descent | 0.0 % | 1.3 % | 16.0 % | **97.3 %** | 5.4 m (n = 4) |

`tau_horiz` 7.49 s ⇒ `tau_lead` 7.49 s, on an approach with ~18 s of time-to-go. One of the
two worst aircraft in the fleet; only the 11.7 s Akıncı is worse. And **83 % of sampled
descents wanted a flight-path angle the aircraft cannot descend at** — a 51 m/s cruise with a
5.8 m/s ceiling holds about 6.5°. Ground interception is not in this aircraft's envelope at
all, which is correct and useless simultaneously.

### On PX4

`git apply --check` passed against this tree and `git apply --check --reverse` failed, so
the vendored `0001-fixed-wing-offboard-navigation.patch` **applies cleanly and was simply
never applied**. Applied, and `make px4_sitl_nolockstep` rebuilt (1145/1145, no errors).

`BUILD_DIR=...` turned out to be ignored by this PX4 version, so the rebuild wrote to
`build/px4_sitl_nolockstep` and **overwrote the stock binary** — the fallback I had promised
does not exist any more. The patch is guarded to fixed-wing only
(`if (vehicle_type != vehicle_status_s::VEHICLE_TYPE_FIXED_WING)`), and the quad was
re-verified afterwards on the rebuilt binary: `ground_on_slope` **0.01 m**, −0.10°,
`aligned` 0.01 m. Restoring a stock binary means reverting the patch and rebuilding.

**Progress:** stock PX4 could not arm the Geran at all. With the patch, simuav's parameter
set applied, and a fixed-wing climb-out in the driver, it **arms, launches off the catapult
at 25.9 m/s and holds OFFBOARD.** Four distinct blockers had to be cleared to get there:

1. **The driver had no fixed-wing path at all.** `climb_out` was a *position hold* at the
   aircraft's current horizontal position — asking a wing to hold a position asks it to stop,
   and a stopped wing falls. Replaced with a velocity-only climb-out that waits for the
   catapult (the FDM fires it `CATAPULT_ARM_DELAY_S` after arming, because PX4's launch
   detector has to be running first or it misses the spike), then climbs along the rail
   heading. `send_placeholder`, `transit_to`'s hardcoded ±3 m/s and `settle_velocity`'s
   zero vertical were the same assumption in three more places.
2. **`NAV_DLL_ACT = 2`, which gates ARMING.** `1030_gazebo-classic_plane` sets it, and
   `rcAndDataLinkCheck.cpp` uses `NAV_DLL_ACT > 0` to decide whether a GCS connection is
   *required to arm*. The wrong value does not cause a failsafe — it prevents the aircraft
   from ever arming, with no other symptom. Now forced to 0: this link is the ground station.
   It had to be applied **before** the arm attempt, not after; that ordering was a real bug
   and the symptom was a fixed wing that refused to arm while the quad armed every time.
3. **Integer parameters were never decoded.** `NAV_DLL_ACT` read back as `2.8e-45`. MAVLink
   types `PARAM_VALUE.param_value` as `float` in every dialect, so PX4 carries an integer by
   writing its BITS there; read as a float64 an integer comes back as a denormal whose
   mantissa is the integer. The number looked like noise and was in fact a policy decision
   that had been blocking arming. `_param_value_as_number` now converts it.
4. **simuav's plane parameters were never applied.** `trainer.json` is empty *by design* —
   a plane's PX4 parameters live in `engine/fdm/fixedwing.py`'s `PX4_PARAM_OVERRIDES` and
   simuav's loader merges them. A rig that reads only the JSON pushes nothing, so PX4 kept
   `gazebo-classic_plane`'s stock envelope (`FW_AIRSPD_MAX` 20 m/s against an airframe that
   cruises at 30). `apply_profile_params` now calls simuav's own `px4_params()` helper
   rather than reimplementing the merge, and reports 19/19 applied.

Also fixed en route: `set_params` retries a failed write as `INT32`, because an enum
parameter refuses a `PARAM_SET` that declares itself `REAL32` — which is why
`NAV_DLL_ACT = 0` initially did not take either.

### Where it stops: airspeed

PX4 reports **"Airspeed sensor failure detected"** and `VFR_HUD.airspeed = nan`, from the
moment the rail fires, with TECS having no speed to fly on. The FDM does publish
differential pressure (`0.5·ρ·V²` in `HIL_SENSOR`), so the data is there and PX4 is not
using it.

The obvious suspect is the pitot scale — `COM_ASD_SCALE`, which PX4 treats as "this tube is
unusable" at 0 — and this is exactly the parameter that is **not in this PX4 version at
all**: PX4 does not even answer a `PARAM_REQUEST_READ` for it, and neither
`DIFF_PRESS_SCALE` nor `dp_scale` appears anywhere in `src/modules/`. So the equivalent has
to be found before it can be set, and `EKF2_ASPD_MODE` / `EKF2_ASPD_FALL` / `COM_ASD_TYP`
are in the same position: requested, never answered.

A secondary "High Gyro Bias" also appears after launch, which may or may not be independent
of the airspeed problem — TECS and the estimator are both unhappy at once and it is not yet
established which is which.

**So: the Geran now arms, launches and holds OFFBOARD on a patched PX4, where before it
could do none of those. It has not yet flown a pass, and the blocker is PX4's airspeed
plumbing in this version rather than the guidance law.** The next step is to find how this
tree's `airspeed_selector` selects and calibrates its source, which is a contained piece of
work rather than a guess — and the `tau_lead` problem in §7 has to be solved anyway before
any pass this aircraft flies will be worth scoring.

## 11. The fleet

`flyby/airframes.py` states each aircraft's capability **once**, with its provenance, and
feeds three consumers that must not disagree: the guidance, the reference plant, and PX4's
own parameters.

For a profile that has no hand-written entry, the capability is **derived** from the
simuav profile's published performance rather than guessed — minimum speed 1.25× stall,
maximum speed cruise × 1.15, descent climb × 1.25, and turn response
`τ = V/(g·tan 35°)` which is the one that varies most (7.1 s for a Switchblade 300 at
48.6 m/s against 11.8 s for a Bayraktar TB2 at 67 m/s). `tau_lead_s` follows it, and its
upper clamp had to be raised from 1.2 s to 12 s, or a 92 m/s Akıncı would have been given a
lead one tenth of its turn response.

12 profiles were added to `swarmpu-simuav/config/profiles/` by
`scripts/derive_profile.py`, which derives simuav's aerodynamic coefficients from published
mass, stall speed, cruise and climb rather than having them typed in — so they are
reproducible and a changed published figure moves the coefficients with it. All carry
`confidence: published`, meaning arithmetic on published data, **not** measurements of
these aircraft.

The derivation self-validates: from MQ-1 Predator mass/stall/span it predicts
**S = 20.13 m²** against a published 20.1 m², and TB2 **AR = 13.9** against ~15.

`Orlan-10` was **rejected**. Its published mass, stall speed and span are mutually
inconsistent — they imply a wing area of 0.67 m² on a 14 m span, an aspect ratio of 292,
which no winged aircraft occupies. The script refuses to write a profile whose inputs
contradict each other rather than shipping something that flies like a different aeroplane.

```
shahed136  geran2  switchblade300  switchblade600  bayraktar_tb2  tb3  akinci  mq1_predator
mavic3  mavic2ent  mini4pro  matrix350rtk      (plus iris, trainer)
```

### The multirotor profiles only fly once they are self-consistent

`iris.json` in simuav is empty, and that is not laziness — the Iris **is** PX4's default, so
the FDM and the autopilot agree without anyone saying so. Every other aircraft has to be
declared to both halves, and getting that wrong fails in a way that looks nothing like a
model problem:

* The FDM ran a Mavic 3 profile happily at real-time factor 1.0 while PX4 **refused to arm**,
  with "Preflight Fail: Attitude failure (roll)" then "High Gyro Bias" — an EKF that cannot
  reconcile two different aircraft.
* Cause 1: the profile changed the FDM's rotor layout but not PX4's. `10015_gazebo-classic_iris`
  pins `CA_ROTOR0_PX .. CA_ROTOR3_KM` as `param set-default`, so the profiles now emit those
  plus `MPC_THR_HOVER`, and `fly_live.py` pushes and reads them back (simuav does this in its
  mavlink relay, which the rig bypasses — without this the `px4_params` block is decoration).
* Cause 2, which the first cause was masking: **the derived inertia was ~10× too large.**
  Putting all the mass at the rotors gives the Mavic 3 an Ixx of 0.30 kg m² against the
  Iris's 0.029, because a real quad's mass is mostly battery and motors, in the middle.
  Inertia is not derivable from a published mass and wheelbase, so it is now *scaled from the
  Iris* by mass and wheelbase² — 0.0178 for the Mavic 3, 0.199 for a 6.5 kg Matrice, 0.0012
  for a 249 g Mini — and rotor positions keep the Iris's "plus" proportions and signs.

With both corrected, the Mavic 3 arms, takes off, climbs out in 17.4 s and flies the pass.
The profile-parameter push initially reported every value as `None` and looked like PX4
refusing the write; it was not — `apply_profile_params` was pushing before the link had a
heartbeat, so the read-backs timed out while the writes went out fine. A premature push
produces a failure report indistinguishable from a real one, which is the §4b lesson again.

## 12. The fixed-wing plant was wrong

The fixed-wing rows in §7 were measured against a plant that does not describe a fixed wing.
`from_profile` set τ_horizontal = V / (g tan 35°), which is how long a plane takes to turn
through **one radian** at full bank, and the reference plant then applied it as a
first-order lag on **every** velocity command, however small. At 80 m/s that is 11.65 s for
a 1 m/s correction: no law closes a metre of miss through that, and no aircraft has it. The
apparent "accuracy is a monotone function of τ" in §7 was largely this artefact.

What the stack actually does, all from simuav and PX4 defaults: the velocity command goes
through the vendored offboard patch into NPFG (`NPFG_PERIOD` 6 s, which simuav applies to
**every** plane profile; `NPFG_DAMPING` 0.7). NPFG reduces to L1, whose course response is
first order with τ = ζP/2π = 0.67 s. The roll loop follows with `FW_R_TC` 0.4 s, bank clipped
at `FW_R_LIM` 35°. Course rate is g tan(bank)/V, so speed enters through the **turn rate**,
where it physically belongs, and not as a lag. Course plus roll is 1.07 s, against the
1.2 s measured on the trainer, the only plane with a measurement.

`PlantConfig.turn_model` implements that chain for planes, and `from_profile` now derives
τ_horizontal = course + roll (1.07 s) and uses the trainer's measured 1.0 s as the class
τ_vertical (the old 0.6·V/(g tan φ) had the same flaw). Step responses of the new plant:

| command | 63 % | 90 % |
|---|---|---|
| 3° course nudge at 40 m/s | 0.90 s | 1.30 s |
| 90° turn at 40 m/s | 6.25 s | 8.70 s |
| 90° turn at 80 m/s | 12.0 s | 16.9 s |

Checked for physical sanity on full benchmark runs: peak lateral acceleration stays well
inside g·tan(bank), and speeds stay inside each envelope. With the plant corrected, the
read-ahead cap of §7 and per-tick re-aiming of the pass bearing no longer change the result,
so neither was added to the law.

`bench.py --suite final_free --heading-err N` samples approaches whose crossing direction
is free and whose initial heading is within ±N° of the line of sight; the track-angle term
is not scored for that family.

**What this does not establish.** The turn plant is derived from PX4's loop constants, not
measured; the trainer agreeing to 0.13 s is one data point. Roll-rate limits, wind, NPFG's
track-error term and TECS speed/height coupling are not modelled. Separately, several simuav
profiles store a dash or maximum speed as `cruise_ms` and are flown above their published
maximum (max is derived as 1.15 × cruise). The next measurement is a heading step on the
trainer through PX4, the way `vstep.py` did the quad's vertical axis.

## 13. Viewer and streaming

* `bench.py --fleet` ignored `--stream`; it now streams every case like the single-airframe
  mode.
* `live3d.py` drew the point P from the meta file, which describes only the latest case, and
  passed it to the projection as [north, east] where [east, north] was expected, so P was
  mirrored and stale. Each stream row now carries its target (`tu`) and pass bearing (`brg`),
  and P is derived per sample.
* The playhead was reset on every arriving case, so an unpaced benchmark only ever showed the
  start of each approach. Playback now finishes the case on screen (3 s past its gate), then
  jumps to the newest; `[` / `]` pin a case. Only the shown case plus a few recent ones are
  drawn.
* A running viewer serves the page it loaded at start-up: after editing `live3d.py`, restart
  it.
