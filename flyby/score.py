"""Scoring a pass: did the aircraft go through the point, at the angle, at the speed?

## Why the gate is the line, not the point

The obvious metric -- closest approach to the point -- is degenerate for a good pass. A
trajectory that goes exactly through P has distance-to-P with a double zero there, so
the "closest approach" time is decided by sensor noise: a 0.2 m jitter in the telemetry
moves the reported crossing time by an arbitrary amount, and everything scored at that
instant (angle, speed) with it. Any scoring that reports three decimals of "closest
approach" on a tangential trajectory is measuring noise.

So the pass is scored where it is well-defined: **the crossing of the pass line**.
The requirement "go through P on this bearing at this speed" is a gate, and the gate is
the infinite line through P with direction `angle_deg`. A pass is an event -- s changes
sign -- and at that instant the along-track coordinate is zero *by definition*, so all
three numbers are well posed:

  * `cross_track_m`     signed offset from the line at the crossing (the miss)
  * `angle_err_deg`     commanded track minus the required track
  * `speed_err_mps`     measured ground speed minus the required speed

## ...except for a terminal descent, where the gate is the ARRIVAL

A pass line through a point at altitude is a line in a *plane*, so "crossed the line" is
a well-posed event and the score above is the sharp one. For a point ON THE GROUND the
same line extends infinitely upward, and the aircraft crosses that plane while it is still
hundreds of metres short and high: the crossing is real and meaningless. Measured on a
live ground approach, the first line crossing was at 202 m altitude -- everything except
the altitude was already correct to 0.04 m and 0.5 deg.

So the gate is selectable. `gate="line"` scores the crossing (the tangent case);
`gate="closest"` scores the moment of minimum 3-D distance to the point, which for a
ground target IS the touchdown and is well defined for exactly the reason the line
crossing is not.

For a MOVING point the gate is a moving gate: the line is re-evaluated at the crossing
time against the point's position then, using the same relative-velocity geometry the
guidance used. The target's own motion is therefore not counted as a miss.
"""
import math
from dataclasses import dataclass, field

from . import geo
from .guidance import decompose


@dataclass
class Sample:
    """One row of a recorded trajectory."""
    t: float
    ned: tuple            # own aircraft NED
    vel: tuple            # own aircraft NED velocity
    target_ned: tuple
    target_vel: tuple
    plan: object = None   # the PassPlan that was issued, if any

    @property
    def alt_m(self):
        return -self.ned[2]


@dataclass
class PassScore:
    """The verdict on one crossing."""
    t_cross: float
    ned_at_cross: tuple
    vel_at_cross: tuple
    # The required quantities
    target_ned: tuple
    target_vel: tuple
    want_angle_deg: float
    want_speed_mps: float
    want_alt_m: float
    want_fp_angle_deg: float | None = None
    # What was achieved
    cross_track_m: float = 0.0      # signed: + is right of the required track
    along_at_cross_m: float = 0.0
    miss_2d_m: float = 0.0
    miss_3d_m: float = 0.0
    got_angle_deg: float = 0.0
    angle_err_deg: float = 0.0
    got_speed_mps: float = 0.0
    speed_err_mps: float = 0.0
    got_alt_m: float = 0.0
    alt_err_m: float = 0.0
    got_fp_angle_deg: float = 0.0
    fp_angle_err_deg: float | None = None
    t_go_at_cross_s: float = 0.0
    feasible: bool = True
    # For a terminal descent that ended on the ground: the instant the point was closest
    # (the touchdown) and the instant the aircraft was still FLYING (the arrival). They are
    # different samples and the difference matters -- see `score_run`.
    landed: bool = False
    t_arrival_s: float | None = None

    @property
    def heading_err_deg(self):
        return self.angle_err_deg

    def score_vector(self):
        """(miss, angle error, speed error) in the units a gate is written in."""
        return (self.miss_2d_m, abs(self.angle_err_deg), abs(self.speed_err_mps))

    def as_dict(self):
        d = dict(self.__dict__)
        d["score_vector"] = self.score_vector()
        return d

    def one_line(self):
        fp = "" if self.fp_angle_err_deg is None else f" fp={self.got_fp_angle_deg:+6.1f} ({self.fp_angle_err_deg:+6.1f})"
        arr = f"  [touchdown; kinematics at arrival t={self.t_arrival_s:.1f}s]" \
            if self.landed and self.t_arrival_s is not None else ""
        return (f"t={self.t_cross:6.1f}s  miss={self.miss_2d_m:7.2f}m  "
                f"track={self.got_angle_deg:6.2f} ({self.angle_err_deg:+6.2f}deg)  "
                f"gs={self.got_speed_mps:6.2f} ({self.speed_err_mps:+5.2f}m/s)  "
                f"dAlt={self.alt_err_m:+6.1f}m{fp}{arr}")


def find_crossings(samples, want_angle_deg: float, want_fp_angle_deg=None):
    """Every crossing of the pass line, linearly interpolated onto s = 0.

    `samples` is a list of Sample ordered in time. The crossing is found in the pass
    frame of the (moving) point at each instant, so a target that walks across the gate
    is not mistaken for the aircraft crossing it.
    """
    out = []
    prev = None
    for s in samples:
        if prev is not None:
            s0, _, _, _ = decompose(
                (prev.ned[0] - prev.target_ned[0], prev.ned[1] - prev.target_ned[1]),
                (prev.vel[0] - prev.target_vel[0], prev.vel[1] - prev.target_vel[1]),
                want_angle_deg)
            s1, _, _, _ = decompose(
                (s.ned[0] - s.target_ned[0], s.ned[1] - s.target_ned[1]),
                (s.vel[0] - s.target_vel[0], s.vel[1] - s.target_vel[1]),
                want_angle_deg)
            if s0 < 0.0 <= s1:
                f = -s0 / (s1 - s0) if s1 != s0 else 0.0
                out.append(_lerp_sample(prev, s, f))
        prev = s
    return out


def _lerp_sample(a: Sample, b: Sample, f: float) -> Sample:
    lerp = lambda x, y: tuple(x[k] + f * (y[k] - x[k]) for k in range(len(x)))
    return Sample(
        t=a.t + f * (b.t - a.t),
        ned=lerp(a.ned, b.ned),
        vel=lerp(a.vel, b.vel),
        target_ned=lerp(a.target_ned, b.target_ned),
        target_vel=lerp(a.target_vel, b.target_vel),
        plan=b.plan if f >= 0.5 else a.plan,
    )


def score_crossing(c: Sample, spec, cfg=None) -> PassScore:
    """Score one interpolated crossing against a PassSpec."""
    rel = (c.ned[0] - c.target_ned[0], c.ned[1] - c.target_ned[1])
    rel_vel = (c.vel[0] - c.target_vel[0], c.vel[1] - c.target_vel[1])
    s, e, s_dot, e_dot = decompose(rel, rel_vel, spec.angle_deg)

    gs = math.hypot(c.vel[0], c.vel[1])
    trk = math.degrees(math.atan2(c.vel[1], c.vel[0])) % 360.0 if gs > 0.2 else float("nan")
    dh = c.ned[2] - c.target_ned[2]
    miss2 = math.hypot(rel[0], rel[1])
    miss3 = math.hypot(rel[0], rel[1], dh)
    v3 = math.sqrt(c.vel[0] ** 2 + c.vel[1] ** 2 + c.vel[2] ** 2)
    fp = math.degrees(math.atan2(-c.vel[2], math.hypot(c.vel[0], c.vel[1]))) if v3 > 0.2 else float("nan")
    # NED z is a DOWN offset from the origin, so a positive dh means we are above the
    # point. Scoring the difference rather than the absolute altitude keeps this
    # origin-free: the number is the same whether the rig is sited in Zurich or Nairobi.
    alt_err = -dh

    sc = PassScore(
        t_cross=c.t,
        ned_at_cross=c.ned,
        vel_at_cross=c.vel,
        target_ned=c.target_ned,
        target_vel=c.target_vel,
        want_angle_deg=spec.angle_deg,
        want_speed_mps=spec.speed_mps,
        want_alt_m=spec.alt_m,
        want_fp_angle_deg=spec.fp_angle_deg,
        cross_track_m=e,
        along_at_cross_m=s,
        miss_2d_m=miss2,
        miss_3d_m=miss3,
        got_angle_deg=trk,
        angle_err_deg=geo.angle_diff_deg(trk, spec.angle_deg) if gs > 0.2 else float("nan"),
        got_speed_mps=gs,
        speed_err_mps=gs - spec.speed_mps,
        got_alt_m=spec.alt_m + alt_err,
        alt_err_m=alt_err,
        got_fp_angle_deg=fp,
        fp_angle_err_deg=(geo.angle_diff_deg(fp, spec.fp_angle_deg)
                          if (spec.fp_angle_deg is not None and v3 > 0.2) else None),
    )
    if c.plan is not None:
        sc.t_go_at_cross_s = c.plan.t_go_s
        sc.feasible = c.plan.feasible
    return sc


def score_run(samples, spec, cfg=None) -> list:
    """Score a run's gate events.

    `spec.gate` picks which: "line" for a pass through a point at altitude, "closest" for a
    terminal descent onto a point where the pass plane extends above the target.
    """
    if getattr(spec, "gate", "line") == "closest":
        return [_score_terminal_descent(samples, spec, cfg)] if samples else []
    return [score_crossing(c, spec, cfg) for c in find_crossings(samples, spec.angle_deg,
                                                                 spec.fp_angle_deg)]


# How far above the point counts as "still airborne" when scoring a terminal descent. Half a
# metre is about skid height, which is the physical scale that matters here: the samples
# below it are a landed aircraft, and everything measured from a landed aircraft -- ground
# speed, track angle, flight-path angle -- is the manoeuvre of coming to rest, not of
# arriving at the pass.
TOUCHDOWN_M = 0.5


def _arrival_sample(samples, spec):
    """The sample at which the aircraft ARRIVED, as opposed to the one where it STOPPED.

    A terminal descent is supposed to end on the ground, so the minimum-distance sample of a
    good run is the aircraft sitting on the target. Measured on the live ground approach that
    closed this repo's vertical blocker: touchdown at 0.06 m from the point, and then a
    further 24 s of the aircraft settling on its skids, during which its ground speed falls
    3.00 -> 0.18 m/s and its track angle wanders by tens of degrees. Scoring the pass there
    reports "0.2 m/s instead of 3 m/s, 67 deg off track" for a descent that landed on its
    target -- not a measurement but an artefact of measuring a stopped object.

    So the two families of error come from two samples and the score carries both:

      POSITION   at the closest sample    -- that is the miss, and it is exact
      KINEMATICS at the last sample still airborne before it

    Walking back to the last sample above `TOUCHDOWN_M` is the honest definition of "as it
    arrives". The speed rule is kept only as a fallback for a run that ends mid-air without
    ever reaching the point -- there the velocity collapsing is all the evidence there is.
    """
    i = min(range(len(samples)),
            key=lambda k: sum((samples[k].ned[j] - samples[k].target_ned[j]) ** 2
                               for j in range(3)))
    height_above = lambda s: -(s.ned[2] - s.target_ned[2])   # noqa: E731
    for k in range(i - 1, -1, -1):
        if height_above(samples[k]) > TOUCHDOWN_M:
            return samples[k], samples[k] is not samples[i]
    # Never got within touchdown height of the point: fall back to the last sample that was
    # still flying at a sensible fraction of the required arrival speed.
    need = max(0.5 * float(getattr(spec, "speed_mps", 0.0) or 0.0), 0.5)
    for k in range(i - 1, -1, -1):
        v = samples[k].vel
        if math.hypot(v[0], v[1]) >= need:
            return samples[k], samples[k] is not samples[i]
    return samples[i], False


def _score_terminal_descent(samples, spec, cfg=None) -> PassScore:
    arr, backed_up = _arrival_sample(samples, spec)
    sc = score_crossing(_closest_sample(samples), spec, cfg)
    sc.landed = backed_up
    if backed_up:
        kin = score_crossing(arr, spec, cfg)
        # Position from the touchdown, motion from the arrival.
        for f in ("got_angle_deg", "angle_err_deg", "got_speed_mps", "speed_err_mps",
                  "got_fp_angle_deg", "fp_angle_err_deg"):
            setattr(sc, f, getattr(kin, f))
        sc.t_arrival_s = arr.t
    return sc


def _closest_sample(samples):
    """The sample nearest the point in 3-D, refined by a parabolic fit through its two
    neighbours so the reported time is not quantised to the telemetry rate."""
    i = min(range(len(samples)),
            key=lambda k: sum((samples[k].ned[j] - samples[k].target_ned[j]) ** 2
                               for j in range(3)))
    s = samples[i]
    if 0 < i < len(samples) - 1:
        a, b, c = samples[i - 1], s, samples[i + 1]

        def d(x):
            return math.sqrt(sum((x.ned[j] - x.target_ned[j]) ** 2 for j in range(3)))
        d0, d1, d2 = d(a), d(b), d(c)
        denom = (d0 - 2 * d1 + d2)
        if abs(denom) > 1e-9:
            frac = 0.5 * (d0 - d2) / denom
            if -1.0 < frac < 1.0:
                f = frac
                lerp = lambda x, y: tuple(x[j] + f * (y[j] - x[j]) for j in range(3))
                return Sample(t=b.t + f * (c.t - b.t),
                                    ned=lerp(b.ned, c.ned), vel=lerp(b.vel, c.vel),
                                    target_ned=b.target_ned, target_vel=b.target_vel,
                                    plan=b.plan)
    return s


def closest_approach(samples) -> tuple:
    """(time, Sample, 2-D distance) of the nearest approach to the point, diagnostic only."""
    if not samples:
        return None
    best = min(samples, key=lambda s: math.hypot(s.ned[0] - s.target_ned[0],
                                                 s.ned[1] - s.target_ned[1]))
    return best.t, best, math.hypot(best.ned[0] - best.target_ned[0],
                                    best.ned[1] - best.target_ned[1])
