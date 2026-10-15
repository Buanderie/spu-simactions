"""Where the point is, and what the pass has to satisfy.

Scenarios are the experimental design: a point (static or moving) plus a pass
requirement plus the aircraft's starting state. Everything downstream -- the law, the
scorer, the plots -- only ever sees a `PassSpec` and a `target_fn(t)`, so adding a
scenario here is the only thing needed to test a new geometry.

Starting states are the honest part of this file. A flyby law is judged on approach
geometry, not on being handed a perfect one, so `approach_for()` places the aircraft the
way a real one would be: on the line and at speed is the easy case, and the interesting
cases are the offset ones (it is off to one side, it is slow, it is pointed the wrong
way). The offset cases are where guidance choices actually differ.
"""
import math
from dataclasses import dataclass, field

from . import geo
from .guidance import PassSpec, compose, pass_frame


@dataclass
class Scenario:
    name: str
    spec: PassSpec
    start_ned: tuple
    start_vel_ned: tuple
    target_fn: object            # t -> (ned, vel_ned)
    notes: str = ""
    # A moving target needs its own description for the plots; keep it with it.
    target_label: str = "static"


def static_point(lat, lon, alt_m):
    def fn(t):
        return (geo.latlon_alt_to_ned(lat, lon, alt_m), (0.0, 0.0, 0.0))
    return fn


def linear_track(lat, lon, alt_m, speed_mps, bearing_deg):
    """A point walking a straight line at a constant ground speed and bearing."""
    def fn(t):
        p = geo.latlon_alt_to_ned(lat, lon, alt_m)
        v = (speed_mps * math.cos(math.radians(bearing_deg)),
             speed_mps * math.sin(math.radians(bearing_deg)), 0.0)
        return (p[0] + v[0] * t, p[1] + v[1] * t, p[2]), v
    return fn


def orbit(lat, lon, alt_m, radius_m, period_s, phase_deg=0.0, sign=1):
    """A point circling its own lat/lon centre -- the classic 'moving target'."""
    def fn(t):
        c = geo.latlon_alt_to_ned(lat, lon, alt_m)
        w = sign * 2.0 * math.pi / period_s
        a = math.radians(phase_deg) + w * t
        return (c[0] + radius_m * math.cos(a), c[1] + radius_m * math.sin(a), c[2]), \
               (-radius_m * w * math.sin(a), radius_m * w * math.cos(a), 0.0)
    return fn


def zigzag(lat, lon, alt_m, speed_mps, bearing_deg, leg_s):
    """A point walking a line and reversing -- a contact that will not hold a course."""
    def fn(t):
        p = geo.latlon_alt_to_ned(lat, lon, alt_m)
        v = (speed_mps * math.cos(math.radians(bearing_deg)),
             speed_mps * math.sin(math.radians(bearing_deg)), 0.0)
        leg = math.floor(t / leg_s)
        if leg % 2 == 1:
            v = (-v[0], -v[1], 0.0)
        # integrate exactly so the reversal is a corner, not a smear
        n = t - leg_s * math.floor(t / leg_s)
        return (p[0] + v[0] * n, p[1] + v[1] * n, p[2]), v
    return fn


def approach_for(spec: PassSpec, target_ned, offset_m=0.0, along_m=-900.0,
                 alt_offset_m=0.0, speed_mps=None, heading_deg=None):
    """Place the aircraft `along_m` before the point, `offset_m` right of the line,
    `alt_offset_m` above the point's altitude, flying at `speed_mps` on `heading_deg`.

    Defaults put it on the line, at the required speed, on the required heading -- the
    easy case. Override offset_m / along_m / heading_deg to build the cases that matter.
    """
    off = compose(spec.angle_deg, along_m, offset_m)
    pos = (target_ned[0] + off[0], target_ned[1] + off[1], target_ned[2] - alt_offset_m)
    v = speed_mps if speed_mps is not None else spec.speed_mps
    hd = heading_deg if heading_deg is not None else spec.angle_deg
    hd_rad = math.radians(hd)
    return pos, (v * math.cos(hd_rad), v * math.sin(hd_rad), 0.0)


# ------------------------------------------------------------------ the scenario set

def glide_altitude(along_m: float, fp_deg: float) -> float:
    """Metres ABOVE the target at `along_m` before it, on a glide of `fp_deg`.

    Negative for a descent. Used to start scenarios ON the glide slope, which is the only
    way a measured descent angle means anything: start off the slope and the altitude error
    at the gate is the geometry's, not the guidance's, and the two get confused.
    """
    return -along_m * math.tan(math.radians(fp_deg))


class Builder:
    """Fluent scenario construction, so a new pursuit case is four lines and not a copy of
    the last one.

    Deliberately generic: `angle_deg`, `speed_mps`, `approach_speed_mps`, `fp_angle_deg`
    and the target's own motion are all parameters, and a "ground" target is nothing but a
    target at altitude zero. Anything the law can do, a scenario can ask for, and nothing
    here knows which case is being built.
    """

    def __init__(self, origin: geo.Origin):
        self.origin = origin
        self.out = []

    def point(self, ned) -> tuple:
        return geo.ned_to_latlon_alt(*ned, self.origin)

    def add(self, name, *, ned=(600.0, 300.0, 0.0), angle_deg=30.0, speed_mps=12.0,
            approach_speed_mps=None, fp_angle_deg=None, along_m=-900.0, offset_m=0.0,
            alt_offset_m=0.0, heading_deg=None, motion=None, notes="") -> Scenario:
        lat, lon, alt = self.point(ned)
        spec = PassSpec(lat, lon, alt, angle_deg, speed_mps, fp_angle_deg=fp_angle_deg,
                        approach_speed_mps=approach_speed_mps, label=name,
                        gate="closest")
        tgt, tgt_vel = motion(0.0) if motion else static_point(lat, lon, alt)(0.0)
        pos, vel = approach_for(spec, tgt, offset_m, along_m, alt_offset_m,
                                speed_mps=approach_speed_mps or speed_mps,
                                heading_deg=heading_deg)
        sc = Scenario(name, spec, pos, vel, motion or static_point(lat, lon, alt), notes)
        self.out.append(sc)
        return sc

    def ground_case(self, name, *, ned=(600.0, 300.0, 0.0), angle_deg=30.0, speed_mps=3.0,
                    approach_speed_mps=12.0, fp_angle_deg=-20.0, along_m=-900.0,
                    offset_m=0.0, extra_alt_m=0.0, motion=None, notes=""):
        """A target on the ground, started on its glide slope unless told otherwise."""
        return self.add(name, ned=ned, angle_deg=angle_deg, speed_mps=speed_mps,
                        approach_speed_mps=approach_speed_mps, fp_angle_deg=fp_angle_deg,
                        along_m=along_m, offset_m=offset_m,
                        alt_offset_m=-glide_altitude(along_m, fp_angle_deg) + extra_alt_m,
                        motion=motion, notes=notes)


def build_ground_suite(origin: geo.Origin, airframe: str = "plane") -> list:
    """The case this repo was opened for: reach a point ON THE GROUND, from above, at an
    angle, as slowly as the airframe permits.

    Three requirements, and each is different in kind from a level flyby:

      * **the target is the ground**, so the terminal condition is an arrival at altitude
        zero rather than at some cruise height. Nothing about the law changes for this -- a
        ground target is a target at altitude zero -- but it changes which requirement is
        the binding one, because the vertical channel now has to arrive at zero.
      * **the angle is the 3-D flight-path angle**, not the compass track. A descent onto a
        point is a question of how steeply; the bearing is a secondary requirement.
      * **"as slowly as possible" is a requirement on the ARRIVAL**, which means the aircraft
        must be fast enough at the top to make the geometry and slow to the minimum by the
        gate. That is a speed SCHEDULE, tangent at the gate -- the part a level case never
        exercises at all.

    Angles and speeds are chosen from the airframe's real limits, because a suite full of
    demands the airframe cannot meet measures the airframe: a trainer limited to 5 m/s of
    climb cannot hold a 20 deg glide at 25 m/s, and a quad asked to touch down at 0.2 m/s is
    asked to stall in a hover.
    """
    plane = airframe == "plane"
    fp = -9.0 if plane else -20.0
    v_pass = 13.0 if plane else 3.0
    v_start = 25.0 if plane else 12.0
    g = Builder(origin)

    g.ground_case("ground_on_slope", speed_mps=v_pass, approach_speed_mps=v_start,
                  fp_angle_deg=fp,
                  notes="the textbook case: start on the glide slope, decelerate to the "
                        "minimum, arrive on the point at the demanded angle")
    g.ground_case("ground_offset_250", speed_mps=v_pass, approach_speed_mps=v_start,
                  fp_angle_deg=fp, offset_m=250.0,
                  notes="250 m right of the glide: the lateral turn has to fit inside the "
                        "descent")
    g.ground_case("ground_from_high", speed_mps=v_pass, approach_speed_mps=v_start,
                  fp_angle_deg=fp, extra_alt_m=180.0,
                  notes="started 180 m above the glide slope: a steeper descent than the "
                        "pass demands, which the law has to fly down onto")
    g.ground_case("ground_too_slow", speed_mps=0.2, approach_speed_mps=v_start,
                  fp_angle_deg=fp,
                  notes="arrival speed below the airframe's minimum: the LAW should clamp it "
                        "and say so, not fly a 0.2 m/s touchdown")
    g.ground_case("ground_creep", speed_mps=v_pass, approach_speed_mps=v_start,
                  fp_angle_deg=fp,
                  motion=linear_track(*g.point((600.0, 300.0, 0.0)), 0.06 * v_start, 75.0),
                  notes="the target is walking at a sixth of the approach speed: the "
                        "descent becomes an intercept and the arrival speed has to be "
                        "timed against a moving point")
    # ...and one that is not. The minimum arrival speed onto a MOVING point is a geometric
    # floor -- |v_tgt.r| / sin(cross-track angle budget), the same relation as a
    # Zero-Effort-Velocity constraint -- so asking for the airframe's minimum onto a point
    # walking as fast as we are arrives needs 67 deg of cross-track angle and no law can
    # supply it. The point of the scenario is that the law says so.
    g.ground_case("ground_target_too_fast", speed_mps=v_pass, approach_speed_mps=v_start,
                  fp_angle_deg=fp,
                  motion=linear_track(*g.point((600.0, 300.0, 0.0)), 0.9 * v_start, 75.0),
                  notes="a target walking almost as fast as the approach: the intercept "
                        "geometry, not the guidance, sets the arrival speed, and the law "
                        "should report that")
    return g.out


def build_suite(origin: geo.Origin, airframe: str = "plane") -> list:
    """The default experiment suite. Named so a plot's filename says what it was.

    Deliberately spans the three things that break a flyby law: a start that is already
    aligned (does it hold the line?), a start that is laterally offset (does it null the
    offset without arriving late?), and a start that is pointed the wrong way (does it
    re-plan instead of committing?). Plus the two moving-point cases, because a moving
    point is the one that turns a flyby into an intercept.

    Everything that has a velocity is expressed as a fraction of the pass speed, so the
    suite is the same difficulty for a 25 m/s fixed wing and a 12 m/s quadrotor. Scaling
    the contacts by the airframe matters: an orbiting contact at 13.5 m/s is a routine
    intercept for the plane and an impossible one for the quad, and a suite that silently
    reports "guidance failed" for what is really "the target outran the aircraft" teaches
    the wrong lesson.
    """
    plane = airframe == "plane"
    speed = 25.0 if plane else 12.0
    cruise_alt = 100.0
    # A point 600 m north / 300 m east of home, at cruise altitude.
    pt_ned = (600.0, 300.0, -cruise_alt)
    lat, lon, alt = geo.ned_to_latlon_alt(*pt_ned, origin)

    def S(angle, spd=None, fp=None, label=""):
        return PassSpec(lat, lon, alt, angle, spd or speed, fp_angle_deg=fp, label=label)

    out = []

    def add(name, spec, target_fn, offset_m, along_m, alt_off, spd=None, hdg=None, notes=""):
        tn, _ = target_fn(0.0)
        pos, vel = approach_for(spec, tn, offset_m, along_m, alt_off, spd, hdg)
        out.append(Scenario(name, spec, pos, vel, target_fn, notes))

    add("aligned", S(30.0), static_point(lat, lon, alt), 0.0, -900.0, 0.0,
        notes="already on the line at the required speed: the law must not make it worse")
    add("offset_right_300", S(30.0), static_point(lat, lon, alt), 300.0, -900.0, 0.0,
        notes="300 m right of the line, the classic offset intercept")
    add("offset_right_800", S(30.0), static_point(lat, lon, alt), 800.0, -900.0, 0.0,
        notes="far offset: the turn has to start immediately and the profile is feasibility-bound")
    add("wrong_way", S(30.0), static_point(lat, lon, alt), 0.0, -900.0, 0.0,
        hdg=180.0, notes="flying away from the point on the reverse bearing")
    add("low_and_slow", S(30.0), static_point(lat, lon, alt), 250.0, -900.0, -70.0,
        spd=0.55 * speed, notes="below and under speed: two axes of error at once")
    # The dive pass is only a fair test if the aircraft starts ON the glide slope: a
    # required flight-path angle replaces the altitude arrival, so the altitude error at
    # the gate is whatever the approach geometry left over. Placing it consistently is
    # what makes the measured number mean "the glide is right" instead of "the glide is
    # right and the geometry was nonsense".
    #
    # The angle is 75% of the steepest the airframe can actually hold (a 15 deg dive at
    # 25 m/s wants 6.7 m/s of descent; a trainer capped at 5 cannot do it, and asking it
    # to produces a permanently "failed" row that says nothing). The too-steep case is
    # its own scenario, because watching the law report a requirement it cannot meet is
    # the whole point of having the flag.
    max_climb = 5.0 if plane else 6.0
    fp_deg = -0.75 * math.degrees(math.atan(max_climb / speed))
    add("descend", S(30.0, fp=fp_deg), static_point(lat, lon, alt), 250.0, -900.0,
        -900.0 * math.tan(math.radians(fp_deg)),
        notes=f"dive pass from the glide slope at {fp_deg:.0f} deg: a required "
              "flight-path angle instead of an altitude arrival")
    add("descend_too_steep", S(30.0, fp=fp_deg * 2.5), static_point(lat, lon, alt), 250.0,
        -900.0, -900.0 * math.tan(math.radians(fp_deg * 2.5)),
        notes="a dive the airframe cannot hold: the pass should be REPORTED infeasible, "
              "not silently flattened")
    add("moving_track", S(30.0),
        linear_track(lat, lon, alt, 0.24 * speed, 75.0), 200.0, -900.0, 0.0,
        notes="point walking a quarter of our speed on 075: the closing rate and the "
              "along-track clock both change")
    add("moving_orbit", S(30.0),
        orbit(lat, lon, alt, 6.0 * speed, 2.8 * 40.0), 150.0, -900.0, 0.0,
        notes="point circling: the relative geometry rotates under the law, so the "
              "profile is chording a curve")
    return out
