"""Fly-through-a-point guidance: the algorithm this repo exists to find.

## The problem

Given a geographic point P -- static or moving -- and a required compass angle
`angle_deg` and ground speed `speed_mps` at the moment of crossing, fly an aircraft so
that its trajectory passes as close as possible to P, and at that moment its ground
track is `angle_deg` and its ground speed is `speed_mps`.

"Flying to the point" does not solve this. A flyby is a *tangency* problem: the
trajectory has to intersect the pass line AT P, and arrive there with zero cross-track
rate, or the aircraft cuts the corner or loops around the point. Steering at the point
with a shrinking aimpoint (pure pursuit) has a permanent following lag, so it crosses
the line somewhere near P but not through it, and its speed at that moment is whatever
the pursuit happened to produce.

## The law

Everything happens in the *pass frame*: d is the required track direction and r is
90 deg clockwise from it ("right of the line"). Writing the own aircraft's position
relative to P as s (along-track, negative before the point) and e (cross-track, positive
right of the line), with the matching rates:

    e_dot = V * sin(sigma - theta)          sigma = own heading, theta = required angle

The vehicle's turn dynamics are too slow to use `e -> 0` as a bare exponential: the
cross-track error would still be metres when s crosses zero. So instead of a feedback
target, the law regenerates a *trajectory* every tick -- a cubic Hermite in the
remaining time T that starts at the measured state (e, e_dot) and ends at (0, 0):

    e(tau)     = e0*h00(u) + e_dot0*T*h10(u)        u = tau/T
    e_dot(tau) = e0*h00'(u)/T + e_dot0*h10'(u)
    h00(u) = 1 - 3u^2 + 2u^3     h10(u) = u - 2u^2 + u^3

At tau = T both terms vanish: the aircraft arrives at P with zero cross-track error AND
zero cross-track rate, which is exactly the tangency a flyby needs, and its velocity is
then V along d no matter what it had to do to get there. The same shape is used for
altitude, so the 3D arrival is also tangent. Recomputing it from the live state every
tick (rather than tracking a pre-planned profile) makes it self-correcting: wind,
model error and a moving target all show up in (e, e_dot) and are absorbed.

Two refinements matter in practice:

* **Lead.** The feedforward assumes the aircraft can produce `e_dot(0)` instantly. It
  cannot -- a fixed wing needs seconds to roll. Command `e_dot(0) + tau_lead*e_ddot(0)`
  instead, i.e. fly the profile `tau_lead` in the future. Without it the aircraft is
  always late and clips the pass line short of the point.
* **Feasibility.** A profile has to be built that the vehicle can actually fly:
  `max|e_ddot| <= a_lat_max` bounds T from below. When the geometry is too demanding
  (far out, close in, very fast) T is clamped, the profile is no longer reachable, and
  the aircraft flies the best it can -- the scorer reports the resulting miss rather
  than the law pretending otherwise.

For a *moving* point the same law runs on the relative velocity (own minus target), so
the along-track and cross-track rates are closing rates: the profile becomes an
intercept without any extra machinery. T itself is derived from the closing rate
along the pass line, smoothed, because it is the loop's slowest state.

## What this module does not do

It does not command anything. It turns (own state, point state, spec) into a
`PassPlan` -- a desired horizontal velocity vector, a climb rate and a speed -- and
`mavlink_link.py` decides whether to hand that to PX4 as an OFFBOARD velocity setpoint
(quadrotor), an OFFBOARD velocity *bearing* plus a TECS trim airspeed (fixed wing), or
anything else. Keeping the geometry free of the transport is what makes it portable
into swarmpu's firmware, where the transport is a MAVLink link at 115200 baud.
"""
import math
from dataclasses import dataclass, field

from . import geo

G = 9.80665


# --------------------------------------------------------------------------- frames

def pass_frame(angle_deg: float):
    """(along-track unit, right-of-track unit) in (north, east) metres."""
    return geo.track_unit(angle_deg), geo.right_unit(angle_deg)


def decompose(rel_ned, rel_vel_ned, angle_deg: float):
    """Split a relative position/velocity into (s, e) and their rates in the pass frame."""
    d, r = pass_frame(angle_deg)
    s = rel_ned[0] * d[0] + rel_ned[1] * d[1]
    e = rel_ned[0] * r[0] + rel_ned[1] * r[1]
    s_dot = rel_vel_ned[0] * d[0] + rel_vel_ned[1] * d[1]
    e_dot = rel_vel_ned[0] * r[0] + rel_vel_ned[1] * r[1]
    return s, e, s_dot, e_dot


def compose(angle_deg: float, s: float, e: float):
    """Inverse of `decompose` for positions."""
    d, r = pass_frame(angle_deg)
    return (s * d[0] + e * r[0], s * d[1] + e * r[1])


# ------------------------------------------------------------------- the Hermite shape
#
# One shape, three axes: a cubic Hermite on POSITION, from where the aircraft is now and
# how fast it is going, to where the point WILL BE when the aircraft gets there and at
# the velocity the pass requires.
#
# Doing it on absolute position rather than on the relative error is what makes the
# moving-point case fall out for free, and it is worth being explicit about why, because
# the obvious formulation -- a cubic on the relative cross-track error, with the target's
# velocity subtracted at the end -- is wrong in a way that is invisible until the target
# moves:
#
#   * the requirement ("track theta, ground speed V") is about the AIRCRAFT, so it is an
#     absolute quantity, while the error the law measures is RELATIVE. Any formulation
#     that plans in one frame and commands in the other has to convert at the boundary,
#     and missing a term there is silent. Commanding "relative lateral rate = the current
#     relative lateral rate" through a first-order plant integrates into a constant
#     acceleration: measured as a moving point being left 340 m out with the nose 33 deg
#     off, having flown a profile that looked correct the whole way.
#   * the target's path is curved, so a cubic on the RELATIVE error is a chord of a
#     curve, and the angle error at the gate grows with how much of the turn the approach
#     spans (+12 deg measured against a contact on a 70 s orbit).
#
# Anchoring the terminal point at the prediction and the terminal velocity at the
# requirement removes both: the chord is in the aircraft's own frame, where it is exact,
# and the target's curvature only enters through the endpoint, which is where it belongs.

def hermite4(p0: float, v0: float, p1: float, v1: float, T: float, u: float):
    """Cubic on [0, T] through (p0, v0) -> (p1, v1). Returns (value, rate, accel) at u."""
    if T <= 1e-6:
        return p1, 0.0, 0.0
    u2, u3 = u * u, u * u * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2
    dh00 = 6.0 * u2 - 6.0 * u
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = -dh00
    dh11 = 3.0 * u2 - 2.0 * u
    ddh00 = 12.0 * u - 6.0
    ddh10 = 6.0 * u - 4.0
    ddh01 = -ddh00
    ddh11 = 6.0 * u - 2.0
    return (p0 * h00 + v0 * T * h10 + p1 * h01 + v1 * T * h11,
            (p0 * dh00 + v0 * T * dh10 + p1 * dh01 + v1 * T * dh11) / T,
            (p0 * ddh00 + v0 * T * ddh10 + p1 * ddh01 + v1 * T * ddh11) / (T * T))


def hermite4_accel(p0: float, v0: float, p1: float, v1: float, T: float, u: float) -> float:
    """|accel| of `hermite4` at u. Linear in u, so a bound over an interval is at an end."""
    if T <= 1e-6:
        return float("inf")
    return abs((p0 * (12.0 * u - 6.0) + v0 * T * (6.0 * u - 4.0)
                + p1 * (6.0 - 12.0 * u) + v1 * T * (6.0 * u - 2.0)) / (T * T))


def hermite4_needed(p0, v0, p1, v1, T, tau_lead) -> float:
    """|accel| over the stretch the law actually reads, [0, min(1, tau_lead/T)]."""
    if T <= 1e-6:
        return float("inf")
    u = min(1.0, tau_lead / T)
    return max(hermite4_accel(p0, v0, p1, v1, T, 0.0), hermite4_accel(p0, v0, p1, v1, T, u))


def hermite4_stretch_T(p0, v0, p1, v1, a_max, tau_lead, T0, cap=600.0, grow=1.25) -> float:
    """Smallest horizon T >= T0 this profile can be flown within `a_max`, else T0.

    Searches UPWARD from the horizon the geometry actually calls for, and only that way.
    Two reasons, both learned the hard way:

      * The bound is not monotone in T (see `hermite4_min_T`'s note), so the feasible set
        is a window. Sweeping the whole positive axis from 1 ms finds the window's FAR
        edge: a dive pass whose glide was being tracked to a centimetre over a 0.70 s
        remaining horizon had its vertical profile stretched to 3.6 s -- the far edge of
        a window that also contained 0.70 -- and arrived with the descent flattened to a
        third of the demanded flight-path angle.
      * Stretching DOWNWARD is never wanted. The profile is supposed to end when the
        aircraft reaches the gate; a shorter horizon would be aiming at the gate before
        the aircraft got there.

    The multiplicative sweep is deliberately finer (1.25x) than the window is wide, so a
    narrow window cannot be stepped over between two probes.
    """
    if hermite4_needed(p0, v0, p1, v1, T0, tau_lead) <= a_max:
        return T0
    lo, hi = T0, T0
    for _ in range(80):
        hi = min(cap, hi * grow)
        if hermite4_needed(p0, v0, p1, v1, hi, tau_lead) <= a_max:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if hermite4_needed(p0, v0, p1, v1, mid, tau_lead) > a_max:
                    lo = mid
                else:
                    hi = mid
            return hi
        lo = hi
        if hi >= cap:
            break
    return T0


def readable_point(p0, v0, p1, v1, T, tau_lead, a_max):
    """How far along its profile to read, so that the vehicle can still obey the tail.

    The obvious answer -- read a fixed lead time ahead -- is only right when the profile is
    already nearly satisfied where you are reading it. It is wrong exactly when the profile
    still has a rate to shed at the point you read it, because then the tail between the
    read point and the gate demands more acceleration than the vehicle has and the residual
    is still there on arrival.

    So solve for the read point instead: find the earliest u whose remaining rate the
    vehicle could still bring to zero within the time left, i.e.
    ``|profile_rate(u)| <= a_max * T * (1 - u)``. Bisected -- the two sides are monotone in
    opposite directions over the useful range, and this runs a few dozen times a second.

    Measured: a ground approach offset 250 m from the glide, where the profile still has
    4 m/s of lateral rate to shed. Read a fixed second ahead, the gate was crossed with
    1.9 m/s of lateral rate -- 27 deg of track error -- with the miss down to 1.7 m and every
    other number exact.
    """
    if T <= 1e-6:
        return 1.0

    def ok(u):
        r = hermite4(p0, v0, p1, v1, T, u)[1]
        return abs(r) <= a_max * T * (1.0 - u) + 1e-9

    # Never read less than the lead: that much of the profile is unreachable behind the
    # vehicle's own lag whatever the acceleration limit says.
    umin = min(1.0, tau_lead / T)
    if ok(umin):
        return umin
    if not ok(1.0):
        # The tail cannot be obeyed at all -- the profile is genuinely infeasible. Read the
        # end and let the feasibility flag say so; there is no better answer to give.
        return 1.0
    # Smallest readable point: reading EARLIER is what makes the vehicle track the profile,
    # so this is a lower-bound search, not an upper one. (Returning the largest one instead
    # commands the terminal rate from the first tick -- no correction at all, and a pass
    # that crosses the gate at exactly the offset it started at.)
    lo, hi = umin, 1.0
    for _ in range(30):
        u = 0.5 * (lo + hi)
        if ok(u):
            hi = u
        else:
            lo = u
    return hi


def zev_rendezvous(own_ned, own_vel, term_ned, term_vel_rel):
    """Generalised-ZEV rendezvous feasibility and earliest arrival, for a double integrator.

    The pass requirement -- position AND velocity at a point -- is a rendezvous with a
    prescribed non-zero terminal relative velocity, which is exactly the problem de Paiva,
    Carvalho & Rodrigues state in closed form (arXiv:2006.09614). Their law is an
    ACCELERATION, `6 ZEM/t_go^2 - 2 ZEV/t_go`; we do not use it, for two reasons recorded in
    `update`: the acceleration form has to be integrated to reach the velocity interface
    this vehicle is commanded through, and its optimality argument assumes no acceleration
    bound at all, which would give up the "refuse rather than attempt" property that the
    benchmark shows is worth more than accuracy.

    What IS worth taking is their FEASIBILITY TEST, which needs no integration and is
    strictly sharper than the geometric floor this file uses. With

        a1 = 2(|v0|^2 + |vf|^2 + v0.vf),   a2 = 6 dr.(v0+vf),   a3 = 6|dr|^2

    the time-penalty-free quartic degenerates to `a1 t^2 + 2 a2 t + 3 a3 = 0`, which has a
    positive real root if and only if

        a2 < 0    and    a2^2 >= 3 a1 a3

    Read out: *the pass is feasible iff the current relative position and the sum of the
    initial and terminal relative velocities point at each other.* The second condition is
    the interesting one and it is the part the geometric floor cannot express -- it is a
    bound on how much of the offset the vehicle can trade against its own velocity.

    Sanity check, straight-line static approach at D with speed v onto arrival speed v_pass:
    a2 = -6 D (v + v_pass) and a2^2 - 3 a1 a3 = 36 D^2 v v_pass > 0, so always feasible, and
    at v == v_pass the small root is exactly D/v -- the trivial answer, as it must be.
    """
    dr = (own_ned[0] - term_ned[0], own_ned[1] - term_ned[1], own_ned[2] - term_ned[2])
    v0, vf = own_vel, term_vel_rel
    dot = lambda a, b: sum(a[i] * b[i] for i in range(3))          # noqa: E731
    n2 = lambda a: sum(c * c for c in a)                            # noqa: E731
    a1 = 2.0 * (n2(v0) + n2(vf) + dot(v0, vf))
    a2 = 6.0 * dot(dr, (v0[0] + vf[0], v0[1] + vf[1], v0[2] + vf[2]))
    a3 = 6.0 * n2(dr)
    disc = a2 * a2 - 3.0 * a1 * a3
    ok = bool(a2 < 0.0 and disc >= 0.0 and a1 > 0.0)
    t = (-a2 - math.sqrt(disc)) / a1 if ok else float("nan")
    return ok, t, (a1, a2, a3, disc)


def ramp(u: float) -> float:
    """w(u) = u - sin(2*pi*u)/(2*pi): 0 at u=0, 1 at u=1, and -- the point -- a derivative
    that is 0 at both ends and 1 through the middle.

    This is the shape a SPEED schedule wants, and it is not the shape everything else in
    this file wants. The minimum-jerk `h00` is fine for position (hold still, then move, then
    stop) and actively wrong for speed: it holds the approach speed until the last fifth of
    the approach and then sheds all of it, so the deceleration needed at the end is several
    times what the airframe can give and the aircraft arrives at the gate still fast. On a
    fixed-wing approach from 25 to 11 m/s that measured +12.3 m/s of arrival-speed error,
    with every other number exact.

    A ramp with constant slope through the middle and the slope eased in and out spreads the
    same total change evenly, so the peak deceleration is the average -- which is the number
    an airframe can actually be checked against.
    """
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u - math.sin(2.0 * math.pi * u) / (2.0 * math.pi)


def hermite4_min_T(p0, v0, p1, v1, a_max, tau_lead=0.0, floor=1e-3, cap=1e5) -> float:
    """The shortest T over which this profile can be flown within `a_max`.

    The acceleration a profile demands at u=0 is
    ``|6(p1-p0) - T(4*v0 + 2*v1)| / T^2``, which falls like 1/T^2, reaches zero where the
    profile is a straight line, and then rises again like 1/T for large T -- because
    starting at a high rate and ending at a high rate with almost no net displacement over
    a long horizon needs a reversal. The feasible set is therefore a WINDOW, not a
    half-line, and a single-root bisection silently returns the wrong edge of it.

    Kept for callers that genuinely want "how fast can this profile be flown at all"
    (the plant's limits, sweeps, tests). The guidance itself uses
    `hermite4_stretch_T`, which searches only upward from the horizon it wants.
    """
    if hermite4_needed(p0, v0, p1, v1, max(floor, 1e-3), tau_lead) <= a_max:
        return floor
    lo = max(floor, 1e-3)
    hi = lo
    for _ in range(48):
        hi = min(cap, hi * 2.0)
        if hermite4_needed(p0, v0, p1, v1, hi, tau_lead) <= a_max:
            break
        if hi >= cap:
            return cap
    else:
        return cap
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if hermite4_needed(p0, v0, p1, v1, mid, tau_lead) > a_max:
            lo = mid
        else:
            hi = mid
    return hi


# ------------------------------------------------------------------------- the spec

@dataclass(frozen=True)
class PassSpec:
    """One pass requirement: where, at what track angle, at what speed.

    `angle_deg` is a compass bearing of the *ground track*, so it is defined by the
    trajectory itself and is what a scoring gate can measure. `fp_angle_deg` is
    optional and means the climb/descent angle of the 3D flight path at the crossing
    (+ up, - down) -- a dive pass is a different requirement from a level one, and it
    conflicts with "arrive at the point's altitude with zero rate", so only one of the
    two vertical modes is active at a time.
    """
    lat: float
    lon: float
    alt_m: float
    angle_deg: float
    speed_mps: float
    fp_angle_deg: float | None = None
    label: str = ""
    # The speed to be doing at the START of the approach, if it is to differ from the
    # arrival speed. This is what makes "arrive as slowly as possible" expressible: the
    # law then flies a decelerating schedule from here down to `speed_mps` at the gate,
    # tangent in speed as well as in position. Leave it None and the speed is constant.
    approach_speed_mps: float | None = None
    # Which event is "the pass" for scoring: the crossing of the pass plane, or the arrival
    # at the point. They are the same thing for a point at altitude and completely different
    # for a point on the ground, where the pass plane extends upward past the target. See
    # `score.score_run`.
    gate: str = "line"

    @property
    def ned(self):
        return geo.latlon_alt_to_ned(self.lat, self.lon, self.alt_m)


@dataclass
class GuidanceConfig:
    """Tuning. The defaults are the fixed wing's; `for_quad()` returns the multirotor's."""
    # Lateral shaping feasibility, expressed as the bank angle the law is allowed to plan
    # a turn into rather than as an acceleration, because that is the number a vehicle
    # actually has. For a coordinated level turn -- a fixed wing or a quadrotor, they
    # differ only in what limits them -- a_lat = g*tan(phi) INDEPENDENT of speed, so this
    # is speed-free by construction and cannot be mis-tuned against a different pass
    # speed. 45 deg is 9.8 m/s^2. Picking an acceleration directly is the easy mistake
    # here: 6 m/s^2 looks reasonable as a number and is only a 31 deg bank, which quietly
    # makes every far-offset approach infeasible.
    max_bank_deg: float = 45.0
    max_lateral_accel_mps2: float | None = None   # absolute override, if set
    # Vertical shaping feasibility, and the hard clamp on the commanded climb rate.
    max_vertical_accel_mps2: float = 2.0
    max_climb_rate_mps: float = 5.0
    # The DESCENT limit, separately, because it is a different number and the vehicle's own
    # is the interesting one. None means "same as the climb limit", which is the honest
    # reading for a fixed wing whose envelope is close to symmetric.
    #
    # This field exists because of a measured bug. PX4 clamps the commanded vertical
    # velocity at `MPC_Z_VEL_MAX_DN` / `MPC_Z_VEL_MAX_UP` -- inside
    # `PositionControl::_positionControl()`, which runs every update and therefore applies
    # to a pure VELOCITY setpoint as well as a position one. The stock iris values are
    # 1.5 m/s down and 3.0 m/s up, so the two axes differ by a factor of two in the default
    # configuration, and a single symmetric clamp in the guidance cannot represent that at
    # all: the law happily commanded 4 m/s of descent and the aircraft was flown at 1.5.
    # See docs/FINDINGS.md section 4.
    max_descent_rate_mps: float | None = None
    # How far ahead of the profile the law reads it. Should match the vehicle's response
    # time constant: a fixed wing must roll before it turns, a quadrotor's velocity loop
    # is nearly instantaneous. Tuned in scripts/lead_sweep.py, not guessed.
    tau_lead_s: float = 1.0
    # Optionally cap the read-ahead at this FRACTION of the time-to-go, so it is small
    # relative to the approach as well as matched to the vehicle. See the note at the use
    # of `tau` in `update`. None = off, which is the behaviour everything was tuned with.
    tau_lead_frac: float | None = None
    # --- optional ZEV components, off by default, both measured before adoption --------
    # Use the generalised-ZEV rendezvous root test INSTEAD OF ADDING TO the geometric
    # intercept floor (see `zev_rendezvous`). It is sharper -- it constrains the whole
    # position/velocity geometry, where the geometric floor only looks at the target's
    # cross-track velocity -- and it is published.
    use_zev_feasibility: bool = False
    # ...and take the arrival TIME from it too, rather than from the measured closing rate.
    # Those disagree, and which is better is an experiment, not an opinion: the measured
    # rate is self-correcting under model error but assumes the aircraft closes at the
    # speed it happens to be doing; the quartic is optimal only for a double integrator
    # that then does what it is told.
    use_zev_tgo: bool = False
    # When the ZEV test is on, does it REPLACE the geometric floor or merely add to it.
    zev_overrides_geometry: bool = True
    # Never extrapolate the profile over less than this. There is deliberately NO upper
    # bound: t_go is a measured quantity (-s / closing rate), and if the geometry says the
    # point is three minutes away then a three-minute profile is the right answer, not a
    # clamped one that silently desynchronises from the arrival. A long T on a distant
    # point produces a gentle, correct drift; a clamped T produces a profile that has
    # "finished" while the aircraft is still 400 m short, and the pass is then scored at
    # whatever the aircraft happened to be doing.
    min_t_go_s: float = 0.35
    max_t_go_s: float = 900.0
    # Closing rate used for T: floor it, because a momentarily-zero closing rate (a
    # target moving directly away) must not send T to infinity, and smooth it, because
    # raw s_dot is noisy enough to make the commanded profile jitter visibly.
    min_close_frac: float = 0.35
    close_tau_s: float = 1.5
    # A quadrotor can be told a speed and it will fly it. A fixed wing cannot (PX4's
    # OFFBOARD path carries no airspeed field -- see docs/FINDINGS.md), so for the fixed
    # wing the commanded magnitude is only ever used as a bearing, and the speed cap is
    # tight because TECS holds the trim airspeed. For the quad the cap is the airframe's.
    # The slowest the airframe will actually be commanded to fly. This is an AIRCRAFT
    # number, not a preference: a fixed wing below about 1.2x stall has no roll authority
    # left, and a quadrotor below a couple of m/s starts losing velocity estimate in its
    # own downwash. "Arrive as slowly as possible" means as slowly as THIS permits, so the
    # floor belongs here where the airframe is configured and not in a pass requirement
    # somebody can set from a console.
    min_speed_mps: float = 11.0
    # Longitudinal acceleration: how fast the approach speed may be shed. A trainer's thrust
    # and drag give it well under 1 m/s^2; a quadrotor can pitch over and give several.
    max_long_accel_mps2: float = 1.5
    # The largest angle off the pass line that is still acceptable at the gate. This is what
    # turns "as slowly as possible" into a bounded question when the point is MOVING.
    cross_angle_budget_deg: float = 45.0
    max_speed_overshoot: float = 1.25
    # The VEHICLE's horizontal speed ceiling, m/s, or None for "no separate limit". This is
    # a different quantity from `max_speed_overshoot`: that is how much faster than the pass
    # the law may go to buy lateral correction, this is what the aircraft will actually do.
    # PX4 enforces it at `MPC_XY_VEL_MAX`, in the same unconditional
    # `PositionControl::_positionControl()` that clamps the vertical setpoint, and it is set
        # to 12 m/s on the stock iris. See the note at the speed cap in `update`.
    max_speed_mps: float | None = None
    # In flight-path-angle mode the altitude RATE is the requirement and is commanded
    # directly; the residual off the glide line is then trimmed over the remaining time,
    # capped so the trim can never cost more than this much of the angle (0.25 m/s against
    # a 12 m/s pass is about a degree). See the vertical-channel note in `update`.
    fp_alt_trim_max_mps: float = 0.25
    # An earlier version of this also bounded the trim as a FRACTION of the pass speed, on
    # the reasoning that an absolute cap is not speed-free -- 0.25 m/s is 4.8 deg of flight
    # path at a 3 m/s terminal descent and 0.6 deg at 25 m/s -- which is the same argument
    # that puts the bank limit in degrees rather than m/s^2. It was measured and it is
    # wrong, so the fraction was removed:
    #
    #   frac   ground_on_slope / offset_250      ground_creep    ground_from_high
    #   0.02   0.01 m, -0.12 deg                 3.22 m          74.4 m
    #   0.08   0.01 m, -0.12 deg                 1.05 m          63.7 m
    #   0.25   0.01 m, -0.12 deg                 0.98 m          63.1 m   (the absolute cap)
    #
    # The textbook cases do not care at all, and the two cases that need the trim are
    # monotonically better the looser it is. The distinction from the bank limit is that the
    # trim is a CLOSURE term, not a steady-state limit: it has to deliver a residual altitude
    # error as a rate over the time remaining, so what matters is the absolute rate it can
    # supply and its peak over the last seconds, not the angle it costs. Bounding an angle
    # there bounds the thing that does the work.
    #
    # (The live ground approach does arrive 3.1 deg off the demanded -20 deg, which looked
    # like this cap at first. It is not: offline the same case scores -0.12 deg with the cap
    # at either value, so the residual is the real plant's lag, not the trim.)
    fp_alt_trim_max_frac: float | None = None
    # Does blowing the trim cap make the pass INFEASIBLE, or just clamp the trim?
    #
    # Measured on 250 random STATIC final approaches: this check was responsible for
    # refusing 40 % of them, and 66 % of those refusals would have flown inside the gate
    # tolerance anyway (100 % on the trainer and the iris, where it fires on every single
    # refusal). The reason is structural rather than a mis-tuned constant: the cap bounds
    # the RATE at which a residual altitude error is closed, but the profile is rebuilt from
    # the measured state every tick and closes that error on its own, so an instantaneous
    # rate above the cap is a statement about one tick, not about whether the approach can
    # be flown. The CLAMP is right -- it stops the trim from costing more than its share of
    # the angle. The verdict was not.
    #
    # False by default on the measurement. It stays available because on a moving point,
    # where the intercept itself is the problem, the verdict may well be earning its keep,
    # and which is which is worth keeping switchable until that is measured too.
    fp_trim_is_infeasible: bool = False
    # Optional trim on the *measured* cross-track rate, to soak up lag the lead term
    # does not cover (wind shear, an unmodelled turn radius). 0 = pure feedforward.
    rate_trim_gain: float = 0.0
    # Guard against re-arming a pass profile from a stale closing rate on the first tick.
    warmup_s: float = 0.0
    # Past this along-track distance the pass is over: stop re-planning a profile that
    # ends behind us and fly on along the line instead. Without it the law keeps trying
    # to null a cross-track error that no longer has a terminal point, and puts a kink
    # in the trajectory exactly where a second pass has to start from.
    fly_on_after_pass_m: float = 15.0
    fly_on_lateral_gain: float = 0.25
    fly_on_vertical_gain: float = 0.35

    @staticmethod
    def for_quad() -> "GuidanceConfig":
        c = GuidanceConfig()
        c.tau_lead_s = 0.35
        c.max_bank_deg = 30.0
        c.max_climb_rate_mps = 6.0
        c.max_vertical_accel_mps2 = 4.0
        c.close_tau_s = 0.8
        c.fly_on_after_pass_m = 8.0
        c.max_speed_overshoot = 2.0
        c.min_speed_mps = 1.5
        c.max_long_accel_mps2 = 4.0
        # A quadrotor can descend faster than it climbs, and the iris-class FDM was measured
        # here at 6 m/s down and 4 m/s up with the vertical limits raised (see
        # docs/FINDINGS.md section 4). The guidance's own numbers are what it plans against,
        # so they have to be the aircraft's and not PX4's stock 1.5/3.0 -- otherwise the law
        # optimises for a descent rate the autopilot will refuse to fly.
        c.max_climb_rate_mps = 5.0
        c.max_descent_rate_mps = 6.0
        return c

    def descent_rate_max(self) -> float:
        """The commanded DESCENT ceiling. `climb_rate_mps` is positive-up."""
        return self.max_climb_rate_mps if self.max_descent_rate_mps is None \
            else self.max_descent_rate_mps

    def vertical_rate_limit(self, climb_mps: float) -> float:
        """How far the law may push `climb_mps` (positive up) in the direction it points."""
        return self.max_climb_rate_mps if climb_mps >= 0.0 else self.descent_rate_max()

    def lat_accel_max(self) -> float:
        return self.max_lateral_accel_mps2 if self.max_lateral_accel_mps2 is not None \
            else G * math.tan(math.radians(self.max_bank_deg))


# ------------------------------------------------------------------------ the plan

@dataclass
class PassPlan:
    """What the law wants this tick. Transport-agnostic."""
    # Horizontal velocity to command, NED m/s (magnitude already clamped to the spec).
    vel_ned: tuple
    # Desired climb rate, m/s (+ up). NaN when the backend should not command it.
    climb_rate_mps: float
    # Ground speed the plan is flying at, m/s. Equals the commanded magnitude unless the
    # lateral correction needed more than max_speed_overshoot and it got clipped.
    speed_mps: float
    # Diagnostics -- every one of these is what the plots and the scorer are made of.
    along_track_m: float
    cross_track_m: float
    along_rate_mps: float
    cross_rate_mps: float
    cross_rate_cmd_mps: float      # what the law asked for, after lead and clamps
    along_rate_cmd_mps: float
    t_go_s: float
    profile_T_s: float       # max of the two axes; what `feasible` is judged against
    profile_T_lat_s: float
    profile_T_vert_s: float
    feasible: bool          # False when T had to be clamped above what the profile needs
    heading_deg: float      # commanded ground-track bearing
    vertical_mode: str      # "tangent" | "flightpath"
    post_pass: bool = False  # the point is behind us; flying on, not planning a profile
    speed_limited: bool = False  # the speed cap bound, so the profile could not be flown
    intercept_floor_mps: float = 0.0  # geometric minimum arrival speed for a moving point
    # Why the plan was refused, if it was. Two floors, reported apart because they mean
    # different things: `intercept_floor_binding` is a property of the GEOMETRY (no speed
    # satisfies this pass), `airframe_floor_binding` is a property of the AIRCRAFT (this
    # vehicle cannot be commanded that slowly and still fly). A benchmark that bins both
    # together cannot tell "the law cannot do this" from "no aircraft can".
    # The ZEV rendezvous test's own verdict and its arrival time, whether or not the
    # config asks for them to be acted on. Recorded always: they are cheap, and a benchmark
    # that can only report the verdict it acted on cannot tell a rejected-and-correct case
    # from one it never asked about.
    # WHY it was called infeasible, one short tag per check that fired. A boolean verdict
    # is not actionable: a benchmark that knows 40 % of static approaches are being refused
    # cannot tell whether that is an honest limit or a badly-tuned check, and the tag is the
    # difference between "look at the numbers" and "go and read guidance.py".
    reasons: tuple = ()
    zev_feasible: bool = True
    zev_t_go_s: float = float("nan")
    airframe_min_speed_mps: float = 0.0
    airframe_floor_binding: bool = False
    intercept_floor_binding: bool = False
    target_ned: tuple = (0.0, 0.0, 0.0)
    target_vel_ned: tuple = (0.0, 0.0, 0.0)


class TerminalProfileGuidance:
    """Stateful law. One instance per pass; call `update` at the control rate.

    The only state is the smoothed closing rate used for T -- everything else is
    recomputed from the measured state each tick, which is what makes the law
    stateless with respect to the aircraft.
    """

    def __init__(self, cfg: GuidanceConfig | None = None):
        self.cfg = cfg or GuidanceConfig()
        self._close_rate: float | None = None
        self._last_t: float | None = None
        # The decelerating speed schedule is a PLAN, and a plan has an epoch. Referencing it
        # to the instantaneous time-to-go does not work: the horizon shrinks as the aircraft
        # slows, so a read-ahead expressed as a fraction of it collapses toward the start of
        # the ramp every tick and the commanded speed never leaves v_start. What the schedule
        # needs is the time-to-go it STARTED with, latched at the first tick of the pass.
        self._sched_t0: float | None = None
        self._sched_v0: float = 0.0
        self._sched_armed: bool = False
        self._was_post_pass: bool = False

    def reset(self) -> None:
        self._close_rate = None
        self._last_t = None
        self._sched_t0 = None
        self._sched_v0 = 0.0
        self._sched_armed = False
        self._was_post_pass = False

    def arm_schedule(self, v0: float, t_go: float) -> None:
        """Start a new approach: this is the speed the ramp starts from, over this horizon."""
        self._sched_v0 = v0
        self._sched_t0 = max(t_go, 1e-3)
        self._sched_armed = True

    # -- T ----------------------------------------------------------------------
    @staticmethod
    def _predict(predict, dt, ned, vel):
        """Where the point will be in `dt` seconds, and how fast. Constant velocity unless
        the caller has a real predictor."""
        if predict is None:
            return (ned[0] + vel[0] * dt, ned[1] + vel[1] * dt, ned[2] + vel[2] * dt), vel
        p, v = predict(dt)
        return tuple(p), tuple(v)

    def _time_to_go(self, s: float, s_dot: float, v_pass: float, t: float) -> tuple:
        """Smoothed along-track closing rate and the time remaining to the point."""
        floor = max(0.5, self.cfg.min_close_frac * v_pass)
        raw = max(s_dot, floor)
        if self._close_rate is None or t < self.cfg.warmup_s:
            self._close_rate = raw
        else:
            dt = max(0.0, t - (self._last_t if self._last_t is not None else t))
            a = 1.0 - math.exp(-dt / max(1e-3, self.cfg.close_tau_s))
            self._close_rate += a * (raw - self._close_rate)
        self._last_t = t
        if s >= 0.0:
            return self._close_rate, 0.0
        return self._close_rate, -s / max(1e-3, self._close_rate)

    # -- the law ----------------------------------------------------------------
    def update(
        self,
        own_ned,
        own_vel_ned,
        spec: PassSpec,
        t: float,
        target_ned=None,
        target_vel_ned=(0.0, 0.0, 0.0),
        target_predict=None,
    ) -> PassPlan:
        """One tick. Returns what the aircraft should be doing, in NED.

        `target_predict(dt) -> (ned, vel_ned)` is the point's forward estimate. Optional;
        constant velocity is assumed without it, which is exact for a straight contact and
        wrong by the contact's turn rate for a curving one.
        """
        cfg = self.cfg
        tgt = tuple(spec.ned) if target_ned is None else tuple(target_ned)
        tgt_vel = tuple(target_vel_ned)
        d, r = pass_frame(spec.angle_deg)
        # Speed at the gate, and at the start of the approach. Both are clamped by the
        # AIRFRAME, not by the requirement: a pass that asks for 4 m/s from a trainer that
        # stalls at 7 is not a pass, it is a stall, and the clamp is the honest reading of
        # it. `h00` gives the decelerating schedule zero rate at the gate, so the arrival is
        # tangent in speed as well as in position -- which is what makes "as slow as
        # possible" a smooth thing to fly rather than an arrival at a deceleration limit.
        v_min = max(0.3, cfg.min_speed_mps)
        # The minimum arrival speed is a GEOMETRIC floor, not just an airframe one, and only
        # when the point is moving. Nulling the point's cross-track rate |v_tgt.r| while
        # arriving with no cross-track rate at all needs some moment of the approach to carry
        # that rate, and the only source of lateral rate is speed times sine of the
        # cross-track angle:
        #
        #     V_arrival >= |v_tgt . r| / sin(cross_angle_budget)
        #
        # which is the same relation as a Zero-Effort-Velocity / rendezvous constraint: the
        # speed you can arrive at is set by the geometry of the intercept, not by what the
        # airframe will do. Without it the law happily accepts "arrive at 3 m/s" onto a point
        # walking at 3 m/s and then reports 34 deg of track error at the gate, because 34 deg
        # is the least-bad answer and the requirement was not achievable at any speed.
        cross_budget = math.radians(max(1.0, cfg.cross_angle_budget_deg))
        intercept_floor = abs(tgt_vel[0] * r[0] + tgt_vel[1] * r[1]) / math.sin(cross_budget)
        v_pass = max(v_min, spec.speed_mps, intercept_floor)
        # Two different floors, and they mean different things, so they are reported apart:
        #
        #   GEOMETRIC   the intercept floor above -- the pass angle and the arrival speed
        #               cannot both be had at this point's motion. No speed satisfies it.
        #   AIRFRAME    the aircraft cannot be commanded below `min_speed_mps` and still fly:
        #               a fixed wing below ~1.2x stall has no roll authority, a quadrotor
        #               loses its own velocity estimate in the downwash. A requirement below
        #               it is not achievable BY THIS AIRCRAFT, and a different aircraft might
        #               satisfy it.
        #
        # Both are infeasibility verdicts rather than silent clamps. Clamping is what the law
        # has to do -- it cannot command a speed the vehicle will refuse -- but saying so is
        # what makes it trustworthy: "arrive at 0.2 m/s" silently flown as 1.5 m/s is a
        # different claim from "arrive at 0.2 m/s, which this aircraft cannot do", and only
        # one of them is true. `ground_too_slow` exists to hold that line.
        speed_floor_binding = intercept_floor > max(v_min, spec.speed_mps) + 1e-6
        airframe_floor_binding = spec.speed_mps < v_min - 1e-6
        v_start = max(v_pass, spec.approach_speed_mps or v_pass)

        # --- measured state, for feedback and for the report ------------------------
        # s: negative before the point. e: positive right of the required track.
        rel = (own_ned[0] - tgt[0], own_ned[1] - tgt[1])
        rel_vel = (own_vel_ned[0] - tgt_vel[0], own_vel_ned[1] - tgt_vel[1])
        s, e, s_dot, e_dot = decompose(rel, rel_vel, spec.angle_deg)
        # The along-track rate the aircraft is actually achieving, for the deceleration
        # feasibility check below. Smoothed, because raw differentiated velocity is noisy
        # enough to flicker the verdict on and off near the gate.
        v_now_along = s_dot
        # (Re)arm the speed schedule's epoch at the start of an approach, and on the tick
        # after a completed one, and whenever the point has motion the schedule does not
        # describe. A pass already under way keeps the epoch it started with -- that is the
        # whole point of having one.
        #
        # Re-arming on anything looser is a trap: a "is the point comfortably ahead?" test
        # re-armed 380 m into a 900 m approach, the ramp restarted from 12 m/s with a 50 s
        # horizon, the aircraft flew the second half of the approach at full speed, passed
        # 200 m above the gate, and then flew on at the arrival speed. Measured live on PX4;
        # the offline suites never see it because they start the pass already at speed, so
        # `s` never enters the region the heuristic was watching.

        # --- when, and where ---------------------------------------------------------
        # The terminal point is where the contact is PREDICTED to be when we arrive, and
        # the terminal velocity is what the pass asks for: v_pass along the track, no
        # lateral rate, no climb. Both absolute, both in the aircraft's own frame.
        # The along-track clock: the MEASURED closing rate, always.
        #
        # The geometry profiles are anchored on it because they have to finish when the
        # AIRCRAFT actually reaches the gate, and the aircraft arrives when its real closing
        # rate carries it there -- not when a schedule says it should. Referencing them on the
        # schedule time instead makes them finish early by exactly the speed-tracking error:
        # measured as a ground approach arriving 15% late, so the lateral profile completed
        # three seconds before the gate and the line was crossed with 1 m/s of lateral rate
        # still on it -- 17 deg of track error, on a profile that was perfectly flyable.
        #
        # (The schedule-derived time 2*(-s)/(v_start+v_pass) is exactly right when the speed
        # tracks perfectly and wrong otherwise, which is the wrong way round for a feedback
        # law. The speed schedule does not need it: it has its own latched epoch, below.)
        close_rate, t_go = self._time_to_go(s, s_dot, v_pass, t)
        t_go = min(max(t_go, cfg.min_t_go_s), cfg.max_t_go_s)
        p_end, _vel_end = self._predict(target_predict, t_go, tgt, tgt_vel)

        post_pass = s > cfg.fly_on_after_pass_m

        rearm = (not self._sched_armed
                 or (self._was_post_pass and not post_pass)
                 or math.hypot(tgt_vel[0], tgt_vel[1]) > 0.5)
        self._was_post_pass = post_pass
        if rearm:
            self._sched_v0 = v_start
            self._sched_t0 = None
            self._sched_armed = True

        # Pass-frame coordinates, anchored at the contact's CURRENT position, so that the
        # endpoints are (where we are relative to the point now) -> (where it will be).
        b0, b0d = rel[0] * r[0] + rel[1] * r[1], own_vel_ned[0] * r[0] + own_vel_ned[1] * r[1]
        b1 = (p_end[0] - tgt[0]) * r[0] + (p_end[1] - tgt[1]) * r[1]
        b1d = 0.0
        # Vertical runs in the NED down coordinate, anchored the same way: c positive
        # means the contact is below us.
        c0, c0d = own_ned[2] - tgt[2], own_vel_ned[2]
        c1 = p_end[2] - tgt[2]
        # A demanded flight-path angle replaces "arrive with no climb": the two are
        # different requirements and only one can be exact. It is an ABSOLUTE rate, and
        # positive means climbing -- so in this down-sense coordinate it enters negated,
        # or the aircraft flies the glide the wrong way and arrives above the point.
        fp = spec.fp_angle_deg
        c1d = (-v_pass * math.tan(math.radians(fp)) if fp is not None else 0.0)
        vertical_mode = "flightpath" if fp is not None else "tangent"

        # --- the ZEV test, optionally replacing the geometric floor --------------------
        # The terminal relative velocity this pass actually asks for: the arrival speed
        # along the pass line, the demanded vertical rate, minus whatever the point is
        # doing. That subtraction is the whole point -- it is what turns "fly through a
        # moving point at 3 m/s" into a rendezvous with a prescribed non-zero terminal
        # RELATIVE velocity, which is the form ZEV is stated in.
        v_rel_f = (v_pass * d[0] - tgt_vel[0],
                   v_pass * d[1] - tgt_vel[1],
                   c1d - tgt_vel[2])
        zev_ok, zev_t, zev_q = zev_rendezvous(own_ned, own_vel_ned, p_end, v_rel_f)

        if cfg.use_zev_tgo and zev_ok and math.isfinite(zev_t):
            # One fixed-point iteration: the ZEV arrival time is computed against the
            # contact point predicted at the measured rate, then the point is re-predicted
            # at the ZEV time. Not iterated to convergence -- one pass is enough to show
            # whether this helps, and a converged fixed point between a prediction and a
            # quartic is more machinery than the question needs.
            t_go = min(max(zev_t, cfg.min_t_go_s), cfg.max_t_go_s)
            p_end, _vel_end = self._predict(target_predict, t_go, tgt, tgt_vel)
            zev_ok, zev_t, zev_q = zev_rendezvous(own_ned, own_vel_ned, p_end, v_rel_f)
        if cfg.use_zev_feasibility and not zev_ok:
            # Replaces the geometric floor only. The per-axis Hermite acceleration, speed,
            # rate and airframe checks all still run: this test answers "is this reachable
            # by a double integrator", which is a different and easier question than "can
            # THIS aircraft fly it", and the benchmark is what says the second one is worth
            # asking (49% of descents are correctly refused on it).
            feasible = False
        if cfg.use_zev_feasibility and cfg.zev_overrides_geometry:
            speed_floor_binding = not zev_ok

        # --- feasibility -------------------------------------------------------------
        # Ask for the acceleration the profile will actually be READ at. Near the gate
        # tau_lead/T clips to 1 and the read stops early, so a whole-interval bound calls
        # a perfectly flyable profile impossible exactly where it matters most.
        #
        # The along axis is NOT in this check, and deliberately so: it is commanded as a
        # rate (below) rather than flown as a profile. A cubic along the track looks
        # reasonable and is a trap -- 1.8 m short of the gate with 0.3 s to run, it has to
        # satisfy "arrive at 25 m/s" from 21 m/s, so it brakes, the aircraft slows, the
        # next tick is even less feasible, and it stalls 0.3 m short forever. The along
        # axis is a speed; the lateral and vertical axes are profiles.
        tau = cfg.tau_lead_s
        # The read-ahead, optionally CAPPED BY THE APPROACH rather than only by the vehicle.
        #
        # A lead that tracks the vehicle's response is right about the vehicle and silent
        # about the problem: on a 900 m approach at 60 m/s there are ~15 s to run, so an
        # 11.65 s lead -- correct for an aircraft that takes 11.65 s to roll into a turn --
        # reads most of the way to the gate on the first tick, on a profile that is then
        # rebuilt every tick from a state the aircraft has not begun to approach. Measured
        # across the whole fleet, accuracy is a monotone function of the turn-response time
        # constant and of nothing else: five multirotors spanning 249 g to 6.5 kg all land
        # within half a point of each other, and every aircraft past ~7 s of turn response
        # falls to more than 100 m of miss.
        #
        # So the lead should also be small compared with the time remaining. `None` keeps
        # the original behaviour and is the default, because this is a guess about a
        # mechanism and has not been measured; the fleet benchmark is how to measure it.
        if cfg.tau_lead_frac is not None:
            tau = min(tau, cfg.tau_lead_frac * t_go)
        a_lat = cfg.lat_accel_max()
        a_vert = cfg.max_vertical_accel_mps2
        # Start optimistic and let each check below knock it down. The combination is ANDed
        # together at the end so that no check can overwrite another check's verdict.
        feasible = True
        # Each axis gets its OWN horizon. The gate is a single event, so t_go is shared --
        # but the acceleration and rate feasibility bounds are per-axis, and sharing their
        # result across axes is a quiet corruption: one axis that needs a long profile
        # (here, the lateral one reacting to a centimetre of cross-track error) stretches
        # the horizon for an axis that is already satisfied, and the stretched vertical
        # profile has to dive and come back to hit its endpoint -- which it does, by
        # flattening the glide a second and a half before the gate and arriving 11 deg
        # off the demanded flight-path angle with everything else perfect.
        T_lat = hermite4_stretch_T(b0, b0d, b1, b1d, a_lat, tau, t_go)
        T_vert = (t_go if vertical_mode == "flightpath"
                  else hermite4_stretch_T(c0, c0d, c1, c1d, a_vert, tau, t_go))
        # The acceleration bound is not the only one: a profile must also respect the
        # vehicle's CLIMB RATE ceiling, a hard separate limit. Ask for more climb than the
        # aircraft has and the command is clamped downstream, the aircraft falls further
        # behind, the next tick asks for even more, and a feasible approach converges to
        # an impossible one.
        #
        # There is deliberately no matching cap on the lateral RATE. The speed cap is a
        # budget, not an authority limit: what bounds how fast the velocity vector can be
        # slewed sideways is the bank angle, and that is already the accel bound above.
        # Capping the lateral rate by the speed budget instead -- the obvious symmetry --
        # makes a 300 m offset look un-flyable to a 25 m/s aircraft that flies it to
        # 1.6 m, and the resulting "infeasible" pass misses by 47 m.
        # ...and the deceleration has to be one the airframe can give. The peak rate of the
        # ramp is (v_start - v_pass)/T, so this is the check that decides whether a
        # "minimum speed" arrival is reachable at all from where the aircraft actually is.
        # Only the EXCESS over the arrival speed is a demand. Signed, because an aircraft that is
        # already slower than the arrival speed -- most obviously one that has landed and
        # stopped, where the measured along-rate is ~0 -- has nothing left to shed, and
        # measuring (0 - 3.0)/0.35 s calls that 8.6 m/s^2 and declares a touchdown that put
        # the aircraft on its target infeasible.
        shed = max(0.0, v_now_along - v_pass)
        reasons = []
        if shed / max(0.1, t_go) > cfg.max_long_accel_mps2:
            # Can the aircraft still SHED the speed it actually has left, in the time it
            # actually has? The measured along-track rate, not the nominal approach speed:
            # the speed schedule has been shedding that difference all the way down, so by
            # the gate the aircraft is already at the arrival speed and the requirement is
            # nothing. Comparing against the nominal 12 m/s instead made the check read
            # (12-3)/0.35 s = 26 m/s^2 at the gate and declare a perfectly flown approach
            # infeasible in its final tick -- which is what happened the moment this check
            # started being honoured at all, after having been dead code behind an
            # overwriting assignment for as long as it had existed.
            feasible = False
            reasons.append("longitudinal_accel")
        if speed_floor_binding:
            # The requested arrival speed is below what the intercept geometry allows. Say so
            # rather than arriving on time at the wrong angle.
            feasible = False
            reasons.append("intercept_floor")
        if airframe_floor_binding:
            # The requested arrival speed is below what THIS AIRCRAFT will fly. The law still
            # has to clamp and fly the best it can, but the requirement is not met and the
            # plan must not claim it is: `ground_too_slow` asks for a 0.2 m/s touchdown, the
            # iris minimum is 1.5, and reporting that as a successful pass would be reporting
            # that we ignored the requirement.
            feasible = False
            reasons.append("airframe_min_speed")
        if vertical_mode != "flightpath":
            # The rate bound applies in the direction of travel: c1 < c0 is a descent and
            # gets the descent ceiling, which for the quad is 6 m/s where the climb ceiling
            # is 5. Using the climb number for both under-rates the profile a diving
            # approach can actually fly.
            v_rate = cfg.descent_rate_max() if c1 < c0 else cfg.max_climb_rate_mps
            T_vert = max(T_vert, abs(c1 - c0) / max(0.1, v_rate))
        # AND the checks already made, not INSTEAD of them. This line used to ASSIGN `feasible`,
        # which silently discarded every verdict set above it -- the deceleration bound, the
        # geometric intercept floor, the airframe speed floor. Nothing crashed and nothing
        # warned: the law simply stopped reporting infeasible for those, while still clamping
        # the command to the limit it had just failed. `ground_target_too_fast`, whose whole
        # purpose is to check that a geometrically impossible pass is REFUSED, was scoring
        # as flown. An infeasibility check whose result is discarded is worse than no check,
        # because the clamp still happens and looks like the check passing.
        feasible = (feasible
                    and hermite4_needed(b0, b0d, b1, b1d, T_lat, tau) <= a_lat
                    and (vertical_mode == "flightpath"
                         or hermite4_needed(c0, c0d, c1, c1d, T_vert, tau) <= a_vert))
        if hermite4_needed(b0, b0d, b1, b1d, T_lat, tau) > a_lat:
            reasons.append("lateral_accel")
        if (vertical_mode != "flightpath"
                and hermite4_needed(c0, c0d, c1, c1d, T_vert, tau) > a_vert):
            reasons.append("vertical_accel")

        # How far ahead of its own profile to read. `tau_lead` alone is enough for an axis
        # whose terminal RATE the vehicle already holds -- the normal case, since "arrive
        # tangent" wants zero lateral rate and zero climb and the aircraft is doing both.
        # An axis whose terminal rate it does NOT hold -- a demanded flight-path angle, and
        # only that -- has to warn the vehicle by the time it takes to get there, or the
        # rate is still wrong at the gate. That was a dive pass arriving 11 deg shallow on
        # a vehicle that could hold the glide perfectly well. `readable_point` is that
        # statement solved rather than guessed.
        # The read-ahead is the vehicle's LAG and nothing else. Both a shorter read and a
        # longer one are wrong, for opposite reasons, and both were measured here:
        #
        #   * too short -> the aircraft lags the profile and arrives late, with the residual
        #     rate still on it at the gate;
        #   * too long  -> the aircraft leads the profile and arrives EARLY, crossing the
        #     line before the gate with the residual rate on it. A "read further ahead so
        #     the vehicle can shed the terminal rate in time" term does exactly this, and it
        #     is silently self-defeating: 0.7 s of over-lead on a 0.3 s lag put the gate
        #     crossing 0.8 m early with 2 m/s of lateral rate -- 28 deg of track error --
        #     on a profile whose terminal condition was perfectly flyable all along.
        #
        # What the vehicle cannot fly is handled where it belongs, by STRETCHING the
        # profile (T_lat, T_vert) until its acceleration fits, not by reading further ahead.
        u_lat = min(1.0, tau / T_lat)
        u_vert = u_lat if vertical_mode == "flightpath" else min(1.0, tau / T_vert)

        # --- the profile, read a lead time ahead --------------------------------------
        # Regenerated from the measured state every tick, so the only evaluation is at
        # u = tau_lead/T: the same profile, read as though it were already being flown.
        # That read IS the lead compensation -- adding tau_lead * accel on top of it would
        # double-count, and measured as a moving-point pass arriving 21 deg off.
        #
        # Phase of the ramp: how far through the ORIGINAL approach we are. 0 at the start
        # of the approach, 1 at the gate.
        if self._sched_t0 is None:
            # The ramp's own horizon: integrating it over the approach gives
            # T*(v0 + v_pass)/2, so T = 2*(-s)/(v0 + v_pass). Latching the MEASURED
            # time-to-go instead is right whenever the aircraft is already flying the
            # approach at speed and wrong the moment it is not -- a live rig hands over at
            # the end of a transit with the aircraft nearly stopped, the measured horizon
            # comes out at 559 s against a ramp that implies 119, and the resulting stable
            # equilibrium commands 4 m/s for the whole approach with nothing wrong anywhere
            # except the epoch. The measured rate is the right clock for the GEOMETRY
            # profiles (they have to finish when the aircraft arrives); this is the right
            # clock for the PLAN.
            self._sched_t0 = max(
                (-2.0 * s / (self._sched_v0 + v_pass)) if s < 0.0 else t_go,
                cfg.min_t_go_s)
        phase = 1.0 - max(0.0, t_go - cfg.tau_lead_s) / self._sched_t0
        # phase runs 0 -> 1 as the approach runs out, so the speed runs v_start -> v_pass
        # over the COMPLEMENT of the ramp. Adding the ramp (rather than subtracting it)
        # makes both ends of the approach the minimum speed, which is the opposite of a
        # decelerating approach and produced a commanded 3 m/s on the first tick of a
        # 12 m/s approach.
        vel_along = self._sched_v0 - (self._sched_v0 - v_pass) * ramp(phase)
        v_start = self._sched_v0
        _b, vel_right, _ = hermite4(b0, b0d, b1, b1d, T_lat, u_lat)
        if vertical_mode == "flightpath":
            # The glide IS the requirement here, so it is commanded as a rate. Running it
            # as a profile to an altitude arrival instead makes the law fight itself: the
            # vertical profile's horizon is the ALONG-track time, the vertical clock is
            # the aircraft's own descent, and the two disagree by a fraction of a second
            # on any approach that is not exactly on the slope. The profile then decides
            # it must arrive at the point's altitude at the gate, so it flattens the last
            # two seconds of a glide it was otherwise tracking to a centimetre -- measured
            # as 6 to 11 deg of flight-path error with position, altitude and lateral
            # angle all perfect.
            #
            # The residual altitude is trimmed PERPENDICULAR TO THE GLIDE, not straight
            # down to the point. The vertical distance to the point is not the error on a
            # glide -- most of it is the glide itself -- so trimming it commands a dive
            # onto a point the aircraft is already correctly aimed at: measured as 20 m
            # of altitude error on an approach whose geometry was exactly right, because
            # the trim kept adding the slope that was already being flown.
            # The descent rate that HOLDS the demanded flight-path angle is the current
            # along-track speed times tan(fp) -- not the arrival speed. Using the arrival
            # speed here means the glide is set for how fast the aircraft will be at the
            # gate, so the approach flattens to a crawl over the last minute and the gate is
            # crossed hundreds of metres above the point: measured as 226 m of altitude
            # error with the lateral geometry and the track angle both exact to 0.0.
            glide = -vel_along * math.tan(math.radians(fp))
            dev = c0 + math.tan(math.radians(fp)) * s
            bias = -dev / max(0.5, t_go)
            # The trim's cap is an absolute rate. It was also tried as a fraction of the pass
            # speed, to bound the angle it costs; measured, that is worse (see
            # `fp_alt_trim_max_frac`) because the trim is a closure term, so it needs
            # the absolute rate. The fraction applies only if one is explicitly set.
            trim_cap = cfg.fp_alt_trim_max_mps
            if cfg.fp_alt_trim_max_frac is not None:
                trim_cap = min(trim_cap, cfg.fp_alt_trim_max_frac * abs(v_pass))
            if abs(bias) > trim_cap:
                bias = math.copysign(trim_cap, bias)
                reasons.append("fp_trim")
                if cfg.fp_trim_is_infeasible:
                    feasible = False
            vz_cmd = glide + bias
        else:
            _c, vz_cmd, _ = hermite4(c0, c0d, c1, c1d, T_vert, u_vert)

        if post_pass:
            # The contact is behind: there is no terminal left to aim a profile at, so
            # hold the LINE rather than the contact -- the absolute lateral rate goes to
            # zero and a small gain trims the remainder. Holding the contact instead
            # (zeroing the RELATIVE rate) makes the aircraft curve away to keep station
            # with a drifting point, which is the opposite of a flyby: the exit track
            # comes out 20 deg off the pass line and stays there.
            vel_right = -cfg.fly_on_lateral_gain * b0
            vz_cmd = -cfg.fly_on_vertical_gain * c0
        if cfg.rate_trim_gain and not post_pass:
            vel_right += cfg.rate_trim_gain * b0d

        # --- to an absolute velocity command ------------------------------------------
        # Already absolute: the profile was built in the aircraft's own frame. Only the
        # total has to be capped, because a vehicle cannot spend more speed than the pass
        # asked for however hard the correction is.
        mag = math.hypot(vel_along, vel_right)
        # The budget for spending speed to buy lateral correction -- and, separately, the
        # vehicle's actual ceiling on it. They are different numbers and conflating them is
        # the lateral twin of the vertical clamp: `max_speed_overshoot` is how much faster
        # the law may go than the pass asks for, while `max_speed_mps` is what the vehicle
        # will do. PX4 enforces the second one in the same `PositionControl` call that
        # enforces the vertical limits, so a law that only knows the first asks for a
        # manoeuvre the autopilot silently refuses.
        #
        # Measured live: nulling a 300 m cross-track offset needs 14.8 m/s from an aircraft
        # whose `MPC_XY_VEL_MAX` is 12. The reference plant allows 30 m/s, so the offline
        # sweep scored the case at 0.01 m; on the vehicle the command was above the limit on
        # 47 % of ticks and the aircraft was pinned at exactly 12.00 m/s throughout, and it
        # crossed the gate 21 m wide. Nothing errored. Same failure, same line of C++, other
        # axis.
        cap = max(v_start, v_pass) * cfg.max_speed_overshoot
        if cfg.max_speed_mps is not None:
            cap = min(cap, cfg.max_speed_mps)
        speed_limited = mag > cap
        if speed_limited:
            reasons.append("speed_cap")
        if speed_limited:
            # The vehicle cannot spend more than the pass asked for, so the profile as
            # shaped is not the profile it will fly. Report it -- a scenario whose whole
            # error is "the airframe could not afford it" should not be averaged in with
            # scenarios where the guidance was the problem.
            feasible = False
        if speed_limited:
            vel_along *= cap / mag
            vel_right *= cap / mag

        climb_want = -vz_cmd - tgt_vel[2]
        # Asymmetric: the descent ceiling and the climb ceiling are separate numbers, and
        # which one applies depends on which way the aircraft is being asked to go. Using one
        # symmetric clamp here is what let a 4 m/s descent be commanded to an autopilot
        # flying it at 1.5.
        v_lim = cfg.vertical_rate_limit(climb_want)
        if abs(climb_want) > v_lim:
            # A demanded flight-path angle the aircraft cannot hold is a requirement it
            # cannot meet, and that is worth saying out loud rather than silently
            # flattening the pass: a 15 deg dive at 25 m/s needs 6.7 m/s of descent and a
            # trainer limited to 5 will arrive 66 m high and 3.7 deg shallow, every time.
            feasible = False
            reasons.append("vertical_rate")
        climb_cmd = max(-v_lim, min(v_lim, climb_want))
        vel = (
            vel_along * d[0] + vel_right * r[0],
            vel_along * d[1] + vel_right * r[1],
            -climb_cmd,
        )
        heading = math.degrees(math.atan2(vel[1], vel[0])) % 360.0

        return PassPlan(
            vel_ned=vel,
            climb_rate_mps=climb_cmd,
            speed_mps=math.hypot(vel[0], vel[1]),
            along_track_m=s,
            cross_track_m=e,
            along_rate_mps=s_dot,
            cross_rate_mps=e_dot,
            cross_rate_cmd_mps=vel_right,
            along_rate_cmd_mps=vel_along,
            speed_limited=speed_limited,
            intercept_floor_mps=intercept_floor,
            zev_feasible=zev_ok,
            zev_t_go_s=zev_t,
            airframe_min_speed_mps=v_min,
            airframe_floor_binding=airframe_floor_binding,
            intercept_floor_binding=speed_floor_binding,
            t_go_s=t_go,
            profile_T_s=max(T_lat, T_vert),
            profile_T_lat_s=T_lat,
            profile_T_vert_s=T_vert,
            feasible=feasible,
            reasons=tuple(reasons),
            heading_deg=heading,
            vertical_mode=vertical_mode,
            post_pass=post_pass,
            target_ned=tgt,
            target_vel_ned=tgt_vel,
        )


# ----------------------------------------------------- the reference plant
#
# Not part of the law -- the yardstick that says how much of a miss is guidance and how
# much is vehicle lag, and what makes `tau_lead_s` tunable without a flight. The
# lateral loop is a first-order lag on the commanded cross-track rate, because that is
# what a vehicle actually is: a commanded turn that takes roll time plus a lag before it
# shows up as the right rate.
#
# The lag matters more than it looks. With a *perfect* tracker (rate obeyed instantly)
# the lead term -- which adds tau_lead * e_ddot to the commanded rate -- is positive
# feedback with nothing to damp it, and the closed loop drifts: measured here as a
# ground track that had converged to 29.5 deg against a 30 deg requirement and then
# walked on to 33.8 deg after the crossing. Every lead compensator has this property.
# So the reference plant models the lag, and `tau_lead_s` is then tuned against the
# plant's time constant, which is the only relationship that means anything.

@dataclass
class PlantConfig:
    """How fast the vehicle answers a commanded velocity. First-order, per axis.

    The limits are an Iris-class quadrotor and a small fixed wing, not arbitrary: the
    quad's vertical rate matters because a climb-rate ceiling the guidance does not know
    about turns a flyable approach into an impossible one (see the rate feasibility check
    in `update`), and that only shows up when the two numbers are within a factor of two.
    """
    tau_horizontal_s: float = 1.2
    tau_vertical_s: float = 1.0
    max_speed_mps: float = 60.0
    max_climb_rate_mps: float = 6.0
    # The descent ceiling, like the guidance's. Defaults to the climb ceiling, which is the
    # right reading for a trainer whose envelope is close to symmetric.
    max_descent_rate_mps: float | None = None
    # Fixed-wing horizontal model. A first-order lag on the velocity VECTOR is right for a
    # multirotor, which tilts and pushes in any direction, and wrong for a wing, which can
    # only change course by banking. The old derivation fed V/(g tan bank) -- the time to
    # turn through one RADIAN at full bank -- in as that lag, so a 1 m/s correction took
    # 11.65 s on an Akinci; no law closes a 1 m miss through that, and no aircraft has it.
    # With `turn_model` the horizontal channel is instead what the PX4 stack actually does:
    #   course loop  NPFG: turn-rate demand = course error / course_tau
    #   roll loop    bank follows its demand with FW_R_TC, clipped at FW_R_LIM
    #   kinematics   course rate = g tan(bank) / V   (so the accel limit is g tan(bank))
    #   speed        first-order on tau_horizontal_s, clipped at max_long_accel, never
    #                below min_speed (a wing does not stop to turn)
    turn_model: bool = False
    course_tau_s: float = 0.67
    roll_tau_s: float = 0.4
    max_bank_deg: float = 35.0
    max_long_accel_mps2: float = 1.5
    min_speed_mps: float = 0.0

    @staticmethod
    def for_quad() -> "PlantConfig":
        # An iris-class quad can climb at better than 8 m/s and slew its velocity vector
        # in a couple of tenths; the guidance's own 6 m/s clamp is the tighter of the two.
        #
        # tau_vertical_s is MEASURED, not guessed: from the vertical step ladder
        # (scripts/vstep.py) with PX4's limits raised, ten-to-ninety per cent crossing is
        # 0.10..0.35 s across steps of +-1 to +-6 m/s, which for a first-order response is
        # tau = 0.121 s. The 20 Hz telemetry quantises that to about +-0.05 s, so 0.15 is
        # taken as a slightly conservative round number. The previous 0.35 s was three times
        # the measurement and made the plant look slower than the vehicle.
        return PlantConfig(tau_horizontal_s=0.30, tau_vertical_s=0.15,
                           max_speed_mps=30.0, max_climb_rate_mps=8.0,
                           max_descent_rate_mps=8.0)

    def descent_rate_max(self) -> float:
        return self.max_climb_rate_mps if self.max_descent_rate_mps is None \
            else self.max_descent_rate_mps

    def vertical_rate_limit(self, cmd_down_mps: float) -> float:
        """The plant's limit on a NED DOWN rate (positive down) in the direction it points."""
        return self.descent_rate_max() if cmd_down_mps >= 0.0 else self.max_climb_rate_mps


def simulate_pass(
    guidance: TerminalProfileGuidance,
    own_ned,
    own_vel_ned,
    spec: PassSpec,
    target_fn,
    dt: float = 0.05,
    duration_s: float = 40.0,
    plant: PlantConfig | None = None,
):
    """Run the law closed-loop against the reference plant.

    `target_fn(t) -> (ned, vel_ned)` makes the point move however the experiment needs.
    Returns one (t, state, velocity, plan) row per step.
    """
    plant = plant or PlantConfig()
    p = tuple(own_ned)
    v = tuple(own_vel_ned)
    ah = 1.0 - math.exp(-dt / plant.tau_horizontal_s)
    av = 1.0 - math.exp(-dt / plant.tau_vertical_s)
    ar = 1.0 - math.exp(-dt / max(1e-3, plant.roll_tau_s))
    bank_max = math.radians(plant.max_bank_deg)
    # Wing state: speed, course and bank. Only read when `turn_model` is set.
    spd = math.hypot(v[0], v[1])
    crs = math.atan2(v[1], v[0])
    bank = 0.0

    out = []
    for i in range(int(duration_s / dt) + 1):
        t = i * dt
        tgt, tgt_vel = target_fn(t)
        plan = guidance.update(p, v, spec, t, target_ned=tgt, target_vel_ned=tgt_vel,
                               target_predict=(lambda dt: target_fn(t + dt)))
        out.append((t, p, v, plan))

        cmd = plan.vel_ned
        sp = math.hypot(cmd[0], cmd[1])
        if sp > plant.max_speed_mps:
            cmd = (cmd[0] * plant.max_speed_mps / sp, cmd[1] * plant.max_speed_mps / sp, cmd[2])
        # cmd[2] is a NED DOWN rate, so the clamp is a plain limit on it -- negating here
        # silently flies the aircraft the wrong way vertically. Asymmetric, because the
        # descent and climb ceilings are different numbers.
        v_lim = plant.vertical_rate_limit(cmd[2])
        cmd = (cmd[0], cmd[1], max(-v_lim, min(v_lim, cmd[2])))
        if plant.turn_model:
            spd_cmd = max(plant.min_speed_mps, math.hypot(cmd[0], cmd[1]))
            crs_cmd = math.atan2(cmd[1], cmd[0]) if math.hypot(cmd[0], cmd[1]) > 1e-6 else crs
            err = (crs_cmd - crs + math.pi) % (2.0 * math.pi) - math.pi
            v_ref = max(1.0, spd)
            bank_cmd = math.atan(v_ref * (err / max(1e-3, plant.course_tau_s)) / G)
            bank_cmd = max(-bank_max, min(bank_max, bank_cmd))
            bank += ar * (bank_cmd - bank)
            crs += G * math.tan(bank) / v_ref * dt
            dv = (spd_cmd - spd) * ah / dt
            dv = max(-plant.max_long_accel_mps2, min(plant.max_long_accel_mps2, dv))
            spd = max(plant.min_speed_mps, spd + dv * dt)
            v = (spd * math.cos(crs), spd * math.sin(crs), v[2] + av * (cmd[2] - v[2]))
        else:
            v = (v[0] + ah * (cmd[0] - v[0]),
                 v[1] + ah * (cmd[1] - v[1]),
                 v[2] + av * (cmd[2] - v[2]))
        p = (p[0] + v[0] * dt, p[1] + v[1] * dt, p[2] + v[2] * dt)
    return out
