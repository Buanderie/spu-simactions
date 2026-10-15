"""What each airframe can actually do, and where that number came from.

This module exists because of a measured failure, and the shape of it is the point.

The vertical channel was silently clipped to 1.5 m/s on the way down and 3.0 m/s on the
way up, and neither number appeared anywhere in the guidance, the scenarios or the plant --
they lived in PX4, in a parameter nobody was reading. The guidance had a single symmetric
`max_climb_rate_mps`, so it could not have represented a limit that differs by a factor of
two between the two directions even if it had known about it. Three separate places needed
to agree on one number and none of them did:

    the autopilot    MPC_Z_VEL_MAX_DN / _UP   -- what PX4 will actually fly
    the guidance     max_climb/descent_rate   -- what the law will plan against
    the plant        max_climb/descent_rate   -- what the reference model will reproduce

When they disagree the run does not fail, it lies: the law is exact against the plant, the
plant is exact against the FDM, and the aircraft is flown at neither number.

So there is now ONE place where each of those numbers is decided and every other place asks
for it:

    config/profiles/<name>.json    the aircraft. Published performance (mass, span, stall,
                                   cruise, climb, top speed, and for the jets the engine's
                                   thrust rating) and everything scripts/derive_profile.py
                                   derived from it: wing area, the polar, thrust, drag.
    engine/fdm/profiles.py         plane_envelope() -- the PX4 airspeed, pitch, vertical-rate,
                                   trim, loiter and NPFG parameters, RECOMPUTED from the
                                   profile at load time, and px4_params() -- what actually
                                   gets pushed. Both are called from here, so the guidance
                                   plans against the parameters the autopilot is given
                                   rather than against a second opinion about them.
    config/px4/fixedwing.json      what is no aircraft's business: the harness settings, the
    config/px4/multirotor.json     throttle range, the TECS behaviours, PX4's control-loop
                                   tuning, and the stock iris envelope.
    engine/fdm/fixedwing.py        the plant's own coefficients, imported from the FDM
    engine/fdm/multicopter.py      rather than copied, so this repo's reference plant and the
                                   sim cannot disagree about the machine they model.

This file used to restate all four. It said "35 deg of bank" and "15 m/s" for every aircraft
in the fleet, which is exactly the class of error it was written to prevent: a number that
looks like a measurement and is really a memory.

The one thing that cannot be loaded is a measurement. `MEASURED` below is the step ladder and
the offline bench of THIS rig; it overrides the derived numbers for the two aircraft it was
taken on and is the anchor every other untestable number is expressed as a ratio of.
"""
import importlib.util
import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass

from .guidance import GuidanceConfig, PlantConfig, G

SIMUAV_ROOT = pathlib.Path(os.environ.get("SIMUAV_ROOT", "/home/nis/dev/swarmpu-simuav"))
SIMUAV_PROFILES = pathlib.Path(os.environ.get(
    "SIMUAV_PROFILE_DIR", SIMUAV_ROOT / "config" / "profiles"))
PX4_CONFIG_DIR = pathlib.Path(os.environ.get(
    "SIMUAV_PX4_CONFIG_DIR", SIMUAV_ROOT / "config" / "px4"))
RHO = 1.225


def _simuav():
    """The swarmpu-simuav package, imported from its checkout.

    Read from the source rather than copied into this repo. A copy is how the trainer's turn
    radius came to differ from the sim's, and how an airframe whose coefficients live in a
    profile JSON ended up with a guidance law that had never heard of profiles.
    """
    if str(SIMUAV_ROOT) not in sys.path:
        sys.path.insert(0, str(SIMUAV_ROOT))
    from engine.fdm import profiles, fixedwing, multicopter     # noqa: E402
    return profiles, fixedwing, multicopter


def _px4(kind: str) -> dict:
    """config/px4/fixedwing.json or multirotor.json."""
    name = "fixedwing.json" if kind == "plane" else "multirotor.json"
    path = PX4_CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: set SIMUAV_PX4_CONFIG_DIR to the "
                                f"swarmpu-simuav config/px4 directory")
    return json.loads(path.read_text())


_simuav_profiles, _fixedwing, _multicopter = _simuav()
PLANE_HARNESS = _px4("plane")
QUAD_STOCK = _px4("multirotor")["stock"]

# The PX4 fixed-wing loops every simuav plane flies with. A velocity command reaches the wing
# through FixedWingModeManager::control_auto_velocity (vendored offboard patch) and NPFG, which
# turns a course error into a turn-rate demand; the roll loop then follows the bank that turn
# needs. NPFG reduces to the L1 law with L1 = damping * period * V / pi, whose course response
# is first order with tau = damping * period / (2 pi), and the roll loop adds FW_R_TC.
#
# NPFG_PERIOD is per aircraft -- it is recomputed from each profile's loiter radius -- and so
# is FW_R_LIM's companion in the envelope; what is read here is the damping and the roll time
# constant, which are PX4's loops' own and not any aircraft's.
FW_NPFG_DAMPING = PLANE_HARNESS["float"]["NPFG_DAMPING"]
FW_R_TC_S = PLANE_HARNESS["float"]["FW_R_TC"]

# ------------------------------------------------------------------ policy constants
#
# The only numbers in this file that are not measurements, published figures, or values read
# out of a config file or a profile. Each is dimensionless and each has a reason; none of them
# is an aircraft capability.
PLANT_SPEED_SLACK = 1.2          # the reference plant may be commanded past the autopilot's cap
PLANT_CLIMB_SLACK = 1.6
PLANT_DESCENT_SLACK = 1.35
TAU_LEAD_BOUNDS_S = (0.2, 12.0)  # see Airframe.guidance_config


@dataclass(frozen=True)
class Airframe:
    """What one aircraft can do, and how we know."""

    name: str
    kind: str                       # "quad" | "plane", matching simuav's --airframe
    title: str = ""
    # Ceilings. Descent and climb are separate because real aircraft and PX4 both treat
    # them separately; symmetric is a special case, not the default.
    max_descent_mps: float = 6.0
    max_climb_mps: float = 5.0
    max_speed_mps: float = 25.0
    # The slowest the aircraft can be commanded to fly, and still hold an attitude. Not a
    # preference: a fixed wing below ~1.2x stall has no roll authority, and a quadrotor
    # starts losing its own velocity estimate in the downwash.
    min_speed_mps: float = 1.5
    # How fast it answers a commanded velocity. Measured from the step ladder where we
    # could, estimated from class where we could not.
    tau_horizontal_s: float = 0.30
    tau_vertical_s: float = 0.15
    max_bank_deg: float = 30.0
    max_vertical_accel_mps2: float = 4.0
    max_long_accel_mps2: float = 4.0
    source: str = ""
    confidence: str = "estimated"   # "measured" | "published" | "estimated"
    notes: str = ""

    def guidance_config(self) -> GuidanceConfig:
        c = (GuidanceConfig.for_quad() if self.kind == "quad" else GuidanceConfig())
        c.max_climb_rate_mps = self.max_climb_mps
        c.max_descent_rate_mps = self.max_descent_mps
        c.max_bank_deg = self.max_bank_deg
        c.max_vertical_accel_mps2 = self.max_vertical_accel_mps2
        c.max_long_accel_mps2 = self.max_long_accel_mps2
        c.min_speed_mps = self.min_speed_mps
        c.max_speed_mps = self.max_speed_mps
        # tau_lead is how far ahead of the profile the law reads it, and it should match
        # the vehicle's response: reading further ahead than the vehicle can be told is
        # positive feedback with nothing to damp it (see the reference-plant note in
        # guidance.py). Bounded only at the extremes -- and the upper bound has to be a
        # real one, because a fixed wing's turn response is set by SPEED and spans 7 s to
        # 12 s across this fleet. Clamping it at 1.2 s, which is right for the trainer, gave
        # a 92 m/s Akinci a lead of 1.2 s against a turn response of 11.7 s and would have
        # made every one of its passes report the same closed-loop drift the guide warns
        # about.
        c.tau_lead_s = round(max(TAU_LEAD_BOUNDS_S[0],
                                min(TAU_LEAD_BOUNDS_S[1], self.tau_horizontal_s)), 2)
        return c

    def plant_config(self) -> PlantConfig:
        p = (PlantConfig.for_quad() if self.kind == "quad" else PlantConfig())
        p.max_climb_rate_mps = self.max_climb_mps * PLANT_CLIMB_SLACK
        p.max_descent_rate_mps = self.max_descent_mps * PLANT_DESCENT_SLACK
        p.max_speed_mps = self.max_speed_mps * PLANT_SPEED_SLACK
        p.tau_horizontal_s = self.tau_horizontal_s
        p.tau_vertical_s = self.tau_vertical_s
        if self.kind == "plane":
            # A wing turns by banking; see PlantConfig.turn_model.
            p.turn_model = True
            p.course_tau_s = _course_tau_s(self)
            p.roll_tau_s = FW_R_TC_S
            p.max_bank_deg = self.max_bank_deg
            p.max_long_accel_mps2 = self.max_long_accel_mps2
            p.min_speed_mps = self.min_speed_mps
        return p


# The PX4 NPFG_PERIOD this airframe is actually flown at, i.e. the one its profile's envelope
# computes. Held on the Airframe as its source string rather than in a table of constants, so
# a profile that changes the loiter radius changes the plant with it.
_NPFG_PERIOD_S: dict = {}


def _course_tau_s(cap: Airframe) -> float:
    """NPFG's first-order course response, damping * period / (2 pi)."""
    period = _NPFG_PERIOD_S.get(cap.name, 1.0)
    return round(FW_NPFG_DAMPING * period / (2.0 * math.pi), 2)


# ------------------------------------------------------------------ the measured two
#
# These cannot be loaded from anywhere: they are step-ladder and offline-bench measurements of
# THIS rig, so they live here, they override the derived numbers for the aircraft they were
# taken on, and they are the anchor everything else is expressed as a ratio of. Provenance for
# each is in the `source` field, which is why they are `confidence="measured"`.

MEASURED = {
    "iris": dict(
        # scripts/vstep.py against swarmpu-simuav's MulticopterFDM with PX4's limits raised:
        # commanded +6 m/s flies at +6.02, -4 at -3.99, so the FDM has far more vertical
        # authority than PX4's stock envelope admits (config/px4/multirotor.json: 1.5 m/s down,
        # 3.0 m/s up, 12 m/s across). tau_vertical_s is the 10-90% crossing time of 0.121 s
        # rounded up.
        max_descent_mps=6.0, max_climb_mps=5.0, max_speed_mps=18.0, min_speed_mps=1.5,
        tau_horizontal_s=0.30, tau_vertical_s=0.15, max_bank_deg=30.0,
        max_vertical_accel_mps2=4.0, max_long_accel_mps2=4.0,
        source="measured on this rig by scripts/vstep.py against swarmpu-simuav's "
               "MulticopterFDM, with PX4's limits raised; see docs/FINDINGS.md section 4",
        confidence="measured",
        notes="The bank limit is PX4's stock MPC_TILTMAX_AIR (config/px4/multirotor.json) "
               "reduced to what the measured step ladder held; the 18 m/s ceiling is the "
               "plant's own drag limit and REPLACES PX4's stock MPC_XY_VEL_MAX of 12, which "
               "PX4 clipped silently on 47% of ticks."),
    "trainer": dict(
        # Offline bench, swarmpu-simuav: climb 4.6 m/s full throttle, 3.78 wings-level, 3.36
        # at 20 deg bank, 2.40 at 35 deg, turn radius 44 m at cruise; sink in a committed
        # delivery dive ~4 m/s.
        #
        # These are what the aircraft SUSTAINS; config/profiles/trainer.json's tuned PX4
        # envelope is what it is TOLD it may do, and the two differ deliberately. PX4's
        # FW_T_SINK_MAX of 14 m/s is dive authority, needed for the delivery run (the
        # stock 2.7 m/s could not follow the glideslope and the aircraft flew past the
        # crate); planning a normal pass against 14 m/s of sink would be planning to dive
        # every time. Where the two differ, this table wins, because it is a measurement.
        max_climb_mps=3.8, max_descent_mps=4.0, min_speed_mps=11.0, max_speed_mps=30.0,
        tau_horizontal_s=1.2, tau_vertical_s=1.0,
        max_vertical_accel_mps2=2.0, max_long_accel_mps2=1.5,
        source="simuav engine/fdm/fixedwing.py Airframe defaults and "
               "config/profiles/trainer.json's tuned envelope; climb 'measured on the offline "
               "bench' at 4.6 m/s full throttle, 3.78 wings-level, 3.36 at 20 deg bank, 2.40 "
               "at 35 deg; turn radius 44 m at cruise",
        confidence="measured",
        notes="The bank limit is FW_R_LIM because climb rate collapses with bank on this "
               "airframe -- 0.56 m/s at the stock 50 deg -- so a steeper turn is a turn that "
               "cannot climb, and every flyby pass ends in a descent. The acceleration "
               "limits are the values bench.py was tuned against, and they are NOT derived "
               "from the plant's surplus thrust the way a published aircraft's are: this "
               "airframe's thrust_max_n is a tuning parameter for the model's cruise speed, "
               "so its arithmetic surplus (3.6 m/s^2) says nothing about what the guidance "
               "should plan with."),
}

# Aliases, so a simuav profile name that is not in the table fails with a useful message
# rather than a bare KeyError.
ALIASES = {"quad": "iris", "plane": "trainer"}


def _stall_speed(mass_kg, area_m2, cl_max, rho=RHO) -> float:
    """W = 0.5 rho V^2 S CL_max, i.e. the relation derive_profile.py inverts."""
    return math.sqrt(2.0 * mass_kg * G / (rho * area_m2 * cl_max))


def _cruise_drag(mass_kg, area_m2, cd0, k_ind, v, rho=RHO) -> float:
    """q S (CD_0 + k CL^2) at v, CL = 2W/(rho V^2 S): the profile's own drag equation."""
    q_s = 0.5 * rho * v * v * area_m2
    cl = 2.0 * mass_kg * G / (rho * v * v * area_m2)
    return q_s * (cd0 + k_ind * cl * cl)


def _thrust_limited_speed(mass_kg, area_m2, cd0, k_ind, thrust_n, rho=RHO) -> float | None:
    """Level-flight speed at which thrust balances drag on the parabolic polar.

    The high root of 0.5 rho V^2 S CD_0 + k W^2 / (0.5 rho V^2 S) = T. Only meaningful where
    thrust does not fall off with speed -- the jet types, whose engine rating is published.
    A piston engine gives P/V, so this overstates a propeller aircraft's top speed; the
    profile says which case it is in `performance.derived.thrust_is_published`.
    """
    W = mass_kg * G
    a = 0.5 * rho * area_m2 * cd0
    b = k_ind * W * W / (0.5 * rho * area_m2)
    disc = thrust_n * thrust_n - 4.0 * a * b
    return math.sqrt((thrust_n + math.sqrt(disc)) / (2.0 * a)) if disc > 0 else None


def _plane_from_profile(prof: dict, name: str) -> Airframe:
    """One plane's capability table, from its profile and the envelope computed from it.

    The vertical and speed limits are the PX4 parameters themselves, read from
    `profiles.px4_params()` -- the very dict that goes out over MAVLink. That is the whole
    point of the module: the law plans against the numbers the autopilot was given, so there
    is no second opinion to drift.

    What is left is arithmetic on the published performance:

        tau_horizontal  NPFG's course response (damping x period / 2 pi, with the period its
                        own loiter radius implies) plus the roll loop's FW_R_TC.
        tau_vertical    the trainer's measured value, carried over as the class value: TECS
                        is shared, so a vertical response is a property of the control loop.
        max long accel  surplus thrust at cruise over the weight -- the only longitudinal
                        acceleration the aircraft has in level flight, and therefore the limit
                        on shedding an intercept's approach speed. Capped by the trainer's.
        max vert accel  the climb it has to reach within its own vertical time constant,
                        capped by the trainer's.
        min speed       PX4's own declared minimum in-flight speed, 1.15 x stall.
    """
    perf = prof.get("performance") or {}
    af = prof["airframe"]
    der = perf.get("derived") or {}
    floats, _ints = _simuav_profiles.px4_params(prof)
    cruise = float(perf["cruise_ms"])
    _NPFG_PERIOD_S[name] = floats["NPFG_PERIOD"]

    mass = float(af["mass_kg"])
    thrust = float(af["thrust_max_n"])
    drag = _cruise_drag(mass, float(af["wing_area_m2"]), float(af["cd0"]),
                        float(af["cd_induced_k"]), cruise)
    surplus = max(0.0, (thrust - drag) / (mass * G))       # a fraction of weight
    v_max = floats["FW_AIRSPD_MAX"]
    if der.get("thrust_is_published"):
        polar = _thrust_limited_speed(mass, float(af["wing_area_m2"]), float(af["cd0"]),
                                      float(af["cd_induced_k"]), thrust)
        if polar:
            v_max = min(v_max, polar)
    tau_v = MEASURED["trainer"]["tau_vertical_s"]
    return Airframe(
        name=name, kind="plane", title=prof.get("title", name),
        max_descent_mps=floats["FW_T_SINK_MAX"], max_climb_mps=floats["FW_T_CLMB_MAX"],
        max_speed_mps=round(v_max, 2), min_speed_mps=round(floats["FW_AIRSPD_MIN"], 2),
        tau_horizontal_s=round(_course_tau_s(Airframe(name=name, kind="plane")) + FW_R_TC_S, 2),
        tau_vertical_s=tau_v, max_bank_deg=floats["FW_R_LIM"],
        max_vertical_accel_mps2=round(min(floats["FW_T_CLMB_MAX"] / tau_v,
                                          MEASURED["iris"]["max_vertical_accel_mps2"]), 2),
        max_long_accel_mps2=round(min(surplus * G, MEASURED["iris"]["max_long_accel_mps2"]), 2),
        source=(f"derived from the simuav profile's PUBLISHED performance (cruise "
                f"{cruise:.0f} m/s, stall {perf['stall_ms']} m/s, climb "
                f"{perf['climb_ms']} m/s, thrust {thrust:.0f} N); the vertical, speed and "
                f"bank limits ARE the PX4 parameters computed from that profile by "
                f"engine/fdm/profiles.py:plane_envelope (trim {cruise:.0f}, stall "
                f"{perf['stall_ms']}, max {floats['FW_AIRSPD_MAX']:.0f} m/s, NPFG period "
                f"{_NPFG_PERIOD_S[name]:.1f} s), and the response times are PX4's own loops"),
        confidence=prof.get("confidence", "published"),
        notes=prof.get("notes", ""))


def _quad_from_profile(prof: dict, name: str) -> Airframe:
    """A multirotor's capability table.

    There is no aerodynamics to derive, so what is published is the mass and the thrust, and
    everything else follows from the thrust-to-weight the profile computed:

        climb/descent  the Iris's MEASURED 5.0 / 6.0 m/s scaled by the surplus above hover
                       through the same ratio the iris was measured at. A quadrotor at a T/W
                       surplus climbs at roughly g (TW - 1); the measured iris manages less
                       than that, and the factor between is the one number here.
        max speed      the published top speed, which every quad profile carries; without one,
                       the plant's own drag limit from its thrust and drag_body /
                       drag_quad_body coefficients -- i.e. how fast this FDM could take it.
        min speed, bank, the two taus, the accelerations
                       the Iris's measured values. They are properties of PX4's multirotor
                       loops and of the measured tilt limit, not of how big the aircraft is.
    """
    perf = prof.get("performance") or {}
    af = prof["airframe"]
    der = perf.get("derived") or {}
    iris = _fixedwing and MEASURED["iris"]
    mass = float(af["mass_kg"])
    thrust = float(af["thrust_max_n"]) * len(af["rotors"])
    tw = der.get("thrust_to_weight") or thrust / (mass * G)
    iris_af = _multicopter.QuadAirframe()
    iris_tw = (len(iris_af.rotors) * iris_af.thrust_max_n) / (iris_af.mass_kg * G)
    scale = max(0.0, tw - 1.0) / (iris_tw - 1.0)
    v_max = perf.get("max_speed_mps")
    basis = "the profile's published top speed"
    if not v_max:
        d1 = float(af["drag_body"][0])
        d2 = float((af.get("drag_quad_body") or [0.0, 0.0, 0.0])[0])
        v_max = ((-d1 + math.sqrt(d1 * d1 + 4.0 * d2 * thrust)) / (2.0 * d2)
                 if d2 > 0 else thrust / d1 if d1 > 0 else iris["max_speed_mps"])
        basis = "the plant's own drag limit (no published top speed)"
    return Airframe(
        name=name, kind="quad", title=prof.get("title", name),
        max_descent_mps=round(iris["max_descent_mps"] * scale, 2),
        max_climb_mps=round(iris["max_climb_mps"] * scale, 2),
        max_speed_mps=round(float(v_max), 2), min_speed_mps=iris["min_speed_mps"],
        tau_horizontal_s=iris["tau_horizontal_s"], tau_vertical_s=iris["tau_vertical_s"],
        max_bank_deg=iris["max_bank_deg"],
        max_vertical_accel_mps2=iris["max_vertical_accel_mps2"],
        max_long_accel_mps2=iris["max_long_accel_mps2"],
        source=(f"PUBLISHED mass ({mass:.2f} kg) and thrust-to-weight ({tw:.2f}) from the "
                f"simuav profile, scaled against the iris's MEASURED step ladder by the "
                f"thrust surplus above hover; max speed from {basis}; bank and the response "
                f"times are the iris's measurements, because they are properties of PX4's "
                f"multirotor loops (stock MPC_TILTMAX_AIR {QUAD_STOCK['MPC_TILTMAX_AIR']} deg, "
                f"MPC_XY_VEL_MAX {QUAD_STOCK['MPC_XY_VEL_MAX']} m/s -- config/px4/"
                f"multirotor.json)"),
        confidence=prof.get("confidence", "published"),
        notes=prof.get("notes", ""))


def from_profile(name: str) -> Airframe:
    """Build a capability entry from a swarmpu-simuav aircraft profile.

    This is what makes the fleet work for EVERY profile simuav knows rather than for the two
    someone remembered to type in.
    """
    path = SIMUAV_PROFILES / f"{name}.json"
    if not path.exists():
        raise KeyError(f"no simuav profile at {path}; known: "
                       f"{', '.join(sorted(p.stem for p in SIMUAV_PROFILES.glob('*.json')))}")
    prof = _simuav_profiles.load_profile(str(path))
    kind = prof.get("kind")
    perf = prof.get("performance") or {}
    af = prof.get("airframe") or {}
    if kind == "multirotor" and af.get("mass_kg"):
        cap = _quad_from_profile(prof, name)
    elif perf.get("cruise_ms"):
        cap = _plane_from_profile(prof, name)
    elif name in MEASURED:
        # An aircraft with no published performance and no airframe block (the trainer, the
        # Iris, whose profiles are empty on purpose). Its capability is the measurement, with
        # whatever the profile's own PX4 parameters say taking precedence -- for the trainer
        # that is the tuned envelope in config/profiles/trainer.json, which is a measurement
        # too, just one that lives in a file because the autopilot has to be told it.
        floats, _ = _simuav_profiles.px4_params(prof)
        fields = dict(MEASURED[name])
        for field, param in (("max_climb_mps", "FW_T_CLMB_MAX"),
                             ("max_descent_mps", "FW_T_SINK_MAX"),
                             ("max_speed_mps", "FW_AIRSPD_MAX"),
                             ("min_speed_mps", "FW_AIRSPD_MIN"),
                             ("max_bank_deg", "FW_R_LIM")):
            # The profile's parameters fill in whatever this rig did not measure, and never
            # override what it did.
            if param in floats and field not in fields:
                fields[field] = floats[param]
        _NPFG_PERIOD_S[name] = floats.get("NPFG_PERIOD", 1.0)
        cap = Airframe(name=name, kind="plane" if kind == "plane" else "quad",
                       title=prof.get("title", name), **fields)
    else:
        raise KeyError(f"{path}: kind {kind!r} has no published performance and no "
                       f"measurement in MEASURED, so there is nothing to plan against")
    return cap


def available() -> list:
    """Every airframe this repo can fly: every simuav profile."""
    if not SIMUAV_PROFILES.is_dir():
        return sorted(set(MEASURED) | set(ALIASES.values()))
    return sorted({p.stem for p in SIMUAV_PROFILES.glob("*.json")})


def get(name: str) -> Airframe:
    """One airframe by name, or a loud error.

    A fallback here would be actively harmful: the failure mode of a wrong capability
    number is a pass that scores well offline and misses by 60 m on the vehicle, and the
    only defence is that asking for an aircraft nobody measured is an error.
    """
    return from_profile(ALIASES.get(name, name))