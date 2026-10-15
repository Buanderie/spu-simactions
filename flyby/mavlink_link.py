"""MAVLink OFFBOARD link: the same guidance law, flown by PX4 instead of a reference plant.

This is where `guidance.TerminalProfileGuidance` meets a real flight stack. The split is
deliberate and is the main reason the law is testable at all:

    guidance.py            geometry only -- "given where we are and where the point will
                           be, what should the velocity be?"     no MAVLink, no I/O
    mavlink_link.py        transport + vehicle plumbing            no guidance maths

Two things had to be established about PX4 before anything here could work, and both are
recorded because they are not in any docs and both cost a session each:

**1. PX4's OFFBOARD has no airspeed, for a fixed wing.** `SET_POSITION_TARGET_*` carries
no airspeed field, `MAV_CMD_NAV_WAYPOINT` does carry one but only reaches the mission
planner, and `VEHICLE_CMD_DO_CHANGE_SPEED` (MAV_CMD_DO_CHANGE_SPEED) is accepted only in
AUTO and the manual modes -- `FixedWingModeManager::vehicle_command_poll()` ignores it
in OFFBOARD. What TECS actually uses in OFFBOARD is `param.equivalent_airspeed_trim`, i.e.
**FW_AIRSPD_TRIM** (`TECS.cpp`: `if (PX4_ISFINITE(input.equivalent_airspeed) ...)
airspeed = input.equivalent_airspeed; else airspeed = param.equivalent_airspeed_trim;`).

So for a fixed wing the pass speed is a *mission parameter*, pushed with PARAM_SET before
the pass, not a control input. It is equivalent airspeed, so a pass specified as ground
speed over the sea needs the density-ratio correction, or it is out by ~3% at 500 m.
A multirotor has no such gap: `mc_pos_control` flies the commanded velocity magnitude.

**2. Fixed-wing OFFBOARD navigation needs a patched PX4.** Stock v1.18 accepts an OFFBOARD
mode switch and holds it while never turning toward the commanded vector -- four separate
bugs, one per axis of the guidance. swarmpu-simuav vendors the fix as
`extdeps/patches/0001-fixed-wing-offboard-navigation.patch`; a multicopter is unaffected
by all four, which is why the quad path here works on a stock binary.

**Bearing, not velocity, for the fixed wing.** With (1) and (2) settled, the fixed-wing
backend does not send a velocity vector at all: it sends the *bearing* of the plan's
velocity and lets TECS hold FW_AIRSPD_TRIM. `control_auto_velocity()` computes
`target_bearing = atan2(vy, vx)` from whatever velocity the setpoint carries and calls
`navigateBearing()`, so sending the same vector but ignoring its magnitude is exact, not a
hack -- and it keeps the geometry's speed request honest instead of quietly double-scaling
it.
"""
import math
import socket
import struct
import sys
import threading
import time

from pymavlink import mavutil

from . import geo
from .guidance import PassPlan

# PX4's own "API/Offboard" mavlink instance (px4-rc.mavlink) sends to 127.0.0.1:14540 --
# PX4's own default, confirmed in that repo's boot log: "mavlink mode: Onboard ... on udp
# port 18580 ... remote port 14540". It resolves against loopback unconditionally, with or
# without a PX4_NET_INTERFACE setting, which is why swarmpu-simuav's relay joins the PX4
# container's network namespace rather than publishing a port.
PX4_UDP_PORT = 14540
HEARTBEAT_HZ = 2.0          # PX4 rejects an offboard flag below 2 Hz
SETPOINT_HZ = 20.0          # PX4 requires setpoints at >2 Hz to stay in OFFBOARD

# MAV_MODE_FLAG
CUSTOM_MODE_OFFBOARD = 6
OFFBOARD_FLAG = 128
SAFETY_ARMED = 128

# SET_POSITION_TARGET_LOCAL_NED type_mask.
#
# Taken from pymavlink's own constants rather than transcribed. Hand-writing these is a
# trap with a very specific shape: the bits are 1,2,4,8,16,32,64,128,256 then a GAP at
# 512 (FORCE_SET) before 1024. Getting that sequence wrong by one -- which is exactly
# what happened here -- silently sets FORCE_SET while leaving AZ unignored, and PX4 answers
# with a critical "SET_POSITION_TARGET_LOCAL_NED force not supported" for every setpoint,
# rejects them all, never enters OFFBOARD, and disarms. Twenty minutes of "why is my
# aircraft not flying" for a mask that looks plausible written down.
_T = mavutil.mavlink
IGNORE_P = _T.POSITION_TARGET_TYPEMASK_X_IGNORE | _T.POSITION_TARGET_TYPEMASK_Y_IGNORE \
    | _T.POSITION_TARGET_TYPEMASK_Z_IGNORE
IGNORE_V = _T.POSITION_TARGET_TYPEMASK_VX_IGNORE | _T.POSITION_TARGET_TYPEMASK_VY_IGNORE \
    | _T.POSITION_TARGET_TYPEMASK_VZ_IGNORE
IGNORE_A = _T.POSITION_TARGET_TYPEMASK_AX_IGNORE | _T.POSITION_TARGET_TYPEMASK_AY_IGNORE \
    | _T.POSITION_TARGET_TYPEMASK_AZ_IGNORE
IGNORE_R = _T.POSITION_TARGET_TYPEMASK_YAW_IGNORE | _T.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE

VELOCITY_ONLY = IGNORE_P | IGNORE_A | IGNORE_R          # 3527
POSITION_AND_VELOCITY = IGNORE_A | IGNORE_R              # 3520
POSITION_ONLY = IGNORE_V | IGNORE_A | IGNORE_R          # 3576

assert not any(m & mavutil.mavlink.POSITION_TARGET_TYPEMASK_FORCE_SET
               for m in (VELOCITY_ONLY, POSITION_AND_VELOCITY, POSITION_ONLY)), \
    "a FORCE_SET bit in a mask PX4 does not support makes it reject every setpoint"


def _param_value_as_number(value):
    """An integer parameter's value, which MAVLink delivers as a denormal float.

    `PARAM_VALUE.param_value` is typed `float` in every dialect, including v2, so PX4
    carries an integer parameter by writing its BITS into that field. Read as a Python float
    -- a float64, so no width is lost -- an integer comes back as a denormal whose
    mantissa is the integer itself: 2 arrives as 2.80e-45, 10 as 1.40e-44, 1 as 1.40e-45.

    This is not an exotic case. `NAV_DLL_ACT` is an enum, and reading it as 2.8e-45 meant
    the rig spent a long while convinced PX4 wanted a datalink-loss Return-mode failsafe
    (it does: `1030_gazebo-classic_plane` sets it to 2), which in turn gated ARMING on a
    ground-station connection. A number that looked like noise turned out to be a policy
    decision, and the reason it looked like noise is that it was never decoded.

    Detection is by magnitude: a real float parameter is never denormal in practice, and
    the smallest positive float32 is 1.4e-45, so anything below ~1e-30 and non-zero is an
    integer in disguise.
    """
    if value != 0.0 and abs(value) < 1e-30:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        return float(bits)
    return value


# Parameters this rig owns, for every aircraft, because the rig's own link is the ground
# station. Left at PX4's default they get in the way of the experiment rather than helping
# it, and they fail quietly.
#
# `NAV_DLL_ACT = 2` ("Return mode" on GCS link loss) is what `1030_gazebo-classic_plane`
# sets, and it is a trap twice over. The controller here is not a ground station PX4 can
# lose -- it is the thing flying the aircraft -- so the action is meaningless. Worse, in
# `HealthAndArmingChecks/checks/rcAndDataLinkCheck.cpp` the check *gates arming* on
# `NAV_DLL_ACT > 0`, so the wrong value does not cause a failsafe, it prevents the aircraft
# from ever arming: a fixed wing that refuses to arm with "Resolve system health failures
# first" and no other symptom, for as long as you care to watch.
#
# Worth noting how it was found: the parameter read back as 2.8e-45, which looked like
# noise. It was an integer that had never been decoded -- see `_param_value_as_number`.
# (A pitot scale, `COM_ASD_SCALE`, is what this needs next -- see FINDINGS section 10 --
# but it does not exist in this PX4 version: PX4 will not even answer a PARAM_REQUEST_READ
# for it, so the equivalent has to be found before it can be set. It is left out rather
# than left in as a write that silently does nothing.)
RIG_PARAMS = {"NAV_DLL_ACT": 0.0}


class Link:
    """A MAVLink connection to one PX4 SITL instance, and its state."""

    def __init__(self, port=PX4_UDP_PORT, timeout_s=30.0, verbose=True):
        # udpin, not udpout: PX4 *sends* to 14540 and nothing is listening for us there.
        # pymavlink binds the port, learns the source address from the first datagram and
        # then sends back to it, which is the only arrangement that works with PX4's
        # unidirectional-by-configuration onboard link.
        self.conn = mavutil.mavlink_connection(f"udpin:127.0.0.1:{port}", source_system=255,
                                                source_component=190)
        self.verbose = verbose
        self.armed = False
        self.mode = None
        self.lat = self.lon = self.alt = None
        self.vx = self.vy = self.vz = 0.0
        self.airspeed = 0.0
        self.groundspeed = 0.0
        self.heading = 0.0
        self.home = None
        self.est_ok = False
        self.est_reasons = ""
        # Diagnostics for the vertical-channel work: the tilt the attitude controller
        # actually commanded, and the motor outputs it asked for. Without these the only
        # evidence that PX4 clipped a setpoint is that the aircraft did not do it.
        self.pitch = self.roll = self.yaw = 0.0
        self.motors = ()
        self.q = 0.0                       # airframe quality 0..1
        self.voltage = 0.0
        self.params = {}                   # every PARAM_VALUE seen, 16-char id -> float
        # Why PX4 last said no. An arm command that is refused is answered with a
        # COMMAND_ACK and usually a STATUSTEXT naming a health check; without capturing
        # them, "arming TIMED OUT" is all you learn, which is how a preflight check that
        # clears after ten seconds looks identical to one that never will.
        self.last_arm_failure = None
        self.last_ack = None
        self._param_wanted = {}            # id -> deadline, for request_param / read_params
        self._last_hb = 0.0
        self._last_status = ""
        self._status_repeats = 0

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # -- receiving ------------------------------------------------------------
    def pump(self, timeout=0.0):
        """Drain whatever has arrived, without blocking when there is nothing to read.

        `timeout <= 0` must mean NON-BLOCKING and that is not what
        `recv_match(blocking=True, timeout=0)` does: it tests `start + timeout < now` before
        it ever looks at the socket, so it returns None immediately having read nothing.
        Passing 0 there silently makes every call a no-op, the telemetry freezes at whatever
        it was when the link came up, and the control loop computes geometry against a
        frozen world -- an aircraft that "does not move" and a transit that reports the same
        distance for a minute and a half, with no error anywhere.
        """
        msgs = []
        first = True
        while True:
            try:
                if first and timeout and timeout > 0:
                    m = self.conn.recv_match(blocking=True, timeout=timeout)
                else:
                    m = self.conn.recv_match(blocking=False)
            except Exception:
                break
            first = False
            if m is None:
                break
            msgs.append(m)
        for m in msgs:
            self._handle(m)
        return msgs

    def _handle(self, m):
        t = m.get_type()
        if t == "HEARTBEAT":
            self.armed = bool(m.base_mode & SAFETY_ARMED)
            self.mode = (m.custom_mode >> 16) & 0xFF
        elif t == "GLOBAL_POSITION_INT":
            self.lat, self.lon = m.lat / 1e7, m.lon / 1e7
            self.alt = m.alt / 1e3
            # hdg is in centidegrees, 0..35999; PX4 sends 65535 for "unknown"
            if m.hdg != 65535:
                self.heading = m.hdg / 100.0
        elif t == "LOCAL_POSITION_NED":
            self.vx, self.vy, self.vz = m.vx, m.vy, m.vz
        elif t == "VFR_HUD":
            self.airspeed = m.airspeed
            self.groundspeed = m.groundspeed
        elif t == "HOME_POSITION":
            self.home = (m.latitude / 1e7, m.longitude / 1e7, m.altitude / 1e3)
        elif t == "ATTITUDE":
            self.pitch, self.roll, self.yaw = m.pitch, m.roll, m.yaw
        elif t == "ACTUATOR_OUTPUTS":
            self.motors = tuple(m.control[:8])
        elif t == "SYSTEM_STATUS":
            # voltage_battery is in mV and is -1 on some builds; on_q is the airframe
            # quality flag (1000 = fully OK), which is the one to read before blaming a
            # controller for a plant that has decided it is sick.
            self.voltage = m.voltage_battery / 1000.0
            self.q = (m.on_q - 1) / 999.0 if m.on_q not in (0, 1000, 65535) else 0.0
        elif t == "COMMAND_ACK":
            try:
                name = mavutil.mavlink.enums.mav_result_enum(m.result).name
            except Exception:
                name = f"result {m.result}"
            self.last_ack = (m.command, name)
        elif t == "PARAM_VALUE":
            pid = m.param_id.rstrip("\x00")
            self.params[pid] = _param_value_as_number(m.param_value)
            self._param_wanted.pop(pid, None)
        elif t == "ESTIMATOR_STATUS":
            # The gates that matter before an aircraft will arm in this rig: the EKF has
            # to have aligned (yaw, position, velocity) and the barometer has to be
            # trusted. Waiting on a bitmask here beats waiting on a timeout and then
            # discovering why.
            good = (m.flags & 0x0001) and (m.flags & 0x0004) and (m.flags & 0x0010)
            self.est_ok = bool(good)
        elif t == "STATUSTEXT":
            # PX4 repeats a rejected-setpoint complaint at the setpoint rate. Print each
            # distinct message once with a count, or one bug buries every other line of
            # evidence -- which is how twenty minutes went into a type_mask.
            text = m.text.rstrip("\x00")
            # Health-check refusals arrive as STATUSTEXT, and they are the only place the
            # reason is named. Keep the most recent one whether or not we are printing.
            if "arm" in text.lower() or "preflight" in text.lower():
                self.last_arm_failure = text
            if self.verbose and m.severity <= 6:
                if text == self._last_status:
                    self._status_repeats += 1
                else:
                    if self._status_repeats > 1:
                        print(f"    [px4] (x{self._status_repeats + 1}) {self._last_status}",
                              flush=True)
                    self._last_status, self._status_repeats = text, 0
                    print(f"    [px4] {text}", flush=True)

    # -- waiting --------------------------------------------------------------
    def wait_heartbeat(self, timeout_s=30.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            msgs = self.pump(timeout=0.5)
            if any(m.get_type() == "HEARTBEAT" for m in msgs):
                return True
        raise TimeoutError("no PX4 HEARTBEAT")

    def wait_state(self, timeout_s=90.0, want_latlon=True):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            self.pump(timeout=0.5)
            if self.lat is not None and (self.home is not None or not want_latlon):
                return True
        raise TimeoutError("no PX4 position")

    def wait_armed(self, timeout_s=30.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            self.pump(timeout=0.2)
            if self.armed:
                return True
        return False

    # -- sending --------------------------------------------------------------
    def send(self, msg):
        try:
            self.conn.mav.send(msg)
        except Exception as e:
            if self.verbose:
                print(f"    [link] send failed: {e!r}", flush=True)

    def heartbeat(self, offboard: bool):
        base = 0
        if offboard:
            base |= OFFBOARD_FLAG
        self.send(mavutil.mavlink.MAVLink_heartbeat_message(
            type=mavutil.mavlink.MAV_TYPE_GCS,
            autopilot=mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            base_mode=base, custom_mode=0, system_status=0, mavlink_version=3))

    def set_mode(self, main_mode: int, sub_mode: int = 0):
        """MAV_CMD_DO_SET_MODE, in PX4's own reading of it.

        `param1` is a BASE MODE, not the mode: PX4 only looks at `param2` when bit 0 of
        `param1` (CUSTOM_MODE_ENABLED) is set -- `Commander.cpp`, `VEHICLE_CMD_DO_SET_MODE`.
        So the intuitive "put the mode number in param1" silently falls into the generic
        base-mode branch, PX4 decides the vehicle asked for a mission, and answers
        "Switching to Mission is currently not available" while the mode never changes.
        `param2` is the custom MAIN mode and `param3` the sub-mode, as separate bytes.
        """
        self.send(mavutil.mavlink.MAVLink_command_long_message(
            command=mavutil.mavlink.MAV_CMD_DO_SET_MODE, confirmation=0,
            target_system=1, target_component=1,
            param1=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            param2=main_mode, param3=sub_mode, param4=0, param5=0, param6=0, param7=0))

    def arm(self, force=True):
        self.send(mavutil.mavlink.MAVLink_command_long_message(
            command=mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, confirmation=0,
            target_system=1, target_component=1,
            param1=1, param2=21196 if force else 0, param3=0, param4=0, param5=0, param6=0, param7=0))

    def try_arm(self, timeout_s=45.0, keepalive=None):
        """Arm, retrying, and keep the link alive while waiting.

        Two things make a single arm attempt unreliable, and both bit the fixed wing:

        * **PX4 has preconditions that are still settling.** A freshly booted SITL answers
          "Preflight Fail: no heading reference" for several seconds while the EKF aligns
          yaw, and "Resolve system health failures first" is the arm command's response to
          that. Sending the command once, at t=0, and giving up is how you get a rig where
          the quad arms and the fixed wing mysteriously does not -- the quad simply won the
          race more often.
        * **No connection to the GCS is itself a health check.** The heartbeat that
          satisfies it was only being sent after OFFBOARD was entered, so during the arm
          wait there was no GCS on the other end and one more reason to refuse.

        So: heartbeat throughout, and re-arm until it takes or the time runs out.
        """
        t0 = time.monotonic()
        attempts = 0
        while time.monotonic() - t0 < timeout_s:
            if keepalive is not None:
                keepalive()
            self.arm()
            attempts += 1
            t1 = time.monotonic()
            while time.monotonic() - t1 < 1.0:
                if keepalive is not None:
                    keepalive()
                self.pump(timeout=0.05)
                if self.armed:
                    return True
            self.pump(timeout=0.0)
        return self.armed

    def takeoff(self, alt_m: float):
        """MAV_CMD_NAV_TAKEOFF, for an aircraft left in AUTO. The OFFBOARD path does not
        use it: in OFFBOARD, a position setpoint above the aircraft IS the takeoff, and
        mixing the two produces a confusing handover. Kept because the fixed-wing rig's
        catapult launch still wants a mission-level climb-out."""
        self.send(mavutil.mavlink.MAVLink_command_long_message(
            command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, confirmation=0,
            target_system=1, target_component=1,
            param1=0, param2=0, param3=0, param4=float(alt_m), param5=0, param6=0, param7=0))

    def set_param(self, name: str, value: float, ptype=None):
        """PARAM_SET, once, with the whole name in the message's 16-byte `param_id`.

        Sending the name ONE CHARACTER PER MESSAGE does not work, and the fact that it was
        written that way -- with a comment saying PX4's own examples do it -- is the most
        expensive kind of bug: it fails silently, so every parameter write looked like it had
        gone out. Measured on this rig: one PARAM_SET carrying `MPC_Z_VEL_MAX_DN` padded to
        16 bytes reads back 7.5 after being set to 7.5; the same parameter set one character
        at a time does not read back at all. That went unnoticed because nothing read the
        value back -- so `set_pass_speed` has also never actually set FW_AIRSPD_TRIM, and
        the fixed-wing pass speeds were whatever PX4 booted with.

        `ptype` matters for the same reason a second time. An ENUM or INT parameter is
        stored as int32, and a PARAM_SET that declares itself REAL32 is simply refused:
        `NAV_DLL_ACT` stayed at 2 through every attempt until the declared type was
        INT32. `set_params` retries with INT32 when a read-back disagrees, so callers do not
        have to know which is which.
        """
        if ptype is None:
            ptype = (mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        self.send(mavutil.mavlink.MAVLink_param_set_message(
            target_system=1, target_component=1,
            param_id=name[:16].encode("ascii", "ignore").ljust(16, b"\x00"),
            param_value=float(value),
            param_type=ptype))

    def set_params(self, mapping: dict, timeout_s=3.0, keepalive=None, tol=1e-3) -> dict:
        """Set several parameters and READ THEM BACK. Returns what PX4 actually has.

        A parameter write that is not verified is a hope. PX4 accepts a PARAM_SET, answers
        with a STATUSTEXT that looks like success, and can still leave the old value in
        place (out-of-range, wrong declared type, not-yet-loaded-at-boot), and nothing
        downstream distinguishes that from success.

        Anything that did not take as REAL32 is retried as INT32, because that is exactly
        what an enum or integer parameter needs and the failure is otherwise indistinguishable
        from the parameter being read-only.
        """
        for k, v in mapping.items():
            self.set_param(k, v)
            self.params.pop(k[:16], None)
        out = self.get_params(list(mapping.keys()), timeout_s=timeout_s, keepalive=keepalive)
        retry = {k: v for k, v in mapping.items()
                 if out.get(k) is None or abs(out[k] - v) > tol}
        if retry:
            for k, v in retry.items():
                self.set_param(k, v, ptype=mavutil.mavlink.MAV_PARAM_TYPE_INT32)
                self.params.pop(k[:16], None)
            out.update(self.get_params(list(retry.keys()), timeout_s=timeout_s,
                                       keepalive=keepalive))
        # Kept for callers that want it; the diagnostic print above is the usual consumer.
        self._last_param_mismatch = {k: (out[k], mapping[k]) for k, v in out.items()
                                     if out.get(k) is None or abs(v - mapping[k]) > tol}
        return out

    def get_param(self, name: str, timeout_s=2.0):
        """One parameter's live value, or None. PARAM_REQUEST_READ, then wait for PARAM_VALUE.

        Reading a limit off the vehicle is not a nicety. PX4 enforces MPC_Z_VEL_MAX_DN /
        _UP inside `PositionControl::_positionControl()` -- `constrain(_vel_sp(2), -_lim_vel_up,
        _lim_vel_down)` -- which runs on every update and therefore clamps a *velocity*
        setpoint exactly like a position one. The stock maximum descent is 1.5 m/s, so a
        guidance law that commands 4 m/s of descent is silently flown at 1.5 m/s and the
        resulting error is indistinguishable from a control bug. The only way to find that
        out is to read the number, and the only way to fix it is to know it.
        """
        name = name[:16]
        if name in self.params:
            return self.params[name]
        self._param_wanted[name] = time.monotonic() + timeout_s
        self.send(mavutil.mavlink.MAVLink_param_request_read_message(
            target_system=1, target_component=1, param_id=name.encode("ascii", "ignore"),
            param_index=-1))
        t0 = time.monotonic()
        while time.monotonic() < t0 + timeout_s:
            self.pump(timeout=0.1)
            if name in self.params:
                return self.params[name]
        self._param_wanted.pop(name, None)
        return None

    def get_params(self, names, timeout_s=3.0, keepalive=None) -> dict:
        """Several at once, so one pump pass covers all of them.

        `keepalive` is called once per poll iteration and is NOT optional in practice. PX4
        drops out of OFFBOARD if the setpoint stream or the offboard heartbeat stops for
        about a second, and a parameter read that blocks for a few seconds without sending
        either is a failsafe with a long fuse: the aircraft keeps flying beautifully right
        up until the read completes, then declares signal loss, goes RTL, lands and
        disarms, and every measurement after it is of an aircraft lying on the ground. That
        is not hypothetical -- it is what the first run of the vertical probe did.
        """
        want = [n[:16] for n in names]
        self._param_wanted.update({n: time.monotonic() + timeout_s for n in want})
        for n in want:
            self.send(mavutil.mavlink.MAVLink_param_request_read_message(
                target_system=1, target_component=1, param_id=n.encode("ascii", "ignore"),
                param_index=-1))
        t0 = time.monotonic()
        while time.monotonic() < t0 + timeout_s:
            if keepalive is not None:
                keepalive()
            self.pump(timeout=0.1)
            if not any(n in self._param_wanted for n in want):
                break
        for n in want:
            self._param_wanted.pop(n, None)
        return {n: self.params.get(n) for n in want}

    # -- setpoints ------------------------------------------------------------
    def set_velocity_ned(self, vx, vy, vz):
        """The whole command for a multirotor: a velocity setpoint in NED."""
        self.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            time_boot_ms=int(time.monotonic() * 1000) & 0xFFFFFFFF,
            target_system=1, target_component=1,
            coordinate_frame=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask=VELOCITY_ONLY,
            x=0.0, y=0.0, z=0.0, vx=float(vx), vy=float(vy), vz=float(vz),
            afx=0.0, afy=0.0, afz=0.0, yaw=0.0, yaw_rate=0.0))

    def set_velocity_ned_alt(self, vx, vy, vz, alt_m_agl):
        """Velocity plus an altitude to hold -- and it is a trap.

        POSITION_AND_VELOCITY means BOTH. PX4 honours the position fields and the velocity
        fields, so the x/y that are structurally required by the message are not optional
        decoration: passing 0 for them is an instruction to fly to the NED ORIGIN, which
        on this rig is home. A pure velocity command against this mask therefore pins the
        aircraft at the origin while the velocity field says otherwise, the two disagree,
        mc_pos_control wins on position, and the aircraft sits still and looks healthy.

        Measured: a transit 347 m from home reporting "347m" every ten seconds for a
        minute and a half, with correct telemetry and no error anywhere. Use
        `set_velocity_ned` (position ignored) for a velocity command, or put the real
        target in x/y if you mean a position command. PX4's own offboard examples --
        mavsdk's included -- use a pure velocity mask, so this is the default worth having.
        """
        self.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            time_boot_ms=int(time.monotonic() * 1000) & 0xFFFFFFFF,
            target_system=1, target_component=1,
            coordinate_frame=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask=POSITION_AND_VELOCITY,
            x=0.0, y=0.0, z=float(alt_m_agl), vx=float(vx), vy=float(vy), vz=float(vz),
            afx=0.0, afy=0.0, afz=0.0, yaw=0.0, yaw_rate=0.0))

    def set_position_ned(self, x, y, z, vx=0.0, vy=0.0, vz=0.0):
        """A moving waypoint with velocity feedforward. The fixed wing's alternative to a
        velocity setpoint, and the one that gets PX4's own waypoint navigation and L1."""
        self.send(mavutil.mavlink.MAVLink_set_position_target_local_ned_message(
            time_boot_ms=int(time.monotonic() * 1000) & 0xFFFFFFFF,
            target_system=1, target_component=1,
            coordinate_frame=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask=POSITION_AND_VELOCITY,
            x=float(x), y=float(y), z=float(z), vx=float(vx), vy=float(vy), vz=float(vz),
            afx=0.0, afy=0.0, afz=0.0, yaw=0.0, yaw_rate=0.0))


# --------------------------------------------------------------------- backends

class OffboardController:
    """Drives the law against PX4. One subclass per vehicle type, because what PX4 will
    accept differs by more than a number."""

    def __init__(self, link: Link, spec, guidance, airframe="quad", verbose=True,
                 airframe_capability=None):
        self.link = link
        self.spec = spec
        self.guidance = guidance
        self.airframe = airframe
        self.verbose = verbose
        # flyby.airframes.Airframe, when we know which aircraft we are flying. Optional so
        # that the probe and the diagnostics can drive the link without claiming to be an
        # airframe.
        self.capability = airframe_capability
        # Optional `flyby.stream.StreamWriter`. Attached by fly_live so that climb-out and
        # transit are watchable too: they are a third of the wall clock and the part where a
        # rig fault shows up first, and a viewer that only starts once the pass does spends
        # its first two minutes saying "waiting for telemetry".
        self.stream = None
        self._stream_t0 = None
        self.origin = geo.Origin()
        self._last_setpoint = 0.0
        self._last_hb = 0.0
        self._last_status = ""
        self._status_repeats = 0

    def _stream(self, t=None, plan=None):
        if self.stream is None:
            return
        n = self.own_ned()
        if n is not None:
            # `mark()` is the stream's own clock, so climb-out, transit and the pass all
            # land on ONE timeline. Each keeping its own was the bug.
            self.stream.write(self.stream.mark() if t is None else t,
                              n, self.own_vel(), None, plan)

    # -- state ----------------------------------------------------------------
    def own_ned(self):
        l = self.link
        if l.lat is None:
            return None
        lat, lon, alt = l.lat, l.lon, l.alt
        n, e = geo.latlon_to_ned(lat, lon, self.origin)
        return (n, e, self.origin.alt_m - alt)

    def own_vel(self):
        l = self.link
        return (l.vx, l.vy, l.vz)

    # -- mode -----------------------------------------------------------------
    def prepare(self, cruise_alt_m=100.0, timeout_s=90.0):  # noqa: D401
        """Everything that has to be true before the law can be closed: PX4 up, position
        estimated, armed, airborne, and in OFFBOARD with setpoints already streaming.

        PX4 will not enter OFFBOARD until setpoints have been arriving for about a second
        with the offboard flag set in the heartbeat -- so the heartbeat and a placeholder
        setpoint go out BEFORE the mode switch, not after. Skipping that ordering is the
        usual reason a first OFFBOARD script "does nothing".
        """
        l = self.link
        print("  waiting for heartbeat ...", end="", flush=True)
        l.wait_heartbeat()
        print(" ok")
        print("  waiting for position estimate ...", end="", flush=True)
        l.wait_state(timeout_s=timeout_s)
        print(f" ok  ({l.lat:.6f}, {l.lon:.6f} @ {l.alt:.1f} m)")

        # The rig's own parameters first, because one of them (`NAV_DLL_ACT`) gates ARMING.
        # Applied after the arm attempt it would be a note in a log explaining why the
        # aircraft never flew, which is precisely the failure it prevents.
        print("  rig parameters ...", flush=True)
        for k, want in sorted(RIG_PARAMS.items()):
            got = self.link.set_params({k: want}, timeout_s=3.0).get(k)
            ok = got is not None and abs(got - want) < 1e-3
            print(f"    {k:20s} {got}{'' if ok else f'   !! wanted {want}'}", flush=True)

        if not l.armed:
            print("  arming ...", end="", flush=True)
            # The heartbeat doubles as the GCS keepalive that one of PX4's arm preconditions
            # wants, so it runs for the whole wait, not just once OFFBOARD is entered.
            ok = l.try_arm(timeout_s=45.0,
                           keepalive=lambda: (self.heartbeat(False), l.pump(timeout=0.0)))
            print(" ok" if ok else f" TIMED OUT ({l.last_arm_failure or 'no reason given'})")

        # Tell the autopilot what this airframe can do BEFORE it is asked to do it. The
        # stock vertical limits are lower than the aircraft's real capability and PX4
        # enforces them on the velocity setpoint, silently -- so this is a precondition,
        # not a tuning step. See `apply_airframe_limits`.
        if self.capability is not None:
            print("  airframe vertical limits ...")
            self.apply_airframe_limits(self.capability)

        print(f"  entering OFFBOARD (climb-out handled separately) ...", end="", flush=True)
        self.enter_offboard()
        print(" ok")
        return True

    def enter_offboard(self, settle_s=1.5):
        l = self.link
        t0 = time.monotonic()
        while time.monotonic() - t0 < settle_s:
            self.heartbeat(offboard=True)
            self.send_placeholder()
            l.pump(timeout=0.05)
            time.sleep(1.0 / SETPOINT_HZ)
        l.set_mode(CUSTOM_MODE_OFFBOARD)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.0:
            self.heartbeat(offboard=True)
            self.send_placeholder()
            l.pump(timeout=0.05)
            time.sleep(1.0 / SETPOINT_HZ)

    def heartbeat(self, offboard=True):
        now = time.monotonic()
        if now - self._last_hb < 1.0 / HEARTBEAT_HZ:
            return
        self._last_hb = now
        self.link.heartbeat(offboard)

    def send_placeholder(self):
        """A setpoint that holds the aircraft where it is. Streamed before the law has a
        valid state, so PX4 sees a live setpoint stream from the moment OFFBOARD starts.

        Velocity-hold for a fixed wing. It cannot take the position setpoint the multirotor
        uses -- asking a wing to hold a position asks it to stop, and a stopped wing falls --
        so before it has a state to hold, a fixed wing is told to keep flying straight and
        level at zero rate. The launch has not happened yet at this point, so the command is
        simply ignored until the rail fires.
        """
        n = self.own_ned()
        if n is None or self.airframe != "quad":
            self.link.set_velocity_ned(0.0, 0.0, 0.0)
            return
        self.link.set_position_ned(n[0], n[1], n[2])

    # -- the loop --------------------------------------------------------------
    def step(self, plan: PassPlan):
        """Send one setpoint from one plan. Called at SETPOINT_HZ."""
        self.heartbeat(True)
        n = self.own_ned()
        if n is None:
            return
        vx, vy, vz = plan.vel_ned
        # A pure velocity setpoint on all three axes: the law owns the vertical channel in
        # both modes (tangent arrival or demanded flight-path angle), and a position field
        # alongside it would be a second, conflicting instruction -- see
        # `set_velocity_ned_alt`. mc_pos_control tracks a velocity setpoint on altitude
        # directly, so there is nothing to gain by wrapping it in a position command.
        self.link.set_velocity_ned(vx, vy, vz)
        self._last_setpoint = time.monotonic()

    # -- climb-out -------------------------------------------------------------
    def climb_out(self, alt_down_m: float, timeout_s=60.0, tol_m=3.0):
        """Get airborne and hold a commanded height, still in OFFBOARD.

        Two entirely different manoeuvres, because the vehicles are not the same kind of
        thing, and using the multirotor one on a fixed wing does not degrade gracefully --
        it does nothing at all, silently.
        """
        if self.airframe != "quad":
            return self._climb_out_fixed_wing(alt_down_m, timeout_s=timeout_s, tol_m=tol_m)

        n = self.own_ned()
        if n is None:
            return False
        x, y = n[0], n[1]
        t0 = time.monotonic()
        print(f"  climb-out to {alt_down_m:.0f} m below home ...", end="", flush=True)
        while time.monotonic() - t0 < timeout_s:
            self.heartbeat(True)
            self.link.set_position_ned(x, y, alt_down_m)   # position command: correct here
            self.link.pump(timeout=0.02)
            self._stream()
            n = self.own_ned()
            if n is not None and abs(n[2] - alt_down_m) < tol_m and abs(n[0] - x) < 5 and abs(n[1] - y) < 5:
                print(f" ok ({time.monotonic()-t0:.1f}s)")
                return True
            time.sleep(1.0 / SETPOINT_HZ)
        print(" TIMED OUT")
        return False

    def _climb_out_fixed_wing(self, alt_down_m: float, timeout_s=120.0, tol_m=8.0):
        """Wait for the catapult, then climb away along the rail heading at flying speed.

        A fixed wing cannot do what `climb_out` does for the multirotor, and the reason is
        worth stating because it is a category error rather than a tuning one: holding a
        *position* setpoint asks the aircraft to stop, and a fixed wing that stops falls out
        of the sky. So there is no position setpoint anywhere in this function -- the whole
        climb is flown on velocity setpoints, heading held where the rail pointed.

        The launch itself is not ours to perform. simuav's FDM fires the catapult
        `CATAPULT_ARM_DELAY_S` after it sees PX4 arm, because PX4's launch detector has to
        be running before the stroke happens or it misses the spike; so all this has to do
        is keep the setpoint stream and the heartbeat alive while the rail does the work,
        and then fly.

        The speed floor is the airframe's, not zero: commanded below the minimum the wing
        stops flying, and a Geran-2 arriving on the rail at 18 m/s has to be told to keep
        30 m/s up or it stalls before the climb starts.
        """
        cap = self.capability
        if cap is None:
            raise RuntimeError(
                "no airframe capability on this link: the launch climb-out speed is the "
                "airframe's own and there is no defensible fallback for it (that was the bug "
                "this replaced -- a literal 12 m/s and 3 m/s here, which is right for the iris "
                "and for nothing else). Pass --airframe, or build the link with a capability.")
        v_climb = cap.max_climb_mps
        v_fly = max(cap.min_speed_mps, cap.max_climb_mps * 2.0)
        t0 = time.monotonic()
        print("  waiting for launch ...", end="", flush=True)
        heading = None
        while time.monotonic() - t0 < timeout_s:
            self.heartbeat(True)
            self.link.set_velocity_ned(0.0, 0.0, 0.0)      # hold still until the rail fires
            self.link.pump(timeout=0.02)
            n = self.own_ned()
            if n is not None and math.hypot(self.link.vx, self.link.vy) > 0.5:
                heading = math.atan2(self.link.vy, self.link.vx)
                print(f" airborne at {math.hypot(self.link.vx, self.link.vy):.1f} m/s, "
                      f"heading {math.degrees(heading) % 360:.0f} deg", end="  ", flush=True)
                break
            time.sleep(1.0 / SETPOINT_HZ)
        else:
            print(" TIMED OUT waiting for launch")
            return False

        while time.monotonic() - t0 < timeout_s:
            self.heartbeat(True)
            # Climb at the best rate the airframe has, along whatever heading it is already
            # flying, and never below its minimum speed: a wing that is told to slow down
            # during a climb-out stalls, and a stalled aircraft is not recoverable by a
            # velocity setpoint.
            vz = v_climb
            vx, vy = v_fly * math.cos(heading), v_fly * math.sin(heading)
            self.link.set_velocity_ned(vx, vy, vz)
            self.link.pump(timeout=0.0)
            n = self.own_ned()
            if n is not None and abs(n[2] - alt_down_m) < tol_m:
                print(f" ok ({time.monotonic()-t0:.0f}s, {-(n[2]):.0f} m)")
                return True
            time.sleep(1.0 / SETPOINT_HZ)
        print(" TIMED OUT")
        return False

    def transit_to(self, ned, speed_mps=8.0, timeout_s=180.0, tol_m=6.0,
                   report_every_s=10.0):
        """Fly to a starting position before the pass is armed, still in OFFBOARD.

        A flyby law is a function of the approach geometry, and the geometry is set by
        where the aircraft starts -- 900 m out and 800 m off the line is a different
        problem from being parked on the line. So the rig has to be able to PUT the
        aircraft there, or every live measurement is of whatever position the climb-out
        happened to leave it in.

        Deliberately a dumb velocity command proportional to the remaining distance. It is
        not the law and must not be confused with it: it exists to set up the law's initial
        condition, and a clever transit would only make the experiment harder to read.
        """
        # NED z is DOWN-positive: a negative z is above home.
        print(f"  transit to N{ned[0]:+.0f} E{ned[1]:+.0f} "
              f"{abs(ned[2]):.0f} m {'above' if ned[2] < 0 else 'below'} home ...",
              end="", flush=True)
        t0 = time.monotonic()
        last = t0
        while time.monotonic() - t0 < timeout_s:
            self.heartbeat(True)
            n = self.own_ned()
            if n is None:
                time.sleep(1.0 / SETPOINT_HZ)
                continue
            d = math.hypot(ned[0] - n[0], ned[1] - n[1])
            dz = ned[2] - n[2]
            v = min(speed_mps, 0.35 * d)
            vx = v * (ned[0] - n[0]) / d if d > 0.2 else 0.0
            vy = v * (ned[1] - n[1]) / d if d > 0.2 else 0.0
            # The vertical budget is the AIRFRAME's, and it differs by kind: a fixed wing
            # at 51 m/s cruise with a 5.8 m/s ceiling is climbing shallowly or not at all,
            # and a fixed 3 m/s is both below what it can do and above what it can sustain
            # while turning. Asymmetric, because climb and descent ceilings are different
            # numbers on every vehicle here. Both come from flyby/airframes.py, which reads
            # them out of the aircraft's profile; there is no fallback number to be wrong by.
            cap = self.capability
            if cap is None:
                raise RuntimeError("no airframe capability on this link: the vertical budget "
                                   "of a transit is the airframe's own (flyby/airframes.py)")
            z_lim, z_lim_dn = cap.max_climb_mps, cap.max_descent_mps
            vz = max(-z_lim, min(z_lim, 0.5 * dz)) if dz >= 0 \
                else max(-z_lim_dn, min(0.0, 0.5 * dz))
            self.link.set_velocity_ned(vx, vy, vz)
            self.link.pump(timeout=0.0)
            self._stream()
            if d < tol_m and abs(dz) < tol_m:
                print(f" ok ({time.monotonic() - t0:.0f}s)")
                return True
            now = time.monotonic()
            if now - last > report_every_s:
                last = now
                print(f" {d:.0f}m/{dz:+.0f}m", end="", flush=True)
            time.sleep(1.0 / SETPOINT_HZ)
        print(" TIMED OUT")
        return False

    def settle_velocity(self, vel_ned, seconds=6.0):
        """Bring the aircraft up to a scenario's intended starting velocity.

        The transit deliberately ends with the aircraft nearly stopped (it stops when it
        arrives), and a flyby law's initial condition is a specific one: on the line, at the
        pass speed, on the pass heading. Starting a pass from a standstill is a different --
        easier in one way, harder in another -- experiment than the one the offline numbers
        describe, and quietly comparing the two would mean measuring something else.

        For a fixed wing the horizontal is held at the requested speed but the VERTICAL is
        flown as a trim rather than commanded to zero: a wing that is told to hold a zero
        vertical rate while it is also being asked to decelerate will fly an oscillating
        phugoid, and the scenario's starting state is a glide, not a level run. That is why
        `settle_velocity` gets a `vz` and does not assume zero.
        """
        vz = vel_ned[2] if len(vel_ned) > 2 else 0.0
        print(f"  settle onto start velocity {vel_ned[0]:.1f}/{vel_ned[1]:.1f} m/s ...", end="",
              flush=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            self.heartbeat(True)
            self.link.set_velocity_ned(vel_ned[0], vel_ned[1], vz)
            self.link.pump(timeout=0.0)
            time.sleep(1.0 / SETPOINT_HZ)
        print(" ok")

    def apply_airframe_limits(self, airframe, keepalive=None) -> dict:
        """Push the airframe's vertical limits to the autopilot, and READ THEM BACK.

        This is the fix for the live vertical blocker, and it is here rather than in the
        guidance because the limit is enforced by PX4, not by us. Measured on this rig with
        the stock iris defaults (MPC_Z_VEL_MAX_DN 1.5, _UP 3.0) and a pure vertical-velocity
        setpoint: commanded +2, +4 and +6 m/s were all flown at +1.50, commanded -4 was
        flown at -3.00, and it made no difference whether setpoints went out at 20 Hz or
        100 Hz. Raising the two parameters to the airframe's actual capability and repeating
        the identical ladder gave +2.01, +4.02, +6.02 and -3.99.

        A fixed wing gets FW_T_CLMB_MAX / FW_T_SINK_MAX instead, because MPC_Z_VEL_* is a
        multirotor parameter and pushing it to a plane writes a parameter nothing reads --
        which is exactly the kind of silent no-op that leaves you debugging the wrong layer.

        Returns what PX4 reports afterwards, not what was asked for. A write that is not
        read back is a hope, and the whole cost of this bug was trusting one.
        """
        if self.airframe == "quad":
            want = {"MPC_Z_VEL_MAX_DN": float(airframe.max_descent_mps),
                    "MPC_Z_VEL_MAX_UP": float(airframe.max_climb_mps),
                    "MPC_Z_V_AUTO_DN": float(airframe.max_descent_mps),
                    "MPC_Z_V_AUTO_UP": float(airframe.max_climb_mps),
                    "MPC_XY_VEL_MAX": float(airframe.max_speed_mps)}
            names = ("MPC_XY_VEL_MAX", "MPC_Z_VEL_MAX_DN", "MPC_Z_VEL_MAX_UP")
        else:
            # TECS enforces these for the fixed wing; the descent one wants headroom for a
            # dive, and both are capped by the airframe's pitch limit rather than its climb
            # rate, so they are set generously and the guidance's own clamp is what holds.
            want = {"FW_T_CLMB_MAX": float(airframe.max_climb_mps),
                    "FW_T_SINK_MAX": float(airframe.max_descent_mps)}
            names = ("FW_T_CLMB_MAX", "FW_T_SINK_MAX")

        before = self.link.get_params(names, timeout_s=3.0, keepalive=keepalive)
        after = self.link.set_params(want, timeout_s=3.0, keepalive=keepalive)
        self._last_limits = after
        for n in names:
            b, a = before.get(n), after.get(n)
            ok = a is not None and abs(a - want[n]) < 1e-3
            print(f"    {n:18s} {b} -> {a}"
                  + ("" if ok else f"   !! wanted {want[n]}"), flush=True)
        bad = [n for n in names if after.get(n) is None or abs(after[n] - want[n]) > 1e-3]
        if bad:
            print(f"    !! {', '.join(bad)} did not take. The autopilot will still clip "
                  f"the vertical setpoint and any pass needing a descent faster than the "
                  f"stock limit will be flown slower than commanded.", flush=True)
        return after

    def verify_vertical_limits(self, airframe) -> bool:
        """Read the vertical limits back and say whether the vehicle can do what we asked.

        Cheap, and worth doing as a pre-flight assertion rather than discovering the answer
        125 m above the glide slope: the failure is completely silent in flight.
        """
        names = (("MPC_XY_VEL_MAX", "MPC_Z_VEL_MAX_DN", "MPC_Z_VEL_MAX_UP") if self.airframe == "quad"
                 else ("FW_T_CLMB_MAX", "FW_T_SINK_MAX"))
        got = self.link.get_params(names, timeout_s=3.0)
        ok = True
        for n, want in zip(names, (airframe.max_speed_mps, airframe.max_descent_mps,
                                  airframe.max_climb_mps)):
            v = got.get(n)
            good = v is not None and v >= want - 1e-3
            ok &= good
            print(f"    {n:18s} vehicle {v}  needs {want}  {'ok' if good else 'TOO LOW'}",
                  flush=True)
        return ok

    def set_pass_speed(self, speed_mps):
        """Fixed wing only: make the pass speed a PX4 parameter (see the module docstring).

        For the quad it is pushed too and READ BACK, because that read-back is the assertion
        that the quad's speed comes from its velocity setpoint rather than from any parameter
        -- and it is also what proves the parameter write worked at all, which until this was
        verified it did not: `set_param` sent the name one character per message, PX4 never
        applied it, and nothing ever checked, so every fixed-wing pass speed on this rig was
        whatever PX4 booted with.
        """
        got = self.link.set_params({"FW_AIRSPD_TRIM": float(speed_mps)}, timeout_s=3.0)
        back = got.get("FW_AIRSPD_TRIM")
        ok = back is not None and abs(back - speed_mps) < 0.05
        if self.airframe != "quad":
            print(f"  pass speed -> FW_AIRSPD_TRIM = {back} (fixed wing: this is the only "
                  f"way OFFBOARD can hold an airspeed)"
                  + ("" if ok else f"   !! wanted {speed_mps}"))
        elif not ok:
            print(f"  pass speed: FW_AIRSPD_TRIM reads {back}, wanted {speed_mps}. Harmless "
                  f"for the quad, whose speed is a velocity setpoint, but it means a "
                  f"parameter write is not reaching PX4.")
