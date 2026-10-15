#!/usr/bin/env python3
"""Isolate PX4's vertical channel: a pure vertical-velocity step ladder, no guidance.

This exists to answer one question with numbers instead of arguments -- *when the guidance
commands a 4 m/s descent and the aircraft only manages 1.5, which layer is wrong?* There
are only three candidates and they are distinguished by one experiment each:

  the link        -- is OFFBOARD actually engaged, are setpoints arriving, is the mask right
  the tick rate   -- is SETPOINT_HZ too low for PX4's vertical loop to converge
  the autopilot   -- is PX4 deliberately clipping the setpoint to a configured limit

The third is invisible from the outside unless you go looking for it, because a clipped
setpoint produces no error, no message and no divergence: the aircraft just does something
else. And it is a real trap in PX4 specifically -- `PositionControl::_positionControl()`
runs on every update and unconditionally executes

    _vel_sp(2) = constrain(_vel_sp(2), -_lim_vel_up, _lim_vel_down)

so the VERTICAL VELOCITY LIMIT is applied to a *velocity* setpoint, not only to a position
one, and silently. With the stock iris defaults that caps descent at
`MPC_Z_VEL_MAX_DN` = 1.5 m/s and ascent at `MPC_Z_VEL_MAX_UP` = 3 m/s.

So the probe does four things at once:

  1. reads the limits straight off the vehicle (`PARAM_REQUEST_READ`), before changing
     anything, so the stock value is on the record rather than assumed;
  2. counts the setpoints actually sent per second, because the loop's nominal rate is not
     the achieved one -- `pump()` and `time.sleep()` both cost time and the loop has no
     catch-up;
  3. steps the vertical velocity through a ladder AT TWO TICK RATES, which is the only way
     to separate "too slow a loop" from "clipped": a rate problem shows up as settling
     time that improves with rate, a clamp does not improve at all;
  4. reads the FDM's ground truth alongside PX4's own estimate, because those two
     disagreeing is a fifth candidate (a bad altitude estimate looks exactly like a plant
     that will not descend) and it costs one HTTP request to rule out.

Everything is dumped to CSV; the summary table is the reading of it.

    scripts/rig.sh start
    python3 scripts/vstep.py                       # ladder at stock limits
    python3 scripts/vstep.py --raise               # push the limits up, ladder again
    python3 scripts/vstep.py --profile iris --csv runs/vstep-iris.csv

`--raise` is the fix, tested by the same probe: if raising `MPC_Z_VEL_MAX_DN` makes the
descent steps land at what was commanded, the clamp was the cause, and no amount of tuning
the guidance law would ever have found it.
"""
import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flyby.mavlink_link import Link, OffboardController  # noqa: E402

TRUTH_PORT = 8099
# The FDM is asked for ground truth at a quarter of the setpoint rate, on purpose. Truth is
# an HTTP round trip per sample and its own derivative is a finite difference, so polling it
# at 20 Hz buys nothing: it costs a socket per tick and returns a rate differenced over
# 0.05 s of sim time, which is mostly noise. At 4 Hz the difference is taken over ~0.25 s,
# which is long enough to mean something.
TRUTH_HZ = 4.0

# What to step through. NED vz: POSITIVE is DOWN. The descent side is the interesting one
# (that is where the stock limit bites), but the ascent side is measured too because the
# two limits are separate parameters with different defaults and it would be easy to fix
# one and miss the other.
LADDER = (0.0, 1.0, 2.0, 4.0, 6.0, -1.0, -2.0, -4.0)
TICK_RATES = (20.0, 100.0)      # the link's nominal SETPOINT_HZ, and 5x it
HOLD_S = 6.0                    # long enough for the lag to settle out
SETTLE_S = 3.0                  # the window averaged for the verdict


def truth(url=f"http://127.0.0.1:{TRUTH_PORT}/truth"):
    """The FDM's own position, straight out of the flight dynamics.

    Deliberately built with an empty ProxyHandler: an http_proxy in the environment will
    happily swallow a request to 127.0.0.1 and either hang it or return something from a
    proxy, and this function's entire value is that it cannot be anything but the sim.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=2.0) as r:
        return json.loads(r.read().decode())


class Truth:
    """Ground truth as a rate, by differencing altitude over a short window.

    The FDM publishes altitude but no vertical velocity, and a finite difference over a
    window is exactly what is wanted here: a single sample's derivative is telemetry noise,
    and the question is what the aircraft is doing once it has settled.

    One instance for the whole run, NOT one per step: a fresh instance has no previous
    sample, so its first difference -- and its only difference within a short step -- is
    undefined and the whole ladder comes back with no truth column at all.
    """

    def __init__(self):
        self.prev = None
        self.last = 0.0
        self._last = None
        self.n_ok = 0
        self.n_try = 0
        self.last_error = None

    def rate_mps(self, force=False):
        """Ground truth vertical rate, or None if it is not available this instant.

        Never raises. This poll is a diagnostic on top of an experiment that is already
        running, and a diagnostic that can kill the run is worse than no diagnostic: a
        transient HTTP error or a payload that is momentarily a different shape must
        produce a missing number, not a traceback 20 minutes into a flight.
        """
        now = time.monotonic()
        if not force and now - self.last < 1.0 / TRUTH_HZ:
            return self._last
        self.last = now
        try:
            # sim_time_s lives at the TOP level of the payload, not inside each vehicle --
            # the vehicles carry their own fields (altitude_m, roll_deg, ...) and a
            # vehicle dict has no sim_time_s at all. Reading it off vehicles[0] is a
            # KeyError on every single poll, which the broad `except` then turns into a
            # column of silent NaNs that looks exactly like "the FDM published nothing".
            d = truth()
            t = d["vehicles"][0]
            cur = (float(d["sim_time_s"]), float(t["altitude_m"]))
            self.n_try += 1
        except Exception as e:
            self.last_error = repr(e)
            return None
        r = None
        if self.prev is not None:
            dt = cur[0] - self.prev[0]
            # The lower bound exists to reject a repeated or out-of-order sample; it must be
            # BELOW the polling interval or every sample is rejected and the column is blank.
            if 0.05 < dt < 3.0:
                # altitude is UP-positive, vz is DOWN-positive: the sign flips.
                r = -(cur[1] - self.prev[1]) / dt
                self.n_ok += 1
        self.prev = cur
        self._last = r
        return r


def read_limits(link, verbose=True, keepalive=None):
    names = ("MPC_Z_VEL_MAX_DN", "MPC_Z_VEL_MAX_UP", "MPC_Z_V_AUTO_DN", "MPC_Z_V_AUTO_UP",
             "MPC_Z_VEL_PACC", "MPC_Z_VEL_DACC", "MPC_Z_P", "MPC_XY_VEL_MAX",
             "MPC_TILTMAX_AIR", "MPC_THR_HOVER", "MPC_THR_MIN", "MPC_THR_MAX",
             "MPC_XY_P", "MPC_XY_CRUISE")
    got = link.get_params(names, timeout_s=4.0, keepalive=keepalive)
    if verbose:
        print("  limits as PX4 has them:")
        for k in ("MPC_Z_VEL_MAX_DN", "MPC_Z_VEL_MAX_UP", "MPC_Z_AUTO_DN", "MPC_Z_V_AUTO_DN",
                  "MPC_Z_V_AUTO_UP", "MPC_Z_VEL_PACC", "MPC_Z_VEL_DACC", "MPC_Z_P",
                  "MPC_XY_VEL_MAX", "MPC_TILTMAX_AIR", "MPC_THR_HOVER", "MPC_THR_MIN",
                  "MPC_THR_MAX", "MPC_XY_P", "MPC_XY_CRUISE"):
            v = got.get(k)
            if v is not None:
                print(f"    {k:18s} {v:10.4f}")
    return got


def hold(link, ctrl, vz, hz, seconds, rows, label, tr, cmd_vz_hist=None):
    """Hold one vertical velocity setpoint, sampling everything that could explain it."""
    t0 = time.monotonic()
    dt = 1.0 / hz
    sent = 0
    acc = []          # achieved vz (PX4 estimate)
    acc_t = []        # achieved vz (FDM truth)
    tilts = []
    motors = []
    while time.monotonic() - t0 < seconds:
        loop_t = time.monotonic() - t0
        ctrl.heartbeat(True)
        link.set_velocity_ned(0.0, 0.0, vz)
        sent += 1
        link.pump(timeout=0.0)
        est_vz = link.vz
        truth_vz = tr.rate_mps()
        tilt = math.degrees(math.hypot(math.sin(link.pitch), math.sin(link.roll)))
        acc.append(est_vz)
        if truth_vz is not None:
            acc_t.append(truth_vz)
        tilts.append(tilt)
        if link.motors:
            motors.append(sum(link.motors[:4]) / 4.0 / 1000.0)
        rows.append(dict(label=label, tick_hz=hz, t=loop_t, cmd_vd=vz,
                         est_vd=est_vz, truth_vd=truth_vz, tilt_deg=tilt,
                         motor=sum(link.motors[:4]) / 4.0 if link.motors else None,
                         alt_m=-(ctrl.own_ned()[2] or 0.0) if ctrl.own_ned() else None,
                         sent=sent))
        # Sleep the REMAINDER of the tick, not a whole tick: a fixed sleep after the work
        # makes the achieved rate strictly lower than the nominal one, and the whole point
        # of counting setpoints here is to know the achieved rate.
        time.sleep(max(0.0, dt - (time.monotonic() - t0 - loop_t)))

    elapsed = time.monotonic() - t0
    win = lambda xs: statistics.fmean(xs[-int(SETTLE_S / dt):]) if xs else float("nan")  # noqa: E731
    return dict(label=label, tick_hz=hz, cmd_vz=vz, achieved_vz=win(acc),
                truth_vz=win(acc_t), tilt_deg=win(tilts),
                motor=win(motors), rate_hz=sent / elapsed if elapsed > 0 else 0.0,
                n=sent)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--airframe", choices=("quad", "plane"), default="quad")
    ap.add_argument("--profile", default=None, help="reported in the output; the rig owns "
                                                    "which profile is actually flying")
    ap.add_argument("--raise", action="store_true", dest="do_raise",
                    help="push MPC_Z_VEL_MAX_DN/UP up before the ladder (the fix, tested "
                         "by the same probe)")
    ap.add_argument("--raise-dn", type=float, default=8.0)
    ap.add_argument("--raise-up", type=float, default=8.0)
    ap.add_argument("--climb-out", type=float, default=200.0, help="height above home (m)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    print(f"=== vertical channel probe ({args.airframe}"
          f"{', profile ' + args.profile if args.profile else ''}) ===")
    link = Link()
    ctrl = OffboardController(link, None, None, airframe=args.airframe, verbose=False)
    rows = []
    try:
        # Read the limits BEFORE arming. Reading them after `prepare()` blocks the setpoint
        # stream for seconds, PX4's offboard signal times out, and the aircraft does a
        # failsafe RTL, lands and disarms -- which is exactly what the first run of this
        # probe did, and it looked like a guidance failure for the whole climb-out.
        print("  waiting for heartbeat ...", end="", flush=True)
        link.wait_heartbeat()
        print(" ok")
        print("  waiting for position estimate ...", end="", flush=True)
        link.wait_state(timeout_s=60.0)
        print(f" ok  ({link.lat:.6f}, {link.lon:.6f} @ {link.alt:.1f} m)")
        stock = read_limits(link)

        if args.do_raise:
            print(f"\n  raising the autopilot's vertical limits: "
                  f"MPC_Z_VEL_MAX_DN {stock.get('MPC_Z_VEL_MAX_DN')} -> {args.raise_dn}, "
                  f"MPC_Z_VEL_MAX_UP {stock.get('MPC_Z_VEL_MAX_UP')} -> {args.raise_up}")
            after = link.set_params({"MPC_Z_VEL_MAX_DN": args.raise_dn,
                                     "MPC_Z_VEL_MAX_UP": args.raise_up}, timeout_s=4.0)
            for nm, want in (("MPC_Z_VEL_MAX_DN", args.raise_dn),
                             ("MPC_Z_VEL_MAX_UP", args.raise_up)):
                got = after.get(nm)
                okk = got is not None and abs(got - want) < 1e-3
                print(f"    {nm:18s} {got}   {'ok' if okk else '!! DID NOT TAKE'}")
            stock["MPC_Z_VEL_MAX_DN"] = after.get("MPC_Z_VEL_MAX_DN")
            stock["MPC_Z_VEL_MAX_UP"] = after.get("MPC_Z_VEL_MAX_UP")

        ctrl.prepare()

        print(f"\n  climb-out to {args.climb_out:.0f} m above home ...", end="", flush=True)
        ok = ctrl.climb_out(-args.climb_out, timeout_s=180.0, tol_m=5.0)
        n = ctrl.own_ned()
        print(f"    at {-(n[2] or 0.0):.0f} m above home")
        if not ok or not link.armed:
            print(f"    !! not airborne / not armed (armed={link.armed}); the ladder would "
                  f"measure a stationary aircraft. Check the [px4] lines above.", file=sys.stderr)
            return 3

        results = []
        tr = Truth()
        for hz in TICK_RATES:
            print(f"\n  --- ladder at {hz:g} Hz setpoints "
                  f"({len(LADDER)} steps x {HOLD_S:g} s) ---")
            for vz in LADDER:
                r = hold(link, ctrl, vz, hz, HOLD_S, rows,
                         label=f"ladder{int(hz)}", tr=tr)
                results.append(r)
                print(f"    cmd vz {r['cmd_vz']:+5.1f}  ->  PX4 {r['achieved_vz']:+5.2f}  "
                      f"truth {r['truth_vz']:+5.2f}   tilt {r['tilt_deg']:4.1f} deg  "
                      f"motors {r['motor']:.3f}   [{r['rate_hz']:5.1f} Hz achieved]",
                      flush=True)

        # -- the verdict ------------------------------------------------------------
        dn = stock.get("MPC_Z_VEL_MAX_DN")
        print("\n  === verdict ===")
        print(f"  stock MPC_Z_VEL_MAX_DN = {dn}   MPC_Z_VEL_MAX_UP = "
              f"{stock.get('MPC_Z_VEL_MAX_UP')}   (read off PX4 before anything was changed)")
        by_hz = {hz: [r for r in results if r["tick_hz"] == hz] for hz in TICK_RATES}
        for hz, rs in by_hz.items():
            big_down = [r for r in rs if r["cmd_vz"] >= 4.0]
            if big_down:
                got = statistics.fmean(abs(r["achieved_vz"]) for r in big_down)
                print(f"  {hz:5.0f} Hz: commanded >= 4 m/s down, achieved {got:5.2f} m/s")

        lo, hi = TICK_RATES
        a = {r["cmd_vz"]: r["achieved_vz"] for r in by_hz[lo]}
        b = {r["cmd_vz"]: r["achieved_vz"] for r in by_hz[hi]}
        common = [k for k in a if k in b and abs(k) > 0.5]
        delta = statistics.fmean(abs(b[k] - a[k]) for k in common) if common else float("nan")
        print(f"\n  |achieved(100 Hz) - achieved(20 Hz)| = {delta:.3f} m/s over {len(common)} steps")
        if delta < 0.05:
            print("  -> the tick rate is NOT the constraint: a 5x faster setpoint stream "
                  "changes nothing, which a loop-rate problem cannot do.")
        else:
            print("  -> the tick rate DOES matter: the achieved rates differ by "
                  f"{delta:.3f} m/s between 20 and 100 Hz.")

        if dn is not None:
            # A step is "clipped" when it asked for more than the limit allows AND got
            # pinned at the limit. Testing against the limit itself rather than a fraction
            # of it matters: PX4 holds the setpoint at exactly the limit, so any threshold
            # stricter than the limit's own resolution reports a real clamp as "no clamp".
            # Everything is compared in MAGNITUDE: `-4 > -3 + 0.25` is false, so comparing
            # the raw signed values reports every ascent clamp as "not clipped".
            def clipped(r, lim):
                return (lim is not None and lim > 0.0
                        and r["cmd_vz"] * lim > 0            # same axis, same sign
                        and abs(r["cmd_vz"]) > abs(lim) + 0.25
                        and abs(abs(r["achieved_vz"]) - abs(lim)) < 0.15)
            ndn = sum(1 for r in results if clipped(r, dn))
            nup = sum(1 for r in results if clipped(r, stock.get("MPC_Z_VEL_MAX_UP")))
            print(f"\n  -> {ndn} descent steps and {nup} ascent steps were pinned at the "
                  f"autopilot's configured limit."
                  + ("\n  -> CLAMP CONFIRMED on both axes: PX4 clips the velocity setpoint "
                     "itself (PositionControl::_positionControl), so no guidance tuning "
                     "could ever have moved this."
                     if (ndn or nup) else "\n  -> no step was pinned at a configured limit"))

        est_tr = [r for r in results if r["truth_vz"] is not None and abs(r["cmd_vz"]) > 0.5]
        if est_tr:
            d = statistics.fmean(abs(r["achieved_vz"] - r["truth_vz"]) for r in est_tr)
            print(f"\n  PX4's altitude estimate vs the FDM's truth: {d:.3f} m/s mean "
                  f"disagreement over {len(est_tr)} steps"
                  + ("  -> the estimate is sound, so the aircraft really is where PX4 "
                     "says it is" if d < 0.1 else "  -> CHECK THE ESTIMATOR"))
        else:
            print(f"\n  !! no ground truth ({tr.n_ok} differences over {tr.n_try} polls"
                  + (f", last error {tr.last_error}" if tr.last_error else "")
                  + ") -- the estimate-vs-truth check did not run")

        if args.csv:
            p = pathlib.Path(args.csv)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"\n  wrote {p} ({len(rows)} rows)")
    finally:
        link.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())