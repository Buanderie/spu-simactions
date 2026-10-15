#!/usr/bin/env python3
"""Fly a pass on the REAL flight stack: PX4 SITL against swarmpu-simuav's FDM.

The offline sweep measures the guidance against a first-order reference plant, which is
the right thing for tuning and the wrong thing for believing. This closes the same law on
PX4's actual autopilot -- its EKF, its TECS, its offboard checks, its real actuator
limits -- with the FDM as the only simulation in the loop, and scores the result with the
same scorer.

    scripts/rig.sh start                       # FDM + PX4, no Docker
    python3 scripts/fly_live.py --scenario offset_right_300
    python3 scripts/fly_live.py --all --airframe quad

Everything downstream of the law is identical to the offline path: same `PassSpec`, same
`TerminalProfileGuidance`, same `score` module, same plots. A number that appears in both
is comparable by construction, which is the entire reason for keeping the geometry free of
MAVLink.
"""
import argparse
import math
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import viz  # noqa: E402
from flyby import airframes, geo, score, scenarios  # noqa: E402
from flyby.stream import StreamWriter  # noqa: E402
from flyby.experiment import AsciiMap, Progress  # noqa: E402
from flyby.guidance import TerminalProfileGuidance, decompose  # noqa: E402
from flyby.mavlink_link import SETPOINT_HZ, Link, OffboardController  # noqa: E402


def pass_duration(scenario, margin=1.3, pad_s=15.0):
    """How long this approach needs, from the geometry rather than a constant.

    A fixed duration silently truncates the easy cases and wastes minutes on the hard ones:
    a 900 m approach at 12 m/s needs 78 s and a hard-coded 45 s reports "NO CROSSING" for a
    pass that was going fine.
    """
    tgt, tgt_vel = scenario.target_fn(0.0)
    s0, _e, s_dot, _ed = decompose(
        (scenario.start_ned[0] - tgt[0], scenario.start_ned[1] - tgt[1]),
        (scenario.start_vel_ned[0] - tgt_vel[0], scenario.start_vel_ned[1] - tgt_vel[1]),
        scenario.spec.angle_deg)
    # The closing rate to budget against is the APPROACH speed, not the arrival speed. A
    # ground approach arrives at 3 m/s over 900 m and that arithmetic says it needs five
    # minutes; it actually takes about a minute and a half, because it flies most of the
    # way at the approach speed and only slows near the end.
    v_app = max(1.0, scenario.spec.approach_speed_mps or scenario.spec.speed_mps)
    closing = max(1.0, min(max(s_dot, 0.35 * v_app), v_app))
    return abs(s0) / closing * margin + pad_s


def run_pass(ctrl, guidance, scenario, duration_s, verbose_map, quiet=False,
             stream=None):
    # Session time, so the pass continues the stream's timeline instead of restarting it.
    stream_base = stream.mark() if stream is not None else 0.0
    """Close the law on PX4 for one scenario. Returns (samples, scores)."""
    link = ctrl.link
    t0 = time.monotonic()
    samples = []
    amap = AsciiMap(scenario.spec, half_span_m=900.0,
                    cols=49, rows=25, origin=ctrl.origin) if verbose_map else None
    prog = Progress(enabled=not quiet, every_s=0.5)
    n = 0
    dt = 1.0 / SETPOINT_HZ
    next_draw = 0.0

    while time.monotonic() - t0 < duration_s:
        loop_t = time.monotonic() - t0
        link.pump(timeout=0.0)
        own = ctrl.own_ned()
        if own is None or link.vx == 0 and link.vy == 0 and link.vz == 0 and not samples:
            time.sleep(dt)
            continue

        tgt, tgt_vel = scenario.target_fn(loop_t)
        plan = guidance.update(own, ctrl.own_vel(), scenario.spec, loop_t,
                               target_ned=tgt, target_vel_ned=tgt_vel,
                               target_predict=(lambda dt_: scenario.target_fn(loop_t + dt_)))
        ctrl.step(plan)

        smp = score.Sample(t=loop_t, ned=own, vel=ctrl.own_vel(),
                           target_ned=tgt, target_vel=tgt_vel, plan=plan)
        samples.append(smp)
        if stream is not None:
            # One small JSON line per tick, flushed. On the control loop, so it must be
            # cheap: if watching the flight were able to slow the flight down, the flight
            # being watched would not be the one the rig would otherwise fly.
            stream.write(stream_base + loop_t, own, ctrl.own_vel(), tgt, plan)
        n += 1
        prog.tick(loop_t, smp, plan)
        if amap is not None and loop_t >= next_draw:
            sys.stderr.write("\033[2J\033[H" + amap.render(
                smp, f"{scenario.name}   t={loop_t:5.1f}s   {scenario.notes}") + "\n")
            next_draw = loop_t + 0.25
        time.sleep(dt)

    prog.end()
    if amap is not None:
        sys.stderr.write("\033[0m")
    return samples, score.score_run(samples, scenario.spec)


def apply_profile_params(link, profile):
    """Push the simuav profile's PX4 parameters -- AND simuav's own kind defaults.

    simuav normally does this in its mavlink relay (`engine/mavlink_relay.py`), which the rig
    bypasses: the controller connects to PX4 directly. So without this step a profile's
    `px4_params` block is decoration -- the FDM flies the profile's airframe while PX4 keeps
    the parameters of whatever `PX4_SIM_MODEL` booted, and the two halves disagree.

    That is not hypothetical. Measured on `mavic3`: the FDM ran it happily at real-time
    factor 1.0 while PX4 refused to arm with "Preflight Fail: Attitude failure (roll)"
    followed by "High Gyro Bias" -- an EKF that could not reconcile a rotor layout it had
    been told about against sensors from a different one. The iris profile gets away with an
    empty block precisely because the Iris IS PX4's default; every other aircraft has to say
    so explicitly.

    It matters MORE for the fixed wing, and for a sharper reason. `trainer.json` is empty by
    design: its PX4 parameters live in `engine/fdm/fixedwing.py`'s `PX4_PARAM_OVERRIDES`,
    and simuav's loader merges them for a `kind: plane` profile. A rig that only reads the
    JSON therefore pushes *nothing* for a plane, and PX4 keeps `gazebo-classic_plane`'s
    stock envelope -- `FW_AIRSPD_MAX` 20 m/s against an airframe that cruises at 30, so
    asking it to fly the pass speed is asking for something above its ceiling, and PX4
    answers with "Airspeed sensor failure detected" and an airspeed of `nan` on the GCS.
    So this calls simuav's own `px4_params()` helper rather than reimplementing the merge.

    Returns (pushed, problems).
    """
    out_dir = pathlib.Path(os.environ.get(
        "SIMUAV_PROFILE_DIR", "/home/nis/dev/swarmpu-simuav/config/profiles"))
    path = out_dir / f"{profile}.json"
    if not path.exists():
        print(f"  no simuav profile at {path}; skipping profile parameters")
        return {}, []
    # Import simuav's own loader, so the merge that gives a plane its parameter set lives
    # in exactly one place: theirs.
    simuav = "/home/nis/dev/swarmpu-simuav"
    if simuav not in sys.path:
        sys.path.insert(0, simuav)
    from engine.fdm import profiles as simuav_profiles

    prof = simuav_profiles.load_profile(str(path))
    floats, ints = simuav_profiles.px4_params(prof)
    want = dict(floats)
    want.update({k: float(v) for k, v in ints.items()})
    if not want:
        return {}, []
    # Wait for the link before asking PX4 anything. Pushing parameters before the heartbeat
    # and position estimate have arrived silently produces a table of `None` reads-back --
    # which looks exactly like PX4 refusing the parameters, and is not: the writes go out,
    # PX4 applies them, and the read-back simply never got an answer in time. So a
    # premature push reports a failure that did not happen, and a real one would be
    # indistinguishable from it.
    link.wait_heartbeat()
    link.wait_state(timeout_s=60.0)
    kind = prof.get("kind")
    print(f"  profile {profile} ({kind}): pushing {len(want)} PX4 parameters ...")
    got = link.set_params(want, timeout_s=10.0)
    problems = []
    for k in sorted(want):
        v = got.get(k)
        okk = v is not None and abs(v - want[k]) <= max(1e-3, abs(want[k]) * 1e-3)
        if not okk:
            problems.append(k)
            print(f"    {k:22s} {v}   !! wanted {want[k]}")
    print(f"    {len(want) - len(problems)}/{len(want)} applied"
          + ("" if not problems else "  (listed above)"))
    return got, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenario", default=None, help="one scenario name from the suite")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--airframe", choices=("quad", "plane"), default="quad")
    ap.add_argument("--profile", default=None,
                    help="the simuav aircraft profile being flown, which selects the "
                         "capability numbers (flyby/airframes.py). Default: the airframe's "
                         "own. Must match what scripts/rig.sh actually started, because the "
                         "vertical limits pushed to PX4 come from here.")
    ap.add_argument("--altitude", type=float, default=100.0, help="cruise height above home (m)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--map", action="store_true", help="live ASCII map in the terminal")
    ap.add_argument("--stream", default=None,
                    help="append this flight's telemetry here as JSON lines, live, so "
                         "scripts/live3d.py can show it in 3-D while it is still flying")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--suite", choices=("level", "ground"), default="level",
                    help="level = fly through a point in the air; ground = descend onto a "
                         "ground point at an angle, as slowly as the airframe allows")
    args = ap.parse_args()

    origin = geo.Origin()
    # One place decides what this aircraft can do, and the same object feeds the guidance,
    # the reference plant and (through OffboardController) PX4's own parameters. They used
    # to be three independent hardcoded numbers that could disagree -- and did.
    cap = airframes.get(args.profile or args.airframe)
    gcfg = cap.guidance_config()
    print(f"=== live pass on PX4 SITL + swarmpu-simuav FDM "
          f"({cap.name}: {cap.title or cap.kind}) ===")
    print(f"  capability [{cap.confidence}]: climb <= {cap.max_climb_mps:.1f} m/s, "
          f"descent <= {cap.max_descent_mps:.1f} m/s, speed <= {cap.max_speed_mps:.1f} m/s, "
          f"tau {cap.tau_horizontal_s:.2f}/{cap.tau_vertical_s:.2f} s")
    if cap.source:
        print(f"  from: {cap.source}")
    suite = (scenarios.build_ground_suite(origin, cap.kind) if args.suite == "ground"
             else scenarios.build_suite(origin, cap.kind))
    if args.scenario:
        suite = [s for s in suite if s.name == args.scenario]
        if not suite:
            print(f"no scenario called {args.scenario!r}; have: "
                  + ", ".join(s.name for s in scenarios.build_suite(origin, cap.kind)),
                  file=sys.stderr)
            return 2
    elif not args.all:
        suite = suite[:1]

    out = pathlib.Path(args.out or (ROOT / "runs" / f"live-{cap.name}"))
    out.mkdir(parents=True, exist_ok=True)

    link = Link()
    ctrl = OffboardController(link, suite[0].spec, None, airframe=cap.kind,
                              airframe_capability=cap)
    sw = None
    if args.stream:
        sw = StreamWriter(args.stream)
        sw.spec_json(suite[0].spec, motion="")
        ctrl.stream = sw
        print(f"  streaming to {args.stream}")
        print(f"  view it live at: python3 scripts/live3d.py {args.stream}   "
              f"(the viewer attaches before the flight starts)")
    try:
        # BEFORE prepare(): the profile's airframe parameters have to be in place before PX4
        # is asked to arm and fly it.
        if args.profile:
            _got, problems = apply_profile_params(link, args.profile)
            if problems:
                print(f"  !! {len(problems)} parameter(s) did not take: "
                      f"{', '.join(problems)}", file=sys.stderr)

        ctrl.prepare()
        # Climb out to the FIRST scenario's starting altitude. A ground approach starts
        # hundreds of metres up on its glide slope, so this is not "some cruise height" --
        # using one would start every ground pass from the wrong place and quietly measure a
        # different experiment.
        alt_down = suite[0].start_ned[2]
        print(f"  scenario start altitude: {abs(alt_down):.0f} m "
              f"{'above' if alt_down < 0 else 'below'} home")
        ctrl.climb_out(alt_down, timeout_s=150.0)

        results = []
        for sc in suite:
            ctrl.spec = sc.spec
            ctrl.guidance = TerminalProfileGuidance(gcfg)
            print(f"\n--- {sc.name} --- {sc.notes}")
            # Put the aircraft at the scenario's starting position first: the approach
            # geometry IS the experiment, and the climb-out left it wherever that ended.
            # The transit runs at the APPROACH speed, not the arrival speed: a ground
            # approach arrives at 3 m/s, and transiting there turns a 30 s move into a
            # two-minute one and hands the law over with the aircraft nearly stopped.
            #
            # Not below the airframe's minimum though, and that matters more for a fixed
            # wing than for anything else here: `transit_to` decelerates as it nears its
            # destination, so on a 51 m/s Geran commanded at 0.35*d it walks the speed
            # down through 21 m/s -- its own stall floor -- and stalls short. The floor is
            # the airframe's, so it is applied here and not baked into the transit.
            v_start = sc.spec.approach_speed_mps or sc.spec.speed_mps
            ctrl.transit_to(sc.start_ned, speed_mps=max(cap.min_speed_mps, 0.9 * v_start))
            ctrl.settle_velocity(sc.start_vel_ned, seconds=6.0)
            ctrl.set_pass_speed(sc.spec.speed_mps)
            time.sleep(1.5)   # let the airspeed setpoint settle before measuring
            dur = pass_duration(sc, margin=1.9)
            print(f"  flying for {dur:.0f} s (from the approach geometry, not a guess)")
            if sw is not None:
                sw.spec_json(sc.spec, motion=sc.target_label)
            samples, scores = run_pass(ctrl, ctrl.guidance, sc, dur, args.map,
                                       args.quiet, stream=sw)

            stem = f"{cap.name}-{sc.name}-live"
            if not scores:
                print("  NO CROSSING OF THE GATE")
            else:
                print("  " + scores[0].one_line())
            results.append((sc.name, scores[0] if scores else None))

            viz.plot_pass(samples, sc.spec, str(out / f"{stem}.png"), scores,
                          title=f"LIVE on PX4 — {sc.name} — {sc.notes}", origin=origin)
            viz.write_3d_player(samples, sc.spec, scores, str(out / f"{stem}-3d.html"),
                                notes="flown by PX4 SITL against swarmpu-simuav's FDM: "
                                      "PX4's EKF, TECS and offboard checks were all in the loop",
                                motion="live", origin=origin)
            (out / f"{stem}.csv").write_text(_csv(samples, sc))

        if results:
            ok = [s for _n, s in results if s and s.feasible]
            print(f"\n=== {len(ok)}/{len(results)} flyable ===")
            if ok:
                print(f"  mean miss {sum(s.miss_2d_m for s in ok)/len(ok):.2f} m   "
                      f"mean |angle| {sum(abs(s.angle_err_deg) for s in ok)/len(ok):.2f} deg   "
                      f"mean |speed| {sum(abs(s.speed_err_mps) for s in ok)/len(ok):.3f} m/s")
        print(f"\n  figures, 3-D players and telemetry in {out}")
        print(f"  open {out}/index.html if it exists, or any *-3d.html in a browser")
    finally:
        if sw is not None:
            sw.close()
        link.close()
    return 0


def _csv(samples, sc):
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["t", "north", "east", "down", "vn", "ve", "vd", "gs", "track_deg",
                "along_err_m", "cross_err_m", "tgt_north", "tgt_east", "tgt_down",
                "cmd_vn", "cmd_ve", "cmd_vd", "feasible"])
    for s in samples:
        gs = math.hypot(s.vel[0], s.vel[1])
        trk = math.degrees(math.atan2(s.vel[1], s.vel[0])) % 360.0 if gs > 0.3 else float("nan")
        _s, e, _sd, _ed = decompose(
            (s.ned[0] - s.target_ned[0], s.ned[1] - s.target_ned[1]),
            (s.vel[0] - s.target_vel[0], s.vel[1] - s.target_vel[1]), sc.spec.angle_deg)
        p = s.plan
        w.writerow([f"{s.t:.3f}"] + [f"{v:.3f}" for v in s.ned] + [f"{v:.3f}" for v in s.vel]
                   + [f"{gs:.3f}", f"{trk:.2f}", f"{_s:.2f}", f"{e:.2f}"]
                   + [f"{v:.3f}" for v in s.target_ned]
                   + ([f"{v:.3f}" for v in p.vel_ned] + [int(p.feasible)]) if p else ["", "", ""])
    return buf.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
