#!/usr/bin/env python3
"""Randomised benchmark: what fraction of passes does this law actually fly?

Everything measured so far in this repo is a handful of hand-picked scenarios. That is the
right way to develop and the wrong way to report: a suite is chosen by the person who built
the law, so it necessarily over-represents the cases that author understood, and a mean
over it is a statement about the suite as much as about the guidance. The question worth
asking is different and much blunter:

    hand me a pass you did not choose. What fraction of them do you fly?

So this samples the geometry instead of enumerating it -- random point, random starting
position, random pass bearing, random approach and arrival speeds, optionally a point that
is moving -- and reports a ratio, with the failures characterised rather than averaged away.

## The one distinction that makes the number mean anything

A pass can fail in two completely different ways and they must never be averaged together:

    REFUSED     the law said INFEASIBLE. The requirement was not achievable at any speed
                or attitude and the law said so instead of flying the least-bad answer.
                That is the system working.
    MISSED      the law said it was feasible and then missed. That is the system failing,
                and no amount of tolerance inflation turns it into anything else.

A benchmark that counts a refusal as a failure is measuring nothing, and one that counts a
miss as a refusal is measuring nothing. Both are reported, separately, at every cut.

## What "success" means, and why the tolerance is a table

A single pass/fail threshold is arbitrary and invites tuning the threshold until the number
looks good. So the tolerances are stated as a table of real requirements -- "went through
the gate within 2 m at within 3 deg of the required track and within 0.5 m/s of the required
speed" is a gate; "within 50 m" is not -- and the results are reported across the table, so
the reader can see how the ratio moves with the standard rather than being handed one number
with no standard attached.

    python3 scripts/bench.py                                   # level suite, quad
    python3 scripts/bench.py --suite ground --n 3000
    python3 scripts/bench.py --airframe plane --seed 7 --csv runs/bench-plane.csv
    python3 scripts/bench.py --worst 12 --figs runs/bench       # 3-D players of the worst

Everything is seeded, so a number here is reproducible and a before/after comparison
between two revisions of the law is a comparison and not an anecdote.
"""
import argparse
import csv
import math
import pathlib
import random
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import viz  # noqa: E402
from flyby import airframes, geo, scenarios  # noqa: E402
from flyby.stream import StreamWriter  # noqa: E402
from flyby.experiment import run_offline  # noqa: E402
from flyby.guidance import PassSpec  # noqa: E402

# (label, miss 2-D m, |track err| deg, |speed err| m/s, |alt err| m, |fp err| deg)
# Named for what they assert, so the table reads as a set of requirements rather than as a
# set of magic numbers. `alt` and `fp` only apply to a ground descent; None means "not
# scored for this family".
TOLERANCES = [
    ("strict",    1.0,  1.5, 0.20,  1.5,  1.5),
    ("gate",      3.0,  3.0, 0.50,  3.0,  3.0),
    ("loose",    10.0,  8.0, 1.50, 10.0,  8.0),
    ("ballpark", 50.0, 25.0, 5.00, 50.0, 25.0),
]


def verdict(sc, family, tol):
    """(passed, which term failed) for one scored crossing against one tolerance row."""
    _label, miss, ang, spd, alt, fp = tol
    def ok(v, t):
        return v is None or not math.isfinite(v) or abs(v) <= t
    if not ok(sc.miss_2d_m, miss):
        return False, "miss"
    # A free-bearing approach chose its own crossing direction, so arriving off it is not a
    # failure of anything that was asked for.
    if family != "final_free" and not ok(sc.angle_err_deg, ang):
        return False, "track"
    if not ok(sc.speed_err_mps, spd):
        return False, "speed"
    if family == "ground":
        if not ok(sc.alt_err_m, alt):
            return False, "alt"
        if sc.fp_angle_err_deg is not None and not ok(sc.fp_angle_err_deg, fp):
            return False, "fp"
    return True, ""


# ----------------------------------------------------------------- case generation

def sample_final(rng, cap, stats=None):
    """A FINAL APPROACH: a long, stabilised glide onto a ground point.

    Added because the stated use case is air-to-ground and the general `sample_ground`
    does not model one. It samples lateral deviation as ±350 m over a 300-1400 m range, and
    at the short end of that range 350 m is a 49 degree track error -- that is an INTERCEPT,
    not an approach, and a law can look mediocre on it while being fine at the thing it is
    actually for.

    What is different, and each difference is something a real final approach has:

      * range 600-3000 m, because the stabilised leg is long;
      * lateral deviation proportional to range (±8 % at 300 m is ±24 m, ±8 % at 3000 m is
        ±240 m) -- a localizer intercept closes as the range closes;
      * vertical deviation ±15 m on the glideslope, not ±40 m, because the vertical channel
        is stabilised long before the lateral one on a real approach;
      * a shallower demanded angle, because the approach speed is high and the descent
        capability is what limits it.
    """
    angle = rng.uniform(0.0, 360.0)
    speed, app, fp = sample_speeds(rng, cap, rng.uniform(4.0, 12.0), stats)
    dist = rng.uniform(600.0, 3000.0)
    off = rng.uniform(-0.08, 0.08) * dist
    dalt0 = rng.uniform(-15.0, 15.0)
    d = geo.track_unit(angle)
    r = geo.right_unit(angle)
    tgt_ned = (rng.uniform(-2500, 2500), rng.uniform(-2500, 2500), 0.0)
    height = max(0.0, dist * math.tan(math.radians(-fp)) + dalt0)
    start = (tgt_ned[0] - dist * d[0] + off * r[0],
             tgt_ned[1] - dist * d[1] + off * r[1],
             -height)
    spec_kwargs = dict(angle_deg=angle, speed_mps=speed, approach_speed_mps=app,
                       gate="closest", fp_angle_deg=fp)
    sc = _scenario(spec_kwargs, tgt_ned, start,
                   (app * d[0], app * d[1], -app * math.tan(math.radians(fp))),
                   lambda t: (tgt_ned, (0.0, 0.0, 0.0)), "final", "final approach")
    return sc, dict(dist=dist, off=off, speed=speed, angle=angle, fp=fp, motion=0.0,
                    dalt0=dalt0, motion_frac=0.0, speed_frac_of_app=1.0)


def sample_final_free(rng, cap, stats=None, heading_err_deg=15.0, bearing="arc"):
    """A final approach where only the POINT matters, not the direction it is crossed in.

    This is the fixed-wing use case: the aircraft starts with its nose roughly toward the
    point and has to arrive on it, from whatever direction it is already coming. There is
    no pass line to capture, so there is no lateral offset -- only a heading error between
    the nose and the line of sight, drawn from ±heading_err_deg.

    The law still needs a pass bearing, and which one is chosen is the whole question:

      * "los": the line-of-sight bearing at the start. The aircraft has to turn its heading
        error out AND come back onto the line -- an S-turn, the thing a slow-turning fixed
        wing is worst at.
      * "arc": the initial heading mirrored about the line of sight (LOS - psi0). A circular
        arc tangent to the initial velocity that passes through the point arrives at exactly
        this bearing, so the requirement is one constant-bank turn and nothing else.

    `final` is scored without a track-angle term for this family (see verdict): the bearing
    was the law's choice, not a requirement, so arriving off it is not a miss.
    """
    los = rng.uniform(0.0, 360.0)
    psi0 = rng.uniform(-heading_err_deg, heading_err_deg)
    speed, app, fp = sample_speeds(rng, cap, rng.uniform(4.0, 12.0), stats)
    dist = rng.uniform(600.0, 3000.0)
    dalt0 = rng.uniform(-15.0, 15.0)
    u = geo.track_unit(los)
    h = geo.track_unit(los + psi0)
    angle = (los - psi0) % 360.0 if bearing == "arc" else los
    tgt_ned = (rng.uniform(-2500, 2500), rng.uniform(-2500, 2500), 0.0)
    height = max(0.0, dist * math.tan(math.radians(-fp)) + dalt0)
    start = (tgt_ned[0] - dist * u[0], tgt_ned[1] - dist * u[1], -height)
    spec_kwargs = dict(angle_deg=angle, speed_mps=speed, approach_speed_mps=app,
                       gate="closest", fp_angle_deg=fp)
    sc = _scenario(spec_kwargs, tgt_ned, start,
                   (app * h[0], app * h[1], -app * math.tan(math.radians(fp))),
                   lambda t: (tgt_ned, (0.0, 0.0, 0.0)), "final_free",
                   f"free-bearing final approach ({bearing})")
    return sc, dict(dist=dist, off=0.0, speed=speed, angle=angle, fp=fp, motion=0.0,
                    dalt0=dalt0, motion_frac=0.0, speed_frac_of_app=1.0, psi0=psi0)


def sample_final_moving(rng, cap, stats=None):
    """A final approach onto a point that walks -- a vehicle, which is the usual case."""
    base, meta = sample_final(rng, cap, stats)
    tgt = base.target_fn(0.0)[0]
    frac = rng.uniform(0.02, 0.30)
    v = base.spec.approach_speed_mps * frac
    th = rng.uniform(0.0, 360.0)
    tv = (v * math.cos(math.radians(th)), v * math.sin(math.radians(th)), 0.0)
    base.target_fn = lambda t: ((tgt[0] + tv[0] * t, tgt[1] + tv[1] * t, tgt[2]), tv)
    base.target_label = "walking"
    meta["motion"] = v
    meta["motion_frac"] = frac
    return base, meta


def _scenario(spec_kwargs, tgt_ned, start_ned, start_vel, target_fn, name, notes):
    """Build a Scenario with the spec's lat/lon/alt derived from the NED geometry.

    PassSpec carries geodetic coordinates and exposes NED as a read-only property, so a
    random case that starts life in metres has to be converted rather than assigned --
    assigning to `spec.ned` raises, and quietly working around that would mean the plots
    and the scorer disagree about where the point is.
    """
    lat, lon, alt = geo.ned_to_latlon_alt(*tgt_ned)
    spec = PassSpec(lat=lat, lon=lon, alt_m=alt, **spec_kwargs)
    assert all(abs(a - b) < 1e-6 for a, b in zip(spec.ned, tgt_ned)), \
        f"spec/geometry mismatch: {spec.ned} vs {tgt_ned}"
    return scenarios.Scenario(name=name, spec=spec, start_ned=start_ned,
                              start_vel_ned=start_vel, target_fn=target_fn, notes=notes)


def sample_speeds(rng, cap, fp_abs_deg=None, stats=None):
    """Arrival speed, approach speed and (optionally) a flight-path angle the aircraft CAN fly.

    This is the difference between benchmarking the law and benchmarking one's imagination.
    Sampling speeds and angles independently produces combinations no aircraft in the fleet
    could execute -- an iris-class quadrotor asked to start an approach at 29 m/s when its
    ceiling is 15, or to descend at 9.9 m/s when it can do 6 -- and those cases come back
    INFEASIBLE, which is the law correctly refusing the impossible and tells you nothing
    about the law at all. In the first version of this benchmark that mistake made 88 % of
    the ground suite "failures" and the real answer was a handful of cases.

    So the samples are drawn inside the airframe's envelope, and every time a draw had to be
    pulled back to fit, it is counted and reported. The envelope is `flyby.airframes`, so
    this stays honest for every aircraft added to the fleet.
    """
    v_hi = min(cap.max_speed_mps, 40.0)
    v_pass = rng.uniform(cap.min_speed_mps, max(cap.min_speed_mps * 1.5, v_hi * 0.5))
    app = rng.uniform(v_pass, v_hi)
    fp = None
    if fp_abs_deg is not None:
        fp = -abs(fp_abs_deg)
        # The demanded descent rate is the APPROACH speed times sin(fp), because the descent
        # has to be flying before the aircraft has slowed to the arrival speed.
        req = app * math.sin(math.radians(abs(fp)))
        lim = min(cap.max_descent_mps, cap.max_climb_mps)
        if req > lim:
            fp = -math.degrees(math.asin(max(0.02, lim / max(app, 0.1))))
            if stats is not None:
                stats["fp_clipped"] = stats.get("fp_clipped", 0) + 1
    if stats is not None:
        stats["cases"] = stats.get("cases", 0) + 1
    return v_pass, app, fp


def sample_level(rng, cap, stats=None):
    """A fly-through: point at altitude, aircraft lined up on the pass bearing."""
    angle = rng.uniform(0.0, 360.0)
    speed, app, _ = sample_speeds(rng, cap, None, stats)
    dist = rng.uniform(150.0, 1200.0)
    off = rng.uniform(-400.0, 400.0)
    d = geo.track_unit(angle)
    r = geo.right_unit(angle)
    tgt_ned = (rng.uniform(-1500, 1500), rng.uniform(-1500, 1500), -rng.uniform(50, 600))
    start = (tgt_ned[0] - dist * d[0] + off * r[0],
             tgt_ned[1] - dist * d[1] + off * r[1],
             tgt_ned[2] - rng.uniform(-200, 200))
    spec_kwargs = dict(angle_deg=angle, speed_mps=speed, approach_speed_mps=app, gate="line")
    sc = _scenario(spec_kwargs, tgt_ned, start, (app * d[0], app * d[1], 0.0),
                   lambda t: (tgt_ned, (0.0, 0.0, 0.0)), "bench", "random fly-through")
    return sc, dict(dist=dist, off=off, speed=speed, angle=angle, motion=0.0, dalt0=0.0,
                    motion_frac=0.0, speed_frac_of_app=1.0)


def sample_ground(rng, cap, stats=None):
    """A terminal descent: point on the ground, arrive at an angle, as slowly as allowed."""
    angle = rng.uniform(0.0, 360.0)
    speed, app, fp = sample_speeds(rng, cap, rng.uniform(5.0, 25.0), stats)
    dist = rng.uniform(300.0, 1400.0)
    off = rng.uniform(-350.0, 350.0)
    d = geo.track_unit(angle)
    r = geo.right_unit(angle)
    tgt_ned = (rng.uniform(-1500, 1500), rng.uniform(-1500, 1500), 0.0)
    # Start ON the glide slope, displaced by `off` across it, plus a random vertical error.
    # Both are what a real approach has, and both are what make the case hard or easy.
    #
    # NED z is DOWN-positive and the point is on the ground at z = 0, so a start `h` metres
    # ABOVE it is z = -h. Getting that sign wrong starts every case underground and makes
    # the entire suite report infeasible, which looks exactly like a broken law and is not
    # one. Clamped at ground level so a case cannot begin inside the world.
    # How far above (or below) the ideal glide slope the aircraft starts. This is the single
    # variable that decides whether a ground descent is flyable at all, so it is bounded to
    # a realistic 40 m and then REPORTED as a bucket rather than averaged away: an
    # approach that starts 40 m high has to dive steeper than the demanded angle and then
    # flatten, which is a different problem from one that starts on the slope.
    dalt0 = rng.uniform(-40.0, 40.0)
    height = max(0.0, dist * math.tan(math.radians(-fp)) + dalt0)
    start = (tgt_ned[0] - dist * d[0] + off * r[0],
             tgt_ned[1] - dist * d[1] + off * r[1],
             -height)
    spec_kwargs = dict(angle_deg=angle, speed_mps=speed, approach_speed_mps=app,
                       gate="closest", fp_angle_deg=fp)
    sc = _scenario(spec_kwargs, tgt_ned, start,
                   (app * d[0], app * d[1], -app * math.tan(math.radians(fp))),
                   lambda t: (tgt_ned, (0.0, 0.0, 0.0)), "bench", "random descent")
    return sc, dict(dist=dist, off=off, speed=speed, angle=angle, fp=fp, motion=0.0,
                    dalt0=dalt0, motion_frac=0.0, speed_frac_of_app=1.0)


def sample_ground_moving(rng, cap, stats=None):
    """A terminal descent onto a point that walks. The hardest family here."""
    base, meta = sample_ground(rng, cap, stats)
    tgt = base.target_fn(0.0)[0]
    frac = rng.uniform(0.05, 0.55)                 # fraction of the APPROACH speed
    v = base.spec.approach_speed_mps * frac
    meta["motion_frac"] = frac
    th = rng.uniform(0.0, 360.0)
    tv = (v * math.cos(math.radians(th)), v * math.sin(math.radians(th)), 0.0)
    base.target_fn = lambda t: ((tgt[0] + tv[0] * t, tgt[1] + tv[1] * t, tgt[2]), tv)
    base.target_label = "walking"
    meta["motion"] = v
    meta["speed_frac_of_app"] = 1.0
    return base, meta


# ---------------------------------------------------------------------- reporting

def stream_case(args, samples, sc, meta, family, score, why):
    """Publish one simulated case to the live 3-D viewer's file.

    The point is that the benchmark is a thing you can WATCH. A pass is a 3-D event -- a
    descent onto a point, an intercept of a moving one -- and the difference between a table
    saying "62 % strict" and seeing sixty approaches arrive, overshoot, or come in short is
    the difference between a number and an understanding of it.

    Each case truncates the file, which the viewer treats as a new flight and resets on, so
    one case replaces the previous rather than accumulating. The hold afterwards is pacing,
    not physics: a case simulates in well under a second and without a pause the view is a
    strobe.
    """
    sw = StreamWriter(args.stream)
    try:
        sw.spec_json(sc.spec, motion=(
            f"range {meta['dist']:.0f} m, offset {meta['off']:+.0f} m, "
            f"{'pass' if family == 'level' else 'descent'} {meta['speed']:.1f} m/s"
            + (f", fp {meta.get('fp', 0):+.1f} deg" if family != "level" else "")
            + (f", target {meta.get('motion', 0):.2f} m/s")))
        # The simulator produces the whole trajectory in one call, so the samples are
        # replayed into the file rather than produced live. Stated plainly because it matters
        # for what the view means: this is the EXACT trajectory the measurement used, shown
        # afterwards, not a second run that merely looks like the first.
        # The gate must be located BEFORE the samples are written. mark_gate() only sets a
        # field that write() stamps onto the row it is currently emitting, so marking after
        # the loop flagged a sample that had already been serialised -- which is why the
        # viewer showed no GATE cross at all.
        if score is not None:
            # The gate marker goes at the FIRST local minimum of range to the point, not at
            # the global minimum. A straight-in approach closes monotonically, so the two
            # agree; but ~6% of these trajectories swing wide past the point and come back
            # (the law switching to fly_on while it is still correcting laterally), which
            # makes the global minimum the *second* crossing. Marking that one put the
            # "GATE" cross hundreds of metres past the approach the pilot would actually
            # call the pass, and made a good approach look like a 30 m miss.
            rng = [math.hypot(s.ned[0] - s.target_ned[0], s.ned[1] - s.target_ned[1])
                   for s in samples]
            cands = [k for k in range(len(rng))
                     if (k == 0 or rng[k] <= rng[k - 1])
                     and (k == len(rng) - 1 or rng[k] <= rng[k + 1])]
            j = min(cands, key=lambda k: rng[k]) if cands else min(range(len(rng)),
                                                                 key=lambda k: rng[k])
            sw.mark_gate(samples[j].t, samples[j].ned, rng[j])
        for smp in samples:
            sw.write(smp.t, smp.ned, smp.vel, smp.target_ned, smp.plan)
    finally:
        sw.close()
    if args.stream_hold > 0:
        time.sleep(args.stream_hold)


def bucket_rows(rows, key, bins, label):
    out = []
    for lo, hi in bins:
        sel = [r for r in rows if lo <= r[key] < hi]
        if not sel:
            continue
        out.append((label, lo, hi, summarise(sel)))
    return out


def summarise(rows):
    n = len(rows)
    refusals = sum(1 for r in rows if r["refused"])
    # Accuracy statistics are over the cases the law ACCEPTED, and say so. Including refused
    # cases in the mean is not merely untidy, it inverts the conclusion: a run that refuses
    # 49 % of its cases will otherwise report a larger mean miss than one that refuses none,
    # because the misses being averaged in are the misses it already called failures. The
    # headline "mean miss 7.17 m" that this produced for the ground suite was really 0.01 m
    # over the flown cases and 7.17 m over all of them.
    flown = [r for r in rows if r["miss"] is not None and not r["refused"]]
    passed = {t[0]: 0 for t in TOLERANCES}
    for r in rows:
        if r["score"] is not None:
            for t in TOLERANCES:
                ok, _ = verdict(r["score"], r["family"], t)
                passed[t[0]] += 1 if ok else 0
    xs = [r["miss"] for r in flown]
    xs_sorted = sorted(xs)
    return dict(n=n, refused=refusals, flown=len(flown),
                no_gate=sum(1 for r in rows if r["miss"] is None and not r["refused"]),
                miss_mean=statistics.fmean(xs) if xs else float("nan"),
                miss_median=xs_sorted[len(xs) // 2] if xs else float("nan"),
                miss_worst=max(xs, default=float("nan")),
                passed={k: 100.0 * v / n for k, v in passed.items()})


def print_table(title, rows, family):
    s = summarise(rows)
    print(f"\n--- {title} ---")
    print(f"  cases {s['n']}   refused (infeasible) {s['refused']} "
          f"({100.0 * s['refused'] / s['n']:.1f}%)   never reached the gate {s['no_gate']}")
    print(f"  of the {s['flown']} cases it accepted: miss mean {s['miss_mean']:.3f} m   "
          f"median {s['miss_median']:.3f} m   worst {s['miss_worst']:.2f} m")
    print(f"  ({s['refused']} refused, {s['no_gate']} never reached the gate)")
    print("  success ratio by tolerance:")
    for t in TOLERANCES:
        print(f"    {t[0]:9s} <= {t[1]:5.1f} m, {t[2]:4.1f} deg, {t[3]:4.2f} m/s"
              + (f", {t[4]:4.1f} m, {t[5]:4.1f} deg" if family == "ground" else "")
              + f"   ->  {s['passed'][t[0]]:5.1f} %")
    return s


def run_fleet(args):
    """The same law, the same seed, every aircraft -- one row each.

    This is the question the single-airframe mode cannot answer. Accuracy is not a property
    of the guidance law alone: it is the law's parameters against an aircraft's response,
    and the single-airframe number hides that by reporting one aircraft's answer as if it
    were the law's. Holding the seed and the case distribution fixed and changing only the
    airframe turns that into a measurement, and the ordering it produces is the useful part.
    """
    names = args.airframes or airframes.available()
    families = (["level"] if args.suite == "level"
                else ["ground"] if args.suite == "ground"
                else ["final"] if args.suite == "final"
                else ["final_free"] if args.suite == "final_free" else ["level", "ground"])
    print(f"=== fleet benchmark: {len(names)} aircraft, "
          f"{args.n} cases per family, seed {args.seed} ===")
    print()
    hdr = (f"{'airframe':16s} {'kind':5s} {'conf':9s} {'tau_h':>6s} {'v_min':>6s} {'v_max':>6s} "
           f"{'refused':>7s} {'strict':>7s} {'gate':>7s} {'mean':>9s} {'worst':>8s}")
    print(hdr)
    print("-" * len(hdr))
    rows_out = []
    for name in names:
        try:
            cap = airframes.get(name)
        except Exception as e:
            print(f"{name:16s} ERROR {e}")
            continue
        gcfg, pcfg = cap.guidance_config(), cap.plant_config()
        if args.lead_frac is not None:
            gcfg.tau_lead_frac = args.lead_frac
        allrows = []
        for family in families:
            rng = random.Random(args.seed + (0 if family == "level" else 977))
            for _i in range(args.n):
                if family == "level":
                    sc, meta = sample_level(rng, cap, {})
                elif family == "final":
                    sc, meta = (sample_final_moving(rng, cap, {}) if rng.random() < args.moving_frac
                                else sample_final(rng, cap, {}))
                elif family == "final_free":
                    sc, meta = sample_final_free(rng, cap, {}, args.heading_err)
                elif rng.random() < args.moving_frac:
                    sc, meta = sample_ground_moving(rng, cap, {})
                else:
                    sc, meta = sample_ground(rng, cap, {})
                try:
                    samples, scores, _ = run_offline(sc, gcfg, pcfg, dt=args.dt)
                except Exception as e:
                    allrows.append(dict(family=family, refused=True, score=None,
                                        miss=None, why=f"EXCEPTION {e!r}", **meta))
                    continue
                if not scores:
                    allrows.append(dict(family=family, refused=True, score=None, miss=None,
                                        why="NO GATE EVENT", **meta))
                    continue
                s = scores[0]
                why = verdict(s, family, TOLERANCES[1])[1]
                allrows.append(dict(family=family, refused=not s.feasible, score=s,
                                    miss=s.miss_2d_m, samples=samples, spec=sc.spec,
                                    why=why, **meta))
                if args.stream:
                    stream_case(args, samples, sc, meta, family, s, why)
        s = summarise(allrows)
        print(f"{cap.name:16s} {cap.kind:5s} {cap.confidence:9s} {cap.tau_horizontal_s:6.2f} "
              f"{cap.min_speed_mps:6.1f} {cap.max_speed_mps:6.1f} "
              f"{100.0 * s['refused'] / s['n']:6.1f}% {s['passed']['strict']:6.1f}% "
              f"{s['passed']['gate']:6.1f}% {s['miss_mean']:9.3f} {s['miss_worst']:8.2f}")
        rows_out.append((cap, s))
        if args.csv:
            _write_fleet_csv(pathlib.Path(args.csv), rows_out)

    if len(rows_out) > 1:
        good = [r for r in rows_out if r[0].kind == "quad"]
        bad = [r for r in rows_out if r[0].kind == "plane"]
        print("\n  by kind:")
        for nm, grp in (("multirotor", good), ("fixed wing", bad)):
            if grp:
                print(f"    {nm:12s} strict {min(r[1]['passed']['strict'] for r in grp):5.1f}"
                      f"..{max(r[1]['passed']['strict'] for r in grp):5.1f}%   "
                      f"accepted mean miss "
                      f"{min(r[1]['miss_mean'] for r in grp):.3f}"
                      f"..{max(r[1]['miss_mean'] for r in grp):.3f} m")
        print("\n  strict success vs turn-response time constant:")
        for cap, s in sorted(rows_out, key=lambda r: r[0].tau_horizontal_s):
            print(f"    tau {cap.tau_horizontal_s:6.2f} s  {cap.name:16s} "
                  f"strict {s['passed']['strict']:5.1f}%  refused "
                  f"{100.0 * s['refused'] / s['n']:5.1f}%  mean miss "
                  f"{s['miss_mean']:9.3f} m")
    return 0


def _write_fleet_csv(path, rows_out):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["airframe", "kind", "confidence", "tau_horizontal_s", "min_speed_mps",
                    "max_speed_mps", "cases", "refused_pct", "accepted", "strict_pct",
                    "gate_pct", "loose_pct", "ballpark_pct", "accepted_mean_miss_m",
                    "accepted_median_miss_m", "accepted_worst_miss_m"])
        for cap, s in rows_out:
            w.writerow([cap.name, cap.kind, cap.confidence, f"{cap.tau_horizontal_s:.2f}",
                        f"{cap.min_speed_mps:.2f}", f"{cap.max_speed_mps:.2f}", s["n"],
                        f"{100.0 * s['refused'] / s['n']:.2f}", s["flown"],
                        f"{s['passed']['strict']:.2f}", f"{s['passed']['gate']:.2f}",
                        f"{s['passed']['loose']:.2f}", f"{s['passed']['ballpark']:.2f}",
                        f"{s['miss_mean']:.4f}", f"{s['miss_median']:.4f}",
                        f"{s['miss_worst']:.4f}"])
    print(f"\n  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--airframe", default="quad", help="flyby.airframes profile name")
    ap.add_argument("--suite", choices=("level", "ground", "final", "final_free", "both"),
                    default="both",
                    help="level = fly-through a point in the air; ground = intercept onto a "
                         "ground point (large offsets); final = a stabilised FINAL APPROACH "
                         "onto a ground point, which is the air-to-ground case; final_free = the "
                         "same with the crossing direction left free and the nose within "
                         "--heading-err of the point (the fixed-wing case); both = all")
    ap.add_argument("--n", type=int, default=600, help="cases per family")
    ap.add_argument("--heading-err", type=float, default=15.0,
                    help="final_free: the nose starts within +-this many degrees of the "
                         "line of sight to the point; the crossing direction is not scored")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--moving-frac", type=float, default=0.35,
                    help="share of ground cases whose point is walking")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--lead-frac", type=float, default=None,
                    help="cap the read-ahead at this fraction of the time-to-go "
                         "(see GuidanceConfig.tau_lead_frac)")
    ap.add_argument("--stream", default=None,
                    help="write every simulated case to this file as it is flown, so "
                         "scripts/live3d.py shows the benchmark running. Each case "
                         "truncates the file, which the viewer detects and resets on.")
    ap.add_argument("--stream-hold", type=float, default=2.0,
                    help="seconds to leave each streamed case on screen. Purely pacing for "
                         "the eye: a case simulates in well under a second, so without it "
                         "the view is a strobe and nothing is watchable. No effect on any "
                         "measurement.")
    ap.add_argument("--fleet", action="store_true",
                    help="compare every aircraft in one table: same law, same seed, "
                         "same case distribution, only the airframe changes")
    ap.add_argument("--airframes", nargs="*", default=None,
                    help="restrict --fleet to these profile names")
    ap.add_argument("--worst", type=int, default=0, help="write 3-D players for the N worst")
    ap.add_argument("--figs", default=None)
    args = ap.parse_args()

    if args.fleet:
        return run_fleet(args)

    cap = airframes.get(args.airframe)
    gcfg, pcfg = cap.guidance_config(), cap.plant_config()
    if args.lead_frac is not None:
        gcfg.tau_lead_frac = args.lead_frac
    origin = geo.Origin()
    print(f"=== randomised pass benchmark ===")
    print(f"  airframe   {cap.name} ({cap.kind}) [{cap.confidence}]  {cap.title}")
    print(f"  capability climb <= {cap.max_climb_mps:.1f}  descent <= {cap.max_descent_mps:.1f} "
          f"m/s  speed <= {cap.max_speed_mps:.1f} m/s  min {cap.min_speed_mps:.1f} m/s")
    print(f"  plant      tau {pcfg.tau_horizontal_s:.2f}/{pcfg.tau_vertical_s:.2f} s   "
          f"lead {gcfg.tau_lead_s:.2f} s   dt {args.dt}s")
    print(f"  seed {args.seed}   {args.n} cases per family")

    families = (["level"] if args.suite == "level"
                else ["ground"] if args.suite == "ground"
                else ["final"] if args.suite == "final"
                else ["final_free"] if args.suite == "final_free" else ["level", "ground"])
    all_rows, worst = [], []
    t_start = time.monotonic()

    for family in families:
        rng = random.Random(args.seed + (0 if family == "level" else 977))
        stats = {}
        rows = []
        for i in range(args.n):
            if family == "level":
                sc, meta = sample_level(rng, cap, stats)
            elif family == "final":
                sc, meta = (sample_final_moving(rng, cap, stats) if rng.random() < args.moving_frac
                            else sample_final(rng, cap, stats))
            elif family == "final_free":
                sc, meta = sample_final_free(rng, cap, stats, args.heading_err)
            elif rng.random() < args.moving_frac:
                sc, meta = sample_ground_moving(rng, cap, stats)
            else:
                sc, meta = sample_ground(rng, cap, stats)
            try:
                samples, scores, traj = run_offline(sc, gcfg, pcfg, dt=args.dt)
            except Exception as e:
                rows.append(dict(family=family, refused=True, score=None, passed=None,
                                 miss=None, why=f"EXCEPTION {e!r}", **meta))
                continue
            if not scores:
                # No gate event at all: the law never got there. Count it as neither a
                # refusal nor a pass, and say so, rather than quietly scoring it zero.
                rows.append(dict(family=family, refused=True, score=None, passed=None,
                                 miss=None, why="NO GATE EVENT", **meta))
                continue
            s = scores[0]
            # "Refused" is the law's OWN verdict at the gate: it reported infeasible for a
            # requirement that cannot be met, rather than flying the least-bad answer. The
            # per-tick fraction is kept alongside it because the verdict is dynamic -- a law
            # can be infeasible early and feasible later -- and a run that was flagged on
            # half its ticks and then missed is not the same thing as one that never believed
            # it could do it.
            ticks = [sm.plan for sm in samples if sm.plan is not None]
            frac_feasible = (sum(1 for p in ticks if p.feasible) / len(ticks)) if ticks else 1.0
            refused = not s.feasible
            ok, why = verdict(s, family, TOLERANCES[1])
            rows.append(dict(family=family, refused=bool(refused), score=s, passed=ok,
                             miss=s.miss_2d_m, why=why, samples=samples, spec=sc.spec,
                             frac_feasible=frac_feasible, **meta))
            if args.stream:
                stream_case(args, samples, sc, meta, family, s, why)
        rows.sort(key=lambda r: (r["miss"] is None, -(r["miss"] or 0.0)))
        all_rows.extend(rows)

        print_table(f"{family} suite", rows, family)
        if stats.get("fp_clipped"):
            print(f"  note: {stats['fp_clipped']} of {stats['cases']} draws wanted a "
                  f"flight-path angle steeper than the airframe's descent capability and "
                  f"were pulled back to it ({100.0 * stats['fp_clipped'] / stats['cases']:.0f} %)")

        # Where it fails, not just how often: a single ratio hides that it is one geometry.
        print("  failure breakdown (by the 'gate' tolerance):")
        by_why = {}
        for r in rows:
            if r["refused"]:
                by_why.setdefault("refused: infeasible", []).append(r)
            elif r["why"]:
                by_why.setdefault(f"missed: {r['why']}", []).append(r)
            else:
                by_why.setdefault("passed", []).append(r)
        for k in sorted(by_why, key=lambda k: -len(by_why[k])):
            print(f"    {k:22s} {len(by_why[k]):5d}  ({100.0 * len(by_why[k]) / len(rows):5.1f} %)")

        print("  success by approach geometry (gate tolerance):")
        for lo, hi, sel, nm in (
                (0, 400, [r for r in rows if r["dist"] < 400], "range"),
                (400, 800, [r for r in rows if 400 <= r["dist"] < 800], "range"),
                (800, 1400, [r for r in rows if r["dist"] >= 800], "range"),
                (0, 50, [r for r in rows if abs(r["off"]) < 50], "offset"),
                (50, 200, [r for r in rows if 50 <= abs(r["off"]) < 200], "offset"),
                (200, 400, [r for r in rows if abs(r["off"]) >= 200], "offset")):
            if not sel:
                continue
            s2 = summarise(sel)
            print(f"    {nm:7s} [{lo:4d},{hi:4d})  n={s2['n']:5d}  refused "
                  f"{100.0 * s2['refused'] / s2['n']:5.1f} %  gate-success "
                  f"{s2['passed']['gate']:5.1f} %")
        if family == "ground":
            # The cliff, if there is one: success against how far above the glide the
            # aircraft started.
            for lo, hi, nm in ((-1e9, -20.0, "20 m BELOW the glide"),
                               (-20.0, -5.0, "5-20 m below"),
                               (-5.0, 5.0, "on the glide"),
                               (5.0, 20.0, "5-20 m above"),
                               (20.0, 1e9, "20 m+ above")):
                sel = [r for r in rows if lo <= r.get("dalt0", 0.0) < hi]
                if not sel:
                    continue
                s2 = summarise(sel)
                print(f"    start {nm:19s} n={s2['n']:5d}  refused "
                      f"{100.0 * s2['refused'] / s2['n']:5.1f} %  gate-success "
                      f"{s2['passed']['gate']:5.1f} %")
            # A walking point is not one thing. The arrival-speed floor is GEOMETRIC --
            # V >= |v_t . r| / sin(cross-angle budget) -- so a point walking at a fraction
            # of the approach speed raises the speed the aircraft must carry into the gate,
            # and past some fraction that exceeds what the pass asked for and the law
            # correctly refuses. Bucketing by the fraction says where that line is, which is
            # the useful number; "moving points fail" is not.
            for lo, hi, nm in ((0.0, 0.05, "static"),
                               (0.05, 0.10, "walking <10% of v_app"),
                               (0.10, 0.20, "walking 10-20%"),
                               (0.20, 0.35, "walking 20-35%"),
                               (0.35, 1e9, "walking >35%")):
                # Bucket on the FRACTION of the approach speed the point walks at, which
                # is what `motion_frac` already is. `motion` is an absolute speed and
                # dividing it by the approach speed again would be a different quantity.
                sel = [r for r in rows
                       if lo <= r.get("motion_frac", 0.0) < hi]
                if not sel:
                    continue
                s2 = summarise(sel)
                print(f"    target {nm:22s} n={s2['n']:5d}  refused "
                      f"{100.0 * s2['refused'] / s2['n']:5.1f} %  gate-success "
                      f"{s2['passed']['gate']:5.1f} %")
        # The worst cases the law ACCEPTED. The worst-scoring cases overall are ones it
        # refused, and drawing those as "worst" implies the law flew them badly when what
        # it actually did was decline -- which is the behaviour being celebrated elsewhere
        # in this report, so showing them here would contradict it.
        flown = [r for r in rows if r["miss"] is not None and not r["refused"]]
        worst.extend(flown[:max(args.worst, 0)])

    dt_run = time.monotonic() - t_start
    total = summarise(all_rows)
    print(f"\n=== overall: {total['n']} cases in {dt_run:.0f}s ===")
    print(f"  refused {100.0 * total['refused'] / total['n']:.1f} %   "
          + "   ".join(f"{t[0]} {total['passed'][t[0]]:5.1f} %" for t in TOLERANCES))

    if args.csv:
        p = pathlib.Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["family", "refused", "why", "miss2d_m", "miss3d_m", "trk_err_deg",
                        "spd_err_mps", "alt_err_m", "fp_err_deg", "dist_m", "off_m",
                        "speed_mps", "motion_mps", "fp_demanded_deg", "pass_strict",
                        "pass_gate", "pass_loose", "pass_ballpark"])
            for r in all_rows:
                s = r["score"]
                g = (lambda f: getattr(s, f) if s is not None else None)
                w.writerow([r["family"], int(bool(r["refused"])), r["why"],
                            f"{g('miss_2d_m'):.3f}" if s else "",
                            f"{g('miss_3d_m'):.3f}" if s else "",
                            f"{g('angle_err_deg'):.3f}" if s else "",
                            f"{g('speed_err_mps'):.3f}" if s else "",
                            f"{g('alt_err_m'):.3f}" if s else "",
                            f"{g('fp_angle_err_deg'):.3f}" if s and s.fp_angle_err_deg
                            is not None else "",
                            f"{r['dist']:.1f}", f"{r['off']:.1f}", f"{r['speed']:.2f}",
                            f"{r['motion']:.2f}",
                            f"{r.get('fp', float('nan')):.2f}" if r["family"] == "ground" else "",
                            *[int(verdict(s, r["family"], t)[0]) if s else 0 for t in TOLERANCES]])
        print(f"\n  wrote {p}")

    if args.worst and args.figs:
        out = pathlib.Path(args.figs)
        out.mkdir(parents=True, exist_ok=True)
        drawn, cards = 0, []
        for i, r in enumerate(sorted(worst, key=lambda r: -(r["miss"] or 0.0))[:args.worst]):
            samples = r.get("samples")
            if not samples:
                continue
            drawn += 1
            stem = f"worst-{i + 1:02d}-{r['family']}-miss{r['miss']:.0f}m"
            viz.plot_pass(samples, r["spec"], str(out / f"{stem}.png"), [r["score"]],
                          title=f"WORST #{i+1}: {r['family']} miss {r['miss']:.1f} m "
                                f"({r['why']})", origin=origin)
            viz.write_3d_player(samples, r["spec"], [r["score"]], str(out / f"{stem}-3d.html"),
                                notes=f"random case: range {r['dist']:.0f} m, offset "
                                      f"{r['off']:.0f} m, pass {r['speed']:.1f} m/s"
                                      + (f", demanded flight path {r.get('fp', 0):+.1f} deg"
                                         if r["family"] == "ground" else ""),
                                motion="bench", origin=origin)
            cards.append((stem, r))
        _write_replay_index(out, cards, args.airframe, args.suite, args.n, args.seed)
        print(f"  wrote {drawn} worst-case figures and players, and an index, to {out}")
    return 0


REPLAY_INDEX_CSS = """
body{background:#111;color:#ddd;font:14px/1.5 ui-monospace,Menlo,monospace;margin:24px}
h1{font-size:18px}h2{font-size:15px;color:#9ad;font-weight:600;margin:22px 0 6px}
.note{color:#888;margin:6px 0 18px;max-width:70em}
.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px}
.card{background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:10px}
.card img{width:100%;background:#000;border-radius:4px;display:block}
.meta{font-size:12px;color:#aaa;margin-top:8px}
.meta b{color:#eee}
.bad{color:#ff8080}.good{color:#80ffb0}
a{color:#7cc}
"""


def _write_replay_index(out, cards, airframe, suite, n, seed):
    """One page linking every replay, with the case that produced it.

    A pile of `worst-07-final-miss73m-3d.html` files is not browsable, and the point of
    drawing a replay is to be able to look at several side by side and see what they have in
    common -- which is the only way a benchmark's failure list turns into a diagnosis.
    """
    parts = [f"<!doctype html><meta charset=utf-8><title>replays</title><style>{REPLAY_INDEX_CSS}</style>",
             "<h1>Worst accepted passes</h1>",
             f"<div class=note>{len(cards)} replays from <b>{airframe}</b>, suite "
             f"<b>{suite}</b>, {n} cases, seed {seed}. These are the cases the law "
             f"<b>accepted</b> and then missed &mdash; not the ones it refused, which is the "
             f"opposite failure and is counted separately in the run output.<br>"
             f"Each player is self-contained: orbit, zoom, play/pause, scrub, speed, a "
             f"top-down minimap and the gate crossing marked on the timeline.</div>",
             "<div class=cards>"]
    for stem, r in cards:
        s = r.get("score")
        parts.append(
            f"<div class=card><a href='{stem}-3d.html'>"
            f"<img src='{stem}.png' alt='{stem}'></a>"
            f"<div class=meta><b>{stem}</b><br>"
            f"miss <b class=bad>{r['miss']:.2f} m</b> &middot; failed on <b>{r['why']}</b><br>"
            f"range {r['dist']:.0f} m &middot; offset {r['off']:+.0f} m "
            f"({100.0 * abs(r['off']) / max(1.0, r['dist']):.1f}% of range)<br>"
            f"pass {r['speed']:.1f} m/s"
            + (f" &middot; demanded fp {r.get('fp', float('nan')):+.1f}&deg; "
               if r["family"] in ("ground", "final") else " ")
            + f"&middot; target {r.get('motion', 0.0):.2f} m/s"
            + (f"<br>track {s.angle_err_deg:+.1f}&deg; &middot; speed {s.speed_err_mps:+.2f} m/s"
               if s is not None else "")
            + "</div></div>")
    parts.append("</div>")
    (out / "index.html").write_text("".join(parts))


if __name__ == "__main__":
    raise SystemExit(main())