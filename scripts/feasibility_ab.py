#!/usr/bin/env python3
"""Do the two infeasibility tests actually PREDICT a miss, or only look like they do?

Both tests answer "is this pass achievable" and both say `infeasible` a lot, which is
worthless on its own: a test that refuses everything is a test that refuses nothing. The
only question that separates a good feasibility test from a plausible one is whether its
verdict tracks the outcome.

    python3 scripts/feasibility_ab.py --n 400 --suite final

For each sampled case this records three things:

    the OUTCOME          did it actually fly the pass (gate tolerance)?
    the GEOMETRIC test   our existing floor, V >= |v_target . n| / sin(sigma)
    the ZEV test         de Paiva et al.'s quartic root-existence condition

and cross-tabulates them. Read the table as: of the passes that really did miss, how many
did each test SEE? A test with high recall and poor precision is refusing good passes, which
costs capability; a test with poor recall is letting bad ones through, which costs accuracy.
The number to compare is `misses caught / misses` (recall) against `refusals that were
actually fine` (the false-refusal rate).

The ZEV verdict is read EARLY in the approach, and that detail is not cosmetic. The test is
a statement about the problem from the CURRENT state, and at the gate the current state IS
the terminal state: dr -> 0 makes a2 -> 0, the condition `a2 < 0` fails, and every approach
-- including perfect ones -- reports infeasible at its own gate. Reading it where the
question is meaningful is the difference between measuring it and misusing it.
"""
import argparse
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flyby import airframes  # noqa: E402
from flyby.experiment import run_offline  # noqa: E402
from scripts.bench import (TOLERANCES, sample_final, sample_final_moving,  # noqa: E402
                           sample_ground, sample_ground_moving, sample_level, verdict)

# Where in the approach to read a feasibility verdict. 0.5 is halfway in TIME remaining,
# which is the honest place to ask "is this approach still achievable" -- early enough that
# the answer can still change the plan, late enough that the geometry has settled.
READ_AT_FRAC = 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--airframe", default="iris")
    ap.add_argument("--suite", choices=("ground", "final", "level"), default="final")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--moving-frac", type=float, default=0.35)
    ap.add_argument("--dt", type=float, default=0.05)
    args = ap.parse_args()

    cap = airframes.get(args.airframe)
    gcfg, pcfg = cap.guidance_config(), cap.plant_config()
    print(f"=== feasibility test A/B: {args.airframe}, {args.suite}, {args.n} cases "
          f"(seed {args.seed}) ===")
    print(f"  outcome = passed the {TOLERANCES[1][0]} tolerance "
          f"({TOLERANCES[1][1]} m / {TOLERANCES[1][2]} deg / {TOLERANCES[1][3]} m/s)")
    print(f"  ZEV verdict read at {READ_AT_FRAC:.0%} of time-to-go remaining\n")

    rng = random.Random(args.seed)
    rows = []
    for _ in range(args.n):
        if args.suite == "level":
            sc, meta = sample_level(rng, cap, {})
        elif args.suite == "final":
            sc, meta = (sample_final_moving(rng, cap, {}) if rng.random() < args.moving_frac
                        else sample_final(rng, cap, {}))
        elif rng.random() < args.moving_frac:
            sc, meta = sample_ground_moving(rng, cap, {})
        else:
            sc, meta = sample_ground(rng, cap, {})
        try:
            samples, scores, _ = run_offline(sc, gcfg, pcfg, dt=args.dt)
        except Exception:
            continue
        if not scores:
            rows.append(dict(outcome=None, geom=None, zev=None, motion=meta["motion"]))
            continue
        s = scores[0]
        passed, _ = verdict(s, args.suite, TOLERANCES[1])
        # The geometric floor is a property of the TARGET'S MOTION and does not vary along
        # the approach, so reading it at the gate is fine. The ZEV test is not.
        geom = s.feasible
        zev = read_zev(samples, s.t_cross, pcfg, sc.spec)
        rows.append(dict(outcome=bool(passed), geom=bool(geom), zev=zev,
                         motion=meta["motion"], miss=s.miss_2d_m,
                         declared_feasible=s.feasible))

    tabulate(rows, args.suite)
    return 0


def read_zev(samples, t_cross, plant, spec):
    """The ZEV verdict at a fixed fraction of the time remaining, or None if unavailable."""
    # The approach runs from the first sample to the gate. Halfway in TIME REMAINING is not
    # halfway in samples -- the aircraft covers ground at varying speed -- so interpolate on
    # the time axis, which is monotonic, rather than assuming a constant rate.
    t0, t1 = samples[0].t, t_cross
    t_read = t1 - (1.0 - READ_AT_FRAC) * (t1 - t0)
    best = None
    for smp in samples:
        if smp.plan is not None and smp.t <= t_read:
            best = smp
        else:
            break
    if best is None or best.plan is None:
        return None
    return bool(best.plan.zev_feasible)


def tabulate(rows, suite):
    scored = [r for r in rows if r["outcome"] is not None and r["zev"] is not None]
    n_miss = sum(1 for r in scored if not r["outcome"])
    print(f"  cases with a usable reading: {len(scored)}   of which really missed: {n_miss}")
    if not scored or n_miss == 0:
        print("  (nothing missed, so no test can be caught being wrong)")
        return
    print(f"\n  {'test':22s} {'refused':>8s} {'caught misses':>13s} {'false refusals':>15s}")
    print("  " + "-" * 62)
    for name, key in (("current (geometric)", "geom"), ("ZEV quartic", "zev")):
        ref = [r for r in scored if not r[key]]
        caught = [r for r in ref if not r["outcome"]]
        false = [r for r in ref if r["outcome"]]
        print(f"  {name:22s} {len(ref):8d} {len(caught):7d} ({100.0 * len(caught) / n_miss:4.0f}%)"
              f" {len(false):15d} ({100.0 * len(false) / max(1, len(ref)):4.0f}% of its refusals)")
    print("\n  A test that catches most of the misses while refusing few passes that would "
          "have\n  flown is worth having. A test that refuses everything catches every miss "
          "and is\n  worth nothing -- so read the two columns together, not the first alone.")

    mov = [r for r in scored if r["motion"] > 0.5]
    if mov:
        nm = sum(1 for r in mov if not r["outcome"])
        print(f"\n  moving point only ({len(mov)} cases, {nm} really missed):")
        for name, key in (("current (geometric)", "geom"), ("ZEV quartic", "zev")):
            ref = [r for r in mov if not r[key]]
            caught = [r for r in ref if not r["outcome"]]
            if nm:
                print(f"    {name:22s} refused {len(ref):4d}   caught "
                      f"{len(caught):4d}/{nm} ({100.0 * len(caught) / nm:4.0f}%)")


if __name__ == "__main__":
    raise SystemExit(main())