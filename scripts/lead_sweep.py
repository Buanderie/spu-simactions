#!/usr/bin/env python3
"""Sweep tau_lead_s over the scenario suite: the one number worth tuning by search.

`tau_lead_s` is how far ahead of the terminal profile the law reads it, and it should
match the vehicle's response lag. Guessing it is how the far-offset and moving-point
cases go wrong, so it is measured instead: for each candidate, run the whole suite and
report the mean and worst case.

The interesting column is `feasible` -- how many scenarios the law itself had to admit
it could not fly. Those are the ones whose error is physics, not tuning, and a tuning
sweep that "improves" the mean by making the law plan more aggressively just moves the
error from guidance into manoeuvring.

    python3 scripts/lead_sweep.py
    python3 scripts/lead_sweep.py --airframe quad --max 1.5
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from flyby import geo, scenarios
from flyby.experiment import run_offline
from flyby.guidance import GuidanceConfig, PlantConfig


def evaluate(suite, gcfg, pcfg, dt):
    """Score the suite, separating the cases the law itself says it cannot fly.

    Averaging those in would be measuring the airframe, not the guidance: a scenario the
    law reports infeasible has an error that no value of tau_lead can fix, and it drags
    the mean around enough to hide a real regression. So the mean is over the flyable
    cases, and the infeasible ones are counted and listed separately.
    """
    flyable, infeasible, misses = [], [], []
    for sc in suite:
        samples, scores, _ = run_offline(sc, gcfg, pcfg, dt=dt)
        if not scores:
            infeasible.append((sc.name, "no crossing"))
            misses.append(999.0)
            continue
        s = scores[0]
        misses.append(s.miss_2d_m)
        if s.feasible:
            flyable.append(s)
        else:
            infeasible.append((sc.name, f"{s.miss_2d_m:.1f} m"))
    if not flyable:
        return None
    return dict(
        mean_miss=sum(s.miss_2d_m for s in flyable) / len(flyable),
        worst_miss=max(misses),
        mean_trk=sum(abs(s.angle_err_deg) for s in flyable) / len(flyable),
        worst_trk=max(abs(s.angle_err_deg) for s in flyable),
        mean_spd=sum(abs(s.speed_err_mps) for s in flyable) / len(flyable),
        mean_alt=sum(abs(s.alt_err_m) for s in flyable) / len(flyable),
        infeasible=infeasible,
        n=len(flyable),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--airframe", choices=("plane", "quad"), default="plane")
    ap.add_argument("--min", type=float, default=0.0)
    ap.add_argument("--max", type=float, default=2.5)
    ap.add_argument("--step", type=float, default=0.1)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--plot", default=None, help="save a PNG of the sweep here")
    args = ap.parse_args()

    suite = scenarios.build_suite(geo.Origin(), args.airframe)
    base = GuidanceConfig.for_quad() if args.airframe == "quad" else GuidanceConfig()
    pcfg = PlantConfig.for_quad() if args.airframe == "quad" else PlantConfig()

    leads, rows = [], []
    n = int(round((args.max - args.min) / args.step)) + 1
    print(f"=== tau_lead sweep ({args.airframe}, plant tau={pcfg.tau_horizontal_s:.2f}s, "
          f"max bank {base.max_bank_deg:.0f} deg) ===")
    print(f"{'tau_lead':>9} {'mean_miss':>10} {'worst_miss':>11} {'mean_trk':>9} "
          f"{'worst_trk':>10} {'mean_spd':>9} {'mean_alt':>9} {'n':>3s}  infeasible")
    for i in range(n):
        lead = args.min + i * args.step
        g = GuidanceConfig(**{**base.__dict__, "tau_lead_s": lead})
        r = evaluate(suite, g, pcfg, args.dt)
        leads.append(lead)
        rows.append(r)
        if r is None:
            print(f"{lead:9.2f}  (no crossings at all)")
            continue
        print(f"{lead:9.2f} {r['mean_miss']:10.2f} {r['worst_miss']:11.2f} "
              f"{r['mean_trk']:9.2f} {r['worst_trk']:10.2f} {r['mean_spd']:9.2f} "
              f"{r['mean_alt']:9.2f} {r['n']:3d}  " + ", ".join(n for n, _ in r["infeasible"]))

    good = [(l, r) for l, r in zip(leads, rows) if r]
    if good:
        best = min(good, key=lambda x: x[1]["mean_miss"])
        print(f"\nbest mean miss at tau_lead = {best[0]:.2f} s "
              f"({best[1]['mean_miss']:.2f} m mean, {best[1]['worst_miss']:.2f} m worst)")

    if args.plot:
        import viz
        viz.plot_lead_sweep(leads, rows, args.plot, args.airframe)


if __name__ == "__main__":
    main()
