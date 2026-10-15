#!/usr/bin/env python3
"""Offline sweep: run the whole scenario suite against the reference plant and tabulate.

This is where the law gets tuned. It is the same law, the same scorer and the same
scenarios the live rig uses -- only the plant is the first-order lag in
`guidance.PlantConfig` instead of PX4 -- so a number that looks good here is a claim
about the guidance, not about the plant, and the live run is the check on that claim.

    python3 scripts/offline_sweep.py
    python3 scripts/offline_sweep.py --airframe quad --lead 1.0
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from flyby import geo, scenarios
from flyby.experiment import run_offline
from flyby.guidance import GuidanceConfig, PlantConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--airframe", choices=("plane", "quad"), default="plane")
    ap.add_argument("--lead", type=float, default=None, help="tau_lead_s override")
    ap.add_argument("--only", default=None, help="substring filter on scenario name")
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--csv", default=None, help="also write the table here")
    ap.add_argument("--suite", choices=("level", "ground"), default="level",
                    help="level = fly-through a point in the air (the tangent case); "
                         "ground = descend onto a ground point at an angle, as slowly as "
                         "the airframe allows (the case this repo was opened for)")
    args = ap.parse_args()

    origin = geo.Origin()
    gcfg = GuidanceConfig.for_quad() if args.airframe == "quad" else GuidanceConfig()
    pcfg = PlantConfig.for_quad() if args.airframe == "quad" else PlantConfig()
    if args.lead is not None:
        gcfg.tau_lead_s = args.lead

    suite = (scenarios.build_ground_suite(origin, args.airframe) if args.suite == "ground"
             else scenarios.build_suite(origin, args.airframe))
    if args.only:
        suite = [s for s in suite if args.only in s.name]

    rows = []
    hdr = (f"{'scenario':19s} {'t_cross':>8s} {'miss2d':>8s} {'miss3d':>8s} "
           f"{'trk_err':>9s} {'spd_err':>9s} {'dAlt':>8s} {'fp_err':>8s} {'ok':>4s}  notes")
    print(f"=== offline sweep ({args.airframe}, tau_lead={gcfg.tau_lead_s:.2f}s, "
          f"plant tau={pcfg.tau_horizontal_s:.2f}s) ===")
    print(hdr)
    print("-" * len(hdr))
    for sc in suite:
        samples, scores, _ = run_offline(sc, gcfg, pcfg, dt=args.dt)
        if not scores:
            print(f"{sc.name:19s} {'--':>8s}  NO CROSSING  ({sc.notes})")
            rows.append((sc.name, None))
            continue
        s = scores[0]
        fp = f"{s.fp_angle_err_deg:+8.2f}" if s.fp_angle_err_deg is not None else f"{'-':>8s}"
        ok = "ok" if s.feasible else "NO"
        print(f"{sc.name:19s} {s.t_cross:8.1f} {s.miss_2d_m:8.2f} {s.miss_3d_m:8.2f} "
              f"{s.angle_err_deg:+9.2f} {s.speed_err_mps:+9.2f} {s.alt_err_m:+8.1f} {fp} {ok:>4s}  {sc.notes}")
        rows.append((sc.name, s))

    ok = [r[1] for r in rows if r[1] is not None]
    if ok:
        print("-" * len(hdr))
        n = len(ok)
        print(f"{'MEAN':19s} {'':8s} {sum(s.miss_2d_m for s in ok)/n:8.2f} "
              f"{sum(s.miss_3d_m for s in ok)/n:8.2f} "
              f"{sum(abs(s.angle_err_deg) for s in ok)/n:9.2f} "
              f"{sum(abs(s.speed_err_mps) for s in ok)/n:9.2f} "
              f"{sum(abs(s.alt_err_m) for s in ok)/n:8.1f}")
        print(f"{'WORST':19s} {'':8s} {max(s.miss_2d_m for s in ok):8.2f} "
              f"{max(s.miss_3d_m for s in ok):8.2f} "
              f"{max(abs(s.angle_err_deg) for s in ok):9.2f} "
              f"{max(abs(s.speed_err_mps) for s in ok):9.2f} "
              f"{max(abs(s.alt_err_m) for s in ok):8.1f}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scenario", "t_cross", "miss2d_m", "miss3d_m", "trk_deg", "trk_err_deg",
                        "gs_mps", "spd_err_mps", "alt_err_m", "fp_err_deg"])
            for name, s in rows:
                if s is None:
                    w.writerow([name, "", "", "", "", "", "", "", "", ""])
                else:
                    w.writerow([name, f"{s.t_cross:.2f}", f"{s.miss_2d_m:.3f}", f"{s.miss_3d_m:.3f}",
                                f"{s.got_angle_deg:.3f}", f"{s.angle_err_deg:.3f}",
                                f"{s.got_speed_mps:.3f}", f"{s.speed_err_mps:.3f}",
                                f"{s.alt_err_m:.3f}",
                                "" if s.fp_angle_err_deg is None else f"{s.fp_angle_err_deg:.3f}"])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
