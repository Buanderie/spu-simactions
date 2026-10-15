#!/usr/bin/env python3
"""Run the whole suite and write the whole picture: per-pass figures, a comparison, a
lead sweep, and one HTML index with all of it.

This is the command to run to actually SEE the results rather than read a table.

    python3 scripts/report.py --airframe quad
    python3 scripts/report.py --airframe plane --gif
"""
import argparse
import datetime
import html
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import viz  # noqa: E402
from flyby import geo, scenarios  # noqa: E402
from flyby.experiment import run_offline  # noqa: E402
from flyby.guidance import GuidanceConfig, PlantConfig  # noqa: E402

STYLE = """
:root { color-scheme: dark; }
body { background:#12151c; color:#e8eaf0; font-family:-apple-system,Segoe UI,Roboto,sans-serif;
       margin:0 auto; max-width:1340px; padding:28px 22px 80px; line-height:1.5; }
h1 { font-size:24px; margin:0 0 4px; }
h2 { font-size:17px; margin:34px 0 10px; padding-bottom:6px; border-bottom:1px solid #2a3040; }
.sub { color:#8b93a7; font-size:13px; margin-bottom:22px; }
table { border-collapse:collapse; font-size:12.5px; margin:12px 0 22px; width:100%; }
th,td { padding:5px 9px; text-align:right; border-bottom:1px solid #222836; }
th:first-child,td:first-child { text-align:left; }
th { color:#8b93a7; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.4px; }
td.good { color:#57cc99; } td.warn { color:#f8961e; } td.bad { color:#f94144; }
tr { color:#e8eaf0; }
figure { margin:0 0 26px; }
img { width:100%; border-radius:7px; border:1px solid #2a3040; display:block; }
figcaption { color:#8b93a7; font-size:12px; margin-top:7px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
code,pre { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
pre { background:#181c25; border:1px solid #2a3040; border-radius:7px; padding:12px 14px;
      font-size:12px; overflow-x:auto; color:#c9d1e0; }
.tldr { background:#181c25; border:1px solid #2a3040; border-left:3px solid #4cc9f0;
        border-radius:7px; padding:13px 17px; margin:18px 0 8px; font-size:13.5px; }
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--airframe", choices=("plane", "quad"), default="quad")
    ap.add_argument("--suite", choices=("level", "ground"), default="level",
                    help="level: pass through a point at altitude; "
                         "ground: terminal descent onto a point on the ground")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gif", action="store_true", help="also write an animated GIF per pass")
    ap.add_argument("--only", default=None)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--lead-max", type=float, default=2.0)
    args = ap.parse_args()

    out = pathlib.Path(args.out or (ROOT / "runs" / f"offline-{args.airframe}"))
    out.mkdir(parents=True, exist_ok=True)

    origin = geo.Origin()
    gcfg = GuidanceConfig.for_quad() if args.airframe == "quad" else GuidanceConfig()
    pcfg = PlantConfig.for_quad() if args.airframe == "quad" else PlantConfig()
    build = (scenarios.build_ground_suite if args.suite == "ground"
             else scenarios.build_suite)
    suite = build(origin, args.airframe)
    if args.only:
        suite = [s for s in suite if args.only in s.name]

    rows, cards, results = [], [], {}
    for sc in suite:
        print(f"  {sc.name} ...", end="", flush=True)
        samples, scores, _ = run_offline(sc, gcfg, pcfg, dt=args.dt)
        s = scores[0] if scores else None
        rows.append((sc.name, s))
        results[sc.name] = s.as_dict() if s else None

        stem = f"{args.airframe}-{sc.name}"
        fig = viz.plot_pass(samples, sc.spec, str(out / f"{stem}.png"), scores,
                            title=f"{sc.name} — {sc.notes}", origin=origin)
        print(" ok" if fig else " (no samples)")
        if fig:
            cap = (f"<b>{html.escape(sc.name)}</b> — {html.escape(sc.notes)}"
                   + (f"<br><code>{html.escape(s.one_line())}</code>" if s else
                      "<br><b>no crossing of the gate</b>"))
            extra = ""
            if args.gif:
                g = viz.animate_pass(samples, sc.spec, str(out / f"{stem}.gif"), scores, origin=origin)
                if g:
                    extra = f'<figure><img src="{pathlib.Path(g).name}"><figcaption>animated: '\
                            f'the pass, coloured by time</figcaption></figure>'
            p3 = viz.write_3d_player(samples, sc.spec, scores, str(out / f"{stem}-3d.html"),
                                     notes=sc.notes, origin=origin)
            if p3:
                extra += (f'<p><a href="{pathlib.Path(p3).name}" target="_blank">'
                          f'▶ 3-D playback of this pass</a> '
                          f'<span class="dim">— orbit with the mouse, play/scrub the flight, '
                          f'vertical exaggeration on the slider</span></p>')
            cards.append((stem, cap, extra))

    # comparison bar chart over the whole suite
    viz.plot_suite(rows, str(out / "suite.png"),
                   title=f"scenario suite — {args.airframe}, reference plant")

    # tau_lead sweep
    leads, sweep_rows = [], []
    n = int(round(args.lead_max / 0.2)) + 1
    for i in range(n):
        lead = round(i * 0.2, 3)
        g = GuidanceConfig(**{**gcfg.__dict__, "tau_lead_s": lead})
        fly, miss = [], []
        for sc in build(origin, args.airframe):
            _, sco, _ = run_offline(sc, g, pcfg, dt=args.dt)
            if not sco:
                miss.append(999.0)
                continue
            miss.append(sco[0].miss_2d_m)
            if sco[0].feasible:
                fly.append(sco[0])
        leads.append(lead)
        sweep_rows.append(None if not fly else dict(
            mean_miss=sum(x.miss_2d_m for x in fly) / len(fly), worst_miss=max(miss),
            mean_trk=sum(abs(x.angle_err_deg) for x in fly) / len(fly),
            worst_trk=max(abs(x.angle_err_deg) for x in fly),
            mean_spd=sum(abs(x.speed_err_mps) for x in fly) / len(fly),
            mean_alt=sum(abs(x.alt_err_m) for x in fly) / len(fly)))
    viz.plot_lead_sweep(leads, sweep_rows, str(out / "lead-sweep.png"), args.airframe)

    (out / "scores.json").write_text(json.dumps(
        dict(airframe=args.airframe, tau_lead_s=gcfg.tau_lead_s,
             generated=datetime.datetime.now().isoformat(timespec="seconds"),
             results=results), indent=2))

    ok = [s for _n, s in rows if s and s.feasible]
    parts = []
    parts.append(f"<h1>fly-through-a-point — {args.airframe} @ reference plant</h1>")
    parts.append(f"<div class='sub'>generated {datetime.datetime.now():%Y-%m-%d %H:%M} · "
                 f"tau_lead {gcfg.tau_lead_s:.2f} s · plant lag "
                 f"{pcfg.tau_horizontal_s:.2f} s · bank limit {gcfg.max_bank_deg:.0f}° · "
                 f"<a href='scores.json'>scores.json</a></div>")
    if ok:
        parts.append(
            f"<div class='tldr'><b>{len(ok)} of {len(rows)} passes the law reported as "
            f"flyable.</b> Over those: mean miss "
            f"<b>{sum(s.miss_2d_m for s in ok)/len(ok):.2f} m</b>, mean angle error "
            f"<b>{sum(abs(s.angle_err_deg) for s in ok)/len(ok):.2f}°</b>, mean speed error "
            f"<b>{sum(abs(s.speed_err_mps) for s in ok)/len(ok):.3f} m/s</b>. The remaining "
            f"scenarios are ones the airframe cannot fly at all — they are kept, and marked, "
            f"because a gate that silently rounds off an impossible requirement is worse "
            f"than one that admits it.</div>")
    parts.append("<h2>all scenarios</h2>")
    parts.append("<img src='suite.png'>")
    parts.append("<table><tr><th>scenario</th><th>t cross</th><th>miss 2D</th><th>miss 3D</th>"
                 "<th>angle err</th><th>speed err</th><th>alt err</th><th>fp err</th>"
                 "<th>law says</th></tr>")
    for name, s in rows:
        if s is None:
            parts.append(f"<tr><td>{name}</td><td colspan='8' class='bad'>no crossing</td></tr>")
            continue
        def cls(v, tol, tol3=None):
            return "good" if abs(v) <= tol else ("warn" if abs(v) <= (tol3 or 3 * tol) else "bad")
        fp = f"{s.fp_angle_err_deg:+.2f}°" if s.fp_angle_err_deg is not None else "—"
        fpc = cls(s.fp_angle_err_deg, 3.0) if s.fp_angle_err_deg is not None else ""
        parts.append(
            f"<tr><td>{name}</td><td>{s.t_cross:.1f}s</td>"
            f"<td class='{cls(s.miss_2d_m, 5.0)}'>{s.miss_2d_m:.2f} m</td>"
            f"<td class='{cls(s.miss_3d_m, 5.0)}'>{s.miss_3d_m:.2f} m</td>"
            f"<td class='{cls(s.angle_err_deg, 5.0)}'>{s.angle_err_deg:+.2f}°</td>"
            f"<td class='{cls(s.speed_err_mps, 1.0)}'>{s.speed_err_mps:+.3f} m/s</td>"
            f"<td class='{cls(s.alt_err_m, 5.0)}'>{s.alt_err_m:+.1f} m</td>"
            f"<td class='{fpc}'>{fp}</td>"
            f"<td class='{'good' if s.feasible else 'bad'}'>"
            f"{'flyable' if s.feasible else 'INFEASIBLE'}</td></tr>")
    parts.append("</table>")
    parts.append("<h2>tau_lead — the one number tuned by search</h2>")
    parts.append("<img src='lead-sweep.png'>")
    parts.append("<div class='sub'>Mean over the scenarios the law reports as flyable; the "
                 "worst case is over all of them. The dip is the lead compensating the "
                 "vehicle's own response lag, and its depth is a direct read of how much "
                 "lag there is: flat here means the profile is not fighting the plant.</div>")
    parts.append("<h2>one figure per pass</h2>")
    parts.append("<div class='sub'>Each has a matching <b>3-D playback</b> — a single "
                 "self-contained HTML file with orbit, playback and a scrub slider. Open one "
                 "in a browser; nothing is fetched from the network.</div>")
    for stem, cap, extra in cards:
        parts.append(f"<figure><img src='{stem}.png'><figcaption>{cap}</figcaption></figure>")
        if extra:
            parts.append(extra)
    (out / "index.html").write_text(
        f"<!doctype html><meta charset=utf-8><style>{STYLE}</style>" + "".join(parts))
    print(f"\nwrote {out}/index.html  ({len(cards)} pass figures)")


if __name__ == "__main__":
    main()
