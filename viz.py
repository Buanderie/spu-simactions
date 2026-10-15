"""Plots. The point of a measurement you cannot see is that you cannot tell a good pass
from a lucky one.

Every figure here answers a question a table cannot:

  * **top-down** -- did the track actually go *through* the point, or curve near it? A
    table of miss distances cannot distinguish a 3 m miss on the right heading from a
    3 m miss on the wrong one, and cannot show that the approach was a spiral.
  * **s(t) / e(t)** -- the pass frame over time. This is where the shape of the guidance
    is visible: an exponential approach, a minimum-jerk hump, a corner, an oscillation.
  * **angle and speed error vs the requirement** -- plotted as *errors*, not raw values,
    so the required value is a flat zero line and the wrap at 360 deg cannot hide an
    error. A plot of raw track angle would show a pass at 030 as "359 -> 1" and a reader
    would have to notice that is a 2 deg error and not a 358 deg one.
  * **altitude** -- the third axis, which is where a "2D-perfect" pass usually is not.

North is up and the axes are equal-scaled, because a map that is stretched is a map that
lies about the geometry.
"""
import html
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow

try:
    from . import geo
    from .score import closest_approach
    from .guidance import decompose
except ImportError:  # running as a top-level script (scripts/report.py) rather than a package
    from flyby import geo
    from flyby.score import closest_approach
    from flyby.guidance import decompose

BG = "#12151c"
FG = "#e8eaf0"
GRID = "#2a3040"
ACCENT = "#4cc9f0"
WARN = "#f8961e"
BAD = "#f94144"
GOOD = "#57cc99"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "axes.edgecolor": GRID,
    "grid.color": GRID,
    "font.size": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 130,
})


def _style(ax, title=None, xlabel=None, ylabel=None, grid=True):
    if title:
        ax.set_title(title, loc="left", color=FG, pad=4)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.35, linewidth=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)
    return ax


def _series(samples, fn):
    return (np.array([s.t for s in samples]), np.array([fn(s) for s in samples], dtype=float))


# ------------------------------------------------------------------ the main figure

def plot_pass(samples, spec, path, scores=None, title=None, show_plan=True, origin=None):
    """The full picture of one pass: map on the left, the four errors on the right."""
    origin = origin or geo.Origin()
    if not samples:
        return None
    sc = scores[0] if scores else None

    fig = plt.figure(figsize=(13.0, 7.4))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.45, 1.0], hspace=0.55, wspace=0.22,
                          left=0.055, right=0.975, top=0.855, bottom=0.075)

    # ---- top-down map -------------------------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    _style(ax, "ground track (north up, equal scale)")

    tn = np.array([s.target_ned for s in samples], dtype=float)
    pn = np.array([s.ned for s in samples], dtype=float)
    px, py = pn[:, 0], pn[:, 1]
    tx, ty = tn[:, 0], tn[:, 1]

    pad = 0.25
    x_lo, x_hi = min(px.min(), tx.min()), max(px.max(), tx.max())
    y_lo, y_hi = min(py.min(), ty.min()), max(py.max(), ty.max())
    cx, cy = (x_lo + x_hi) / 2, (y_lo + y_hi) / 2
    half = max(x_hi - x_lo, y_hi - y_lo) / 2 * (1 + pad)
    x_lo, x_hi, y_lo, y_hi = cx - half, cx + half, cy - half, cy + half

    # the pass line, both ways through the point
    d = geo.track_unit(spec.angle_deg)
    reach = half * 1.5
    ax.plot([tx.mean() - d[0] * reach, tx.mean() + d[0] * reach],
            [ty.mean() - d[1] * reach, ty.mean() + d[1] * reach],
            "--", color=WARN, lw=1.1, alpha=0.85, zorder=2)
    ax.annotate("", xy=(tx.mean() + d[0] * half * 0.42, ty.mean() + d[1] * half * 0.42),
                xytext=(tx.mean(), ty.mean()),
                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.1))
    ax.text(tx.mean() + d[0] * half * 0.45, ty.mean() + d[1] * half * 0.45,
            f"required track {spec.angle_deg:.0f}°", color=WARN, fontsize=7.5,
            va="bottom", ha="left")

    # the point's own track, if it has one
    if math.hypot(tn[0][0] - tn[-1][0], tn[0][1] - tn[-1][1]) > 1.0:
        ax.plot(tx, ty, "-", color="#9b8cff", lw=1.3, alpha=0.9, zorder=3,
                label="point track")
        ax.plot(tx[0], ty[0], "o", color="#9b8cff", ms=3.5, zorder=4)
        ax.text(tx[-1], ty[-1], "  point now", color="#9b8cff", fontsize=7, va="center")

    # our track, coloured by time
    segs = [(px[:-1], py[:-1], px[1:], py[1:]) for _ in (0,)]
    for i in range(len(px) - 1):
        c = plt.cm.viridis(i / max(1, len(px) - 2))
        ax.plot(px[i:i + 2], py[i:i + 2], "-", color=c, lw=2.0, solid_capstyle="round", zorder=5)
    ax.plot(px[-1], py[-1], "o", color=ACCENT, ms=6, zorder=7,
            markeredgecolor=BG, markeredgewidth=1.2)
    ax.text(px[-1], py[-1], "  now", color=ACCENT, fontsize=7.5, va="center")
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=samples[0].t, vmax=samples[-1].t))
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("sim time (s)", color=FG, fontsize=7)
    cb.ax.tick_params(colors=FG, labelsize=7)
    cb.outline.set_edgecolor(GRID)

    ax.plot([tx[0]], [ty[0]], "*", color=BAD, ms=15, zorder=8,
            markeredgecolor=BG, markeredgewidth=0.8)
    ax.text(tx[0], ty[0], "  P", color=BAD, fontsize=8, va="bottom")

    # the scored crossing: the miss vector, and the velocity there
    if sc is not None:
        cxp, cyp = sc.ned_at_cross[0], sc.ned_at_cross[1]
        ax.plot([cxp], [cyp], "o", color=GOOD, ms=8, zorder=9,
                markeredgecolor=BG, markeredgewidth=1.2)
        ax.plot([tx[0], cxp], [ty[0], cyp], "-", color=GOOD, lw=2.6, zorder=9, alpha=0.95)
        ax.annotate(f"miss {sc.miss_2d_m:.1f} m",
                    xy=((cxp + tx[0]) / 2, (cyp + ty[0]) / 2),
                    xytext=(6, -10), textcoords="offset points",
                    color=GOOD, fontsize=8.5, fontweight="bold")
        vx, vy = sc.vel_at_cross[0], sc.vel_at_cross[1]
        L = 0.16 * half
        ax.annotate("", xy=(cxp + vx * L, cyp + vy * L), xytext=(cxp, cyp),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.8))
        ax.text(cxp + vx * L * 1.08, cyp + vy * L * 1.08,
                f"{sc.got_speed_mps:.1f} m/s @ {sc.got_angle_deg:.0f}°",
                color=ACCENT, fontsize=7.5, va="center")

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")
    # a compass rose, because north-up only means something if you can see north
    ax.annotate("N", xy=(x_hi - half * 0.05, y_lo + half * 0.06), color=FG,
                fontsize=9, ha="center")
    ax.annotate("", xy=(x_hi - half * 0.05, y_lo + half * 0.155),
                xytext=(x_hi - half * 0.05, y_lo + half * 0.075),
                arrowprops=dict(arrowstyle="-|>", color=FG, lw=1.0))
    # the actual lat/lon of the point, so the picture is anchored to the geography
    ax.text(0.5, -0.085, f"point {spec.lat:.6f}, {spec.lon:.6f}  @ {spec.alt_m:.1f} m MSL"
                         f"     start {samples[0].ned[0]:.0f} N {samples[0].ned[1]:.0f} E"
                         f" {samples[0].ned[2]:.0f} m",
            transform=ax.transAxes, color="#8b93a7", fontsize=7, ha="center")

    # ---- error panels -------------------------------------------------------
    t = np.array([s.t for s in samples])
    axes = [fig.add_subplot(gs[i, 1]) for i in range(4)]

    def draw(ax, y, label, want=0.0, unit=""):
        y = np.asarray(y, dtype=float)
        ax.axhline(want, color=WARN, lw=1.0, ls="--", alpha=0.8)
        ax.plot(t, y, color=ACCENT, lw=1.2)
        ax.fill_between(t, np.full_like(y, want), y, where=np.isfinite(y),
                        color=ACCENT, alpha=0.08)
        _style(ax, label, "time (s)", unit)
        if sc is not None:
            ax.axvline(sc.t_cross, color=GOOD, lw=1.0, ls=":")
            ax.axvspan(0, sc.t_cross, color="#ffffff", alpha=0.025)

    s_along, e_cross = [], []
    for smp in samples:
        s_, e_, _, _ = decompose(
            (smp.ned[0] - smp.target_ned[0], smp.ned[1] - smp.target_ned[1]),
            (smp.vel[0] - smp.target_vel[0], smp.vel[1] - smp.target_vel[1]),
            spec.angle_deg)
        s_along.append(s_)
        e_cross.append(e_)

    draw(axes[0], s_along, "along-track error (positive = short of the gate)", 0.0, "m")
    draw(axes[1], e_cross, "cross-track error (positive = right of the line)", 0.0, "m")
    draw(axes[2], [math.hypot(s.vel[0], s.vel[1]) for s in samples],
         f"ground speed  (want {spec.speed_mps:.1f})", spec.speed_mps, "m/s")
    trk_err = [geo.angle_diff_deg(math.degrees(math.atan2(s.vel[1], s.vel[0])), spec.angle_deg)
               if math.hypot(s.vel[0], s.vel[1]) > 0.3 else np.nan for s in samples]
    draw(axes[3], trk_err, f"track angle error  (want {spec.angle_deg:.0f}°)", 0.0, "deg")

    # altitude gets its own inset-style panel, stacked onto the speed panel's row
    axes[2].plot([], [])
    fig.suptitle(title or _title(spec, samples, sc), x=0.055, ha="left", fontsize=11,
                 color=FG, fontweight="bold", y=0.985)

    _verdict(fig, sc, samples, spec)
    fig.savefig(path)
    plt.close(fig)
    return path


def _title(spec, samples, sc):
    base = (f"fly-through  {spec.angle_deg:.0f}° @ {spec.speed_mps:.0f} m/s"
            + (f"  (flight path {spec.fp_angle_deg:+.0f}°)" if spec.fp_angle_deg is not None else ""))
    if sc is None:
        return base + "   —   NO CROSSING"
    return base + f"   —   {spec.label or 'pass'}"


def _verdict(fig, sc, samples, spec):
    """The headline numbers, big, in the corner. A reader should not have to add up axes."""
    if sc is None:
        txt, col = "NO CROSSING OF THE GATE", BAD
    else:
        ca = closest_approach(samples)
        lines = [
            f"miss {sc.miss_2d_m:7.2f} m      angle {sc.angle_err_deg:+6.2f}°"
            f"      speed {sc.speed_err_mps:+5.2f} m/s",
            f"alt {sc.alt_err_m:+7.1f} m" + (f"      flight path {sc.fp_angle_err_deg:+6.2f}°"
                                             if sc.fp_angle_err_deg is not None else "")
            + f"      closest approach {ca[2]:.2f} m",
        ]
        txt = "\n".join(lines)
        worst = max(abs(sc.angle_err_deg), abs(sc.speed_err_mps))
        col = GOOD if (sc.miss_2d_m < 5 and abs(sc.angle_err_deg) < 5
                       and abs(sc.speed_err_mps) < 1.0) else (WARN if worst < 15 else BAD)
    fig.text(0.972, 0.925, txt, ha="right", va="top", color=col, fontsize=9.5,
             family="monospace", zorder=10,
             bbox=dict(facecolor=BG, edgecolor=col, boxstyle="round,pad=0.5", alpha=0.92))


# ------------------------------------------------------------------- altitude panel

def plot_altitude(samples, spec, path, scores=None, origin=None):
    """Altitude over the pass, with the point's altitude and the commanded climb rate."""
    t = np.array([s.t for s in samples])
    alt = np.array([-s.ned[2] + (origin.alt_m if origin else 0.0) for s in samples])
    tgt_alt = np.array([-s.target_ned[2] + (origin.alt_m if origin else 0.0) for s in samples])
    fig, ax = plt.subplots(figsize=(11.0, 4.0))
    _style(ax, "altitude profile", "time (s)", "altitude (m MSL)")
    ax.plot(t, alt, color=ACCENT, lw=1.6, label="aircraft")
    ax.plot(t, tgt_alt, color=BAD, lw=1.2, label="point")
    if scores:
        ax.axvline(scores[0].t_cross, color=GOOD, ls=":", lw=1.0)
        ax.plot([scores[0].t_cross], [scores[0].got_alt_m], "o", color=GOOD, ms=7)
        ax.annotate(f"{scores[0].alt_err_m:+.1f} m at the gate",
                    xy=(scores[0].t_cross, scores[0].got_alt_m), xytext=(8, 8),
                    textcoords="offset points", color=GOOD, fontsize=8)
    ax.legend(loc="upper left", framealpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------- suite comparison

def plot_suite(rows, path, title="scenario suite"):
    """One bar per scenario for each of the three gate numbers, side by side."""
    names = [r[0] for r in rows]
    miss = [r[1].miss_2d_m if r[1] else np.nan for r in rows]
    ang = [abs(r[1].angle_err_deg) if r[1] else np.nan for r in rows]
    spd = [abs(r[1].speed_err_mps) if r[1] else np.nan for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.2), sharex=True)
    for ax, vals, lab, unit, tol in (
            (axes[0], miss, "miss at the gate", "m", 5.0),
            (axes[1], ang, "track angle error", "deg", 5.0),
            (axes[2], spd, "ground speed error", "m/s", 1.0)):
        cols = [GOOD if (v == v and v <= tol) else (WARN if (v == v and v <= 3 * tol) else BAD)
                for v in vals]
        ax.bar(range(len(names)), vals, color=cols, width=0.62)
        ax.axhline(tol, color=WARN, ls="--", lw=0.9, alpha=0.7)
        ax.text(len(names) - 0.4, tol, f" tolerance {tol:g} {unit}", color=WARN,
                fontsize=7, va="bottom", ha="right")
        for i, v in enumerate(vals):
            if v == v:
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=FG)
            else:
                ax.text(i, 0.1, "no gate", ha="center", va="bottom", fontsize=6.5, color=BAD)
        _style(ax, lab, None, unit)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=20, ha="right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_lead_sweep(leads, rows, path, airframe="plane"):
    """How the one tunable number trades the mean against the worst case."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ok = [(l, r) for l, r in zip(leads, rows) if r]
    if not ok:
        return None
    L = [l for l, _ in ok]
    _style(ax, f"tau_lead sweep — {airframe}", "tau_lead (s)", "metres / degrees")
    ax.plot(L, [r["mean_miss"] for _, r in ok], "-o", color=ACCENT, lw=1.5, ms=4,
            label="mean miss (m)")
    ax.plot(L, [r["worst_miss"] for _, r in ok], "-o", color=BAD, lw=1.5, ms=4,
            label="worst miss (m)")
    ax2 = ax.twinx()
    ax2.plot(L, [r["mean_trk"] for _, r in ok], "--s", color=WARN, lw=1.2, ms=3.5,
             label="mean angle error (deg)")
    ax2.set_ylabel("degrees", color=FG)
    ax2.tick_params(colors=FG)
    ax2.spines["right"].set_color(GRID)
    ax2.axhline(5.0, color=WARN, ls=":", lw=0.8, alpha=0.6)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", framealpha=0.15)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


# ------------------------------------------------------------------- animation

def animate_pass(samples, spec, path, scores=None, fps=12, origin=None, trail_s=None):
    """A GIF of the pass. Useful for one specific question: does the approach actually
    look like an approach, or does it swing out and come back?"""
    import matplotlib.animation as animation
    from matplotlib import cm

    if not samples:
        return None
    sc = scores[0] if scores else None
    t_all = np.array([s.t for s in samples])
    px = np.array([s.ned[0] for s in samples])
    py = np.array([s.ned[1] for s in samples])
    tx = np.array([s.target_ned[0] for s in samples])
    ty = np.array([s.target_ned[1] for s in samples])

    half = 0.6 * max(px.max() - px.min(), py.max() - py.min(),
                     np.abs(tx[-1] - tx[0]), np.abs(ty[-1] - ty[0]), 200.0)
    cx, cy = (min(px.min(), tx.min()) + max(px.max(), tx.max())) / 2, \
             (min(py.min(), ty.min()) + max(py.max(), ty.max())) / 2

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    _style(ax, f"{spec.angle_deg:.0f}° @ {spec.speed_mps:.0f} m/s")

    d = geo.track_unit(spec.angle_deg)
    ax.plot([tx[0] - d[0] * half * 2, tx[0] + d[0] * half * 2],
            [ty[0] - d[1] * half * 2, ty[0] + d[1] * half * 2], "--", color=WARN, lw=1.0)
    trail, = ax.plot([], [], "-", color=ACCENT, lw=2.0)
    head, = ax.plot([], [], "o", color=ACCENT, ms=8, markeredgecolor=BG)
    pmark, = ax.plot([tx[0]], [ty[0]], "*", color=BAD, ms=16)
    ptrail, = ax.plot(tx, ty, "-", color="#9b8cff", lw=1.2, alpha=0.7)
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", family="monospace",
                  fontsize=9, color=FG)

    def frame(i):
        k = min(i, len(t_all) - 1)
        n = len(t_all)
        trail.set_data(px[:k + 1], py[:k + 1])
        head.set_data([px[k]], [py[k]])
        pmark.set_data([tx[k]], [ty[k]])
        done = "PASSED" if (sc and t_all[k] > sc.t_cross) else ""
        txt.set_text(f"t = {t_all[k]:6.1f} s\n"
                     f"miss now {math.hypot(px[k]-tx[k], py[k]-ty[k]):6.1f} m  {done}")
        return trail, head, pmark, txt

    anim = animation.FuncAnimation(fig, frame, frames=len(t_all),
                                   interval=1000.0 / fps, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


# --------------------------------------------------------------- 3-D playback
#
# matplotlib's mplot3d is broken in this environment and plotly is not installed, and
# neither would give a *playable* view anyway. So this is a single self-contained HTML
# file with a hand-rolled perspective renderer on a 2-D canvas: no dependencies, no CDN,
# opens from disk, works offline.
#
# What it has to get right, for a flight to be legible rather than decorative:
#
#   * **Vertical exaggeration.** A UAV pass is 2 km long and changes altitude by tens of
#     metres. At true scale it is a straight line lying on the floor and the third
#     dimension carries nothing. A slider from 1x to 20x, defaulting to something that
#     makes the climb-out visible, is the difference between "a line" and "a flight".
#   * **The gate, drawn as a surface.** The requirement is a LINE through a point, and a
#     line seen edge-on from most angles is invisible. It is drawn as a long thin
#     translucent corridor, which reads from any camera position.
#   * **Colour by time, with a playhead.** Static 3-D plots of trajectories are hard to
#     read because you cannot tell which way it went. Colour plus a bright head plus a
#     scrub slider makes the direction and the moment of interest obvious.
#   * **A synced top-down minimap**, because "did it go through the point" is a plan-view
#     question and the 3-D view alone will not answer it.
#   * **The numbers that are the score**, live: along-track and cross-track error against
#     zero, speed against the requirement, and the instant of the gate crossing marked on
#     the timeline.

_JS = r"""
const D = window.__FLYBY__;
const cv = document.getElementById('view'), cx2 = cv.getContext('2d');
const mm = document.getElementById('mini'), mx = mm.getContext('2d');
const TAU = Math.PI * 2;

let az = -0.9, el = 0.42, dist = 1.5, panX = 0, panY = 0, vex = 8.0;
let t = 0, playing = true, speed = 1.0, dirty = true;
const A = D.aircraft, P = D.point, T = D.target, SP = D.spec, SC = D.score;

const S = {
  minx:1e9,maxx:-1e9,miny:1e9,maxy:-1e9,minz:1e9,maxz:-1e9
};
for (const s of [A,T,P]) for (const p of s) {
  S.minx=Math.min(S.minx,p[0]); S.maxx=Math.max(S.maxx,p[0]);
  S.miny=Math.min(S.miny,p[1]); S.maxy=Math.max(S.maxy,p[1]);
  S.minz=Math.min(S.minz,p[2]); S.maxz=Math.max(S.maxz,p[2]);
}
const cx0=(S.minx+S.maxx)/2, cy0=(S.miny+S.maxy)/2, cz0=(S.minz+S.maxz)/2;
// The framing distance has to include the EXAGGERATED vertical span, or a steep dive at
// 8x is taller than the approach is long and the start of the flight sits off the top of
// the frame at the default zoom -- which looks like a rendering bug and is not one.
function spanNow(){ return Math.max(S.maxx-S.minx, S.maxy-S.miny, (S.maxz-S.minz)*vex, 1); }
function pt(i){ const p=A[i]; return [p[0],p[1],p[2]]; }

function cam(){
  const span = spanNow();
  const ce=Math.cos(el), se=Math.sin(el), ca=Math.cos(az), sa=Math.sin(az);
  // The camera lives in the same EXAGGERATED space as everything else. Leaving it in raw
  // metres while `proj` scales the world is a units mismatch, and it shows up as the
  // flight sitting at the bottom of the frame with the horizon through the middle -- the
  // camera is looking at 100 m while the aircraft is drawn at 800.
  const cz=cz0*vex;
  const eye=[cx0+panX + dist*span*ce*sa, cy0+panY + dist*span*ce*ca, cz + dist*span*se];
  let f=[cx0+panX-eye[0], cy0+panY-eye[1], cz-eye[2]];
  let fl=Math.hypot(f[0],f[1],f[2]); f=f.map(v=>v/fl);
  // right = normalize(cross(forward, worldUp)) with worldUp = (0, 0, 1)
  let r=[f[1], -f[0], 0];
  let rl=Math.hypot(r[0],r[1],r[2])||1; r=r.map(v=>v/rl);
  const u=[r[1]*f[2]-r[2]*f[1], r[2]*f[0]-r[0]*f[2], r[0]*f[1]-r[1]*f[0]];
  return {eye,f,r,u};
}
function proj(p, C, w, h){
  const q=[p[0]-C.eye[0], p[1]-C.eye[1], (p[2])*vex - C.eye[2]];
  const z = q[0]*C.f[0]+q[1]*C.f[1]+q[2]*C.f[2];
  if (z <= 1) return null;
  const x = q[0]*C.r[0]+q[1]*C.r[1]+q[2]*C.r[2];
  const y = q[0]*C.u[0]+q[1]*C.u[1]+q[2]*C.u[2];
  // The focal length is FIXED and the camera distance is what moves. Scaling the focal
  // length by `dist` as well -- the obvious thing -- cancels exactly: the scene then
  // projects to the same size at every zoom, the wheel changes perspective and nothing
  // else, and the framing cannot adapt when the vertical exaggeration grows the scene.
  const F = Math.min(w,h)*0.95;
  return [w/2 + x*F/z, h/2 - y*F/z, z];
}

function frame(){ requestAnimationFrame(frame); draw(); }

function draw(){
  cv.width = cv.clientWidth*devicePixelRatio; cv.height = cv.clientHeight*devicePixelRatio;
  mm.width = mm.clientWidth*devicePixelRatio; mm.height = mm.clientHeight*devicePixelRatio;
  const w=cv.width, h=cv.height, C=cam();

  // sky/ground wash so the horizon reads
  const g = cx2.createLinearGradient(0,0,0,h);
  g.addColorStop(0,'#0d1017'); g.addColorStop(1,'#181d26');
  cx2.fillStyle=g; cx2.fillRect(0,0,w,h);

  // ---- ground grid at z = 0, 250 m spacing, north-up in world terms
  cx2.lineWidth=1;
  const g0=Math.floor((S.minx-200)/250)*250, g1=Math.ceil((S.maxx+200)/250)*250;
  for (let x=g0;x<=g1;x+=250){
    cx2.strokeStyle = 'rgba(70,90,120,0.20)';
    let a=proj([x,S.miny-200,0],C,w,h), b=proj([x,S.maxy+200,0],C,w,h);
    if(a&&b){cx2.beginPath();cx2.moveTo(a[0],a[1]);cx2.lineTo(b[0],b[1]);cx2.stroke();}
  }
  for (let y=Math.floor((S.miny-200)/250)*250;y<=S.maxy+200;y+=250){
    cx2.strokeStyle = 'rgba(70,90,120,0.20)';
    let a=proj([S.minx-200,y,0],C,w,h), b=proj([S.maxx+200,y,0],C,w,h);
    if(a&&b){cx2.beginPath();cx2.moveTo(a[0],a[1]);cx2.lineTo(b[0],b[1]);cx2.stroke();}
  }

  // ---- the GATE: a translucent corridor along the required track through the point
  const th=SP.angle*Math.PI/180, d=[Math.sin(th),Math.cos(th)];
  const ctr=[P[0][0],P[0][1]], L=Math.max(500, spanNow()*0.5);
  const cor=[[ctr[0]-d[0]*L, ctr[1]-d[1]*L],[ctr[0]+d[0]*L, ctr[1]+d[1]*L]];
  // RAW metres: `proj` applies the exaggeration, so pre-scaling here would square it.
  // The sheet is given a floor in VISUAL terms (260/vex raw metres) so it stays a useful
  // size at any exaggeration instead of collapsing when the slider is up.
  const zspan=Math.max(60, (S.maxz-S.minz) + 300/vex);
  const zc=P[0][2];
  // four corners: both ends of the corridor, at its bottom and its top. One translucent
  // SHEET rather than two walls -- the requirement is a line, a sheet is the least
  // fussy thing that stays visible from every camera angle.
  const p00=[cor[0][0],cor[0][1],zc-zspan/2], p01=[cor[0][0],cor[0][1],zc+zspan/2];
  const p10=[cor[1][0],cor[1][1],zc-zspan/2], p11=[cor[1][0],cor[1][1],zc+zspan/2];
  const pr=[p00,p01,p11,p10].map(q=>proj(q,C,w,h));
  if (pr.every(Boolean)) {
    cx2.fillStyle='rgba(248,150,30,0.11)';
    cx2.beginPath(); cx2.moveTo(pr[0][0],pr[0][1]);
    for(let i=1;i<4;i++) cx2.lineTo(pr[i][0],pr[i][1]);
    cx2.closePath(); cx2.fill();
    cx2.strokeStyle='rgba(248,150,30,0.40)'; cx2.lineWidth=1; cx2.stroke();
    // the top edge, brighter -- this is the gate the requirement actually names
    cx2.strokeStyle='rgba(248,150,30,0.6)';
    cx2.beginPath(); cx2.moveTo(pr[1][0],pr[1][1]); cx2.lineTo(pr[2][0],pr[2][1]); cx2.stroke();
  }
  let a=proj([cor[0][0],cor[0][1],P[0][2]],C,w,h), b=proj([cor[1][0],cor[1][1],P[0][2]],C,w,h);
  if(a&&b){cx2.strokeStyle='#f8961e';cx2.lineWidth=2;cx2.setLineDash([9,6]);
    cx2.beginPath();cx2.moveTo(a[0],a[1]);cx2.lineTo(b[0],b[1]);cx2.stroke();cx2.setLineDash([]);}

  // ---- the point's own track, then the aircraft's, coloured by time
  if (T.length>1){
    cx2.lineWidth=1.5; cx2.strokeStyle='rgba(155,140,255,0.75)';
    cx2.beginPath(); let st=false;
    for(const p of T){ const q=proj([p[0],p[1],p[2]*vex],C,w,h); if(!q){st=false;continue;}
      if(!st){cx2.moveTo(q[0],q[1]);st=true;} else cx2.lineTo(q[0],q[1]); }
    cx2.stroke();
  }
  // A and T are downsampled to a fixed budget, so a SAMPLE index is not a TIME index.
  // Map t onto the arrays through the original sample count, or the aircraft desynchronises
  // from the point by exactly the downsampling ratio -- visible as the marker arriving
  // late on a long pass.
  const idx = x => Math.max(0, Math.min(A.length-1, Math.round(x/D.dur*(D.total-1))));
  const i0=idx(t), i1=Math.min(A.length-1,i0+1);
  const NB=48, per=Math.max(1,Math.ceil(A.length/NB));
  cx2.lineWidth=2.4;
  for(let b=0;b<NB;b++){
    const s=b*per, e=Math.min(A.length-1,(b+1)*per);
    if(e<s) continue;
    const hue=210-150*(e/A.length);
    cx2.strokeStyle=`hsla(${hue},85%,62%,0.95)`;
    cx2.beginPath(); let first=true;
    for(let i=s;i<=e;i++){
      if (i>t+1) break;
      const p=A[i], q=proj([p[0],p[1],p[2]*vex],C,w,h); if(!q){first=true;continue;}
      if(first){cx2.moveTo(q[0],q[1]);first=false;} else cx2.lineTo(q[0],q[1]);
    }
    cx2.stroke();
  }
  // the whole track, faintly, so the shape is visible before you press play
  cx2.lineWidth=1; cx2.strokeStyle='rgba(120,150,190,0.16)';
  cx2.beginPath(); let first=true;
  for(const p of A){ const q=proj([p[0],p[1],p[2]*vex],C,w,h); if(!q){first=true;continue;}
    if(first){cx2.moveTo(q[0],q[1]);first=false;} else cx2.lineTo(q[0],q[1]); }
  cx2.stroke();

  // ---- the point itself, and the scored crossing
  const pq=proj([P[0][0],P[0][1],P[0][2]*vex],C,w,h);
  if(pq){
    cx2.fillStyle='#f94144';
    cx2.beginPath(); cx2.arc(pq[0],pq[1],6,0,TAU); cx2.fill();
    cx2.font='13px monospace'; cx2.fillText('P', pq[0]+9, pq[1]-7);
  }
  if (SC){
    const c=SC[0], s=c.i;
    const cq=proj([c.p[0],c.p[1],c.p[2]*vex],C,w,h);
    if(cq){
      cx2.strokeStyle='#57cc99'; cx2.lineWidth=2.5;
      cx2.beginPath(); cx2.arc(cq[0],cq[1],9,0,TAU); cx2.stroke();
    }
    const tp=proj([P[0][0],P[0][1],P[0][2]*vex],C,w,h);
    if(cq&&tp){ cx2.strokeStyle='#57cc99'; cx2.lineWidth=3;
      cx2.beginPath(); cx2.moveTo(tp[0],tp[1]); cx2.lineTo(cq[0],cq[1]); cx2.stroke(); }
  }
  // ---- the aircraft: a heading wedge, plus a vertical drop line to the ground
  const p=A[i0], hq=proj([p[0],p[1],p[2]*vex],C,w,h);
  if(hq){
    const nxt=A[Math.min(A.length-1,i0+8)];
    let ang=Math.atan2(-(nxt[1]-p[1]), nxt[0]-p[0]);
    const R=15;
    cx2.fillStyle='#4cc9f0';
    cx2.beginPath();
    cx2.moveTo(hq[0]+Math.cos(ang)*R, hq[1]+Math.sin(ang)*R);
    cx2.lineTo(hq[0]+Math.cos(ang+2.5)*R*0.62, hq[1]+Math.sin(ang+2.5)*R*0.62);
    cx2.lineTo(hq[0]+Math.cos(ang-2.5)*R*0.62, hq[1]+Math.sin(ang-2.5)*R*0.62);
    cx2.closePath(); cx2.fill();
    const gq=proj([p[0],p[1],0],C,w,h);
    if(gq){ cx2.strokeStyle='rgba(76,201,240,0.35)'; cx2.lineWidth=1; cx2.setLineDash([3,4]);
      cx2.beginPath(); cx2.moveTo(hq[0],hq[1]); cx2.lineTo(gq[0],gq[1]); cx2.stroke(); cx2.setLineDash([]); }
    // ground shadow
    if(gq){ cx2.fillStyle='rgba(76,201,240,0.30)';
      cx2.beginPath(); cx2.arc(gq[0],gq[1],4,0,TAU); cx2.fill(); }
  }
  // ---- the point's live marker
  const ti=i0;
  const tq=proj([T[ti][0],T[ti][1],T[ti][2]*vex],C,w,h);
  if(tq){ cx2.strokeStyle='#9b8cff'; cx2.lineWidth=2;
    cx2.beginPath(); cx2.arc(tq[0],tq[1],7,0,TAU); cx2.stroke(); }

  // ---- HUD
  const gs=Math.hypot(p[4],p[5]);
  const trk=gs>0.5?(Math.atan2(p[5],p[4])*180/Math.PI+360)%360:0;
  const cross=(p[0]-T[ti][0])*Math.cos(th+Math.PI/2)+(p[1]-T[ti][1])*Math.sin(th+Math.PI/2);
  const along=(p[0]-T[ti][0])*Math.sin(th)+(p[1]-T[ti][1])*Math.cos(th);
  let sline='',tline='';
  if (SC){ sline=`gate crossing at t=${SC[0].t.toFixed(1)}s · miss ${SC[0].miss.toFixed(2)} m · `
    +`angle ${SC[0].ae>=0?'+':''}${SC[0].ae.toFixed(2)}° · speed ${SC[0].se>=0?'+':''}${SC[0].se.toFixed(2)} m/s`;
    tline=`<span class="ok" id="jump" style="cursor:pointer">▸ jump to the gate crossing</span>`; }
  document.getElementById('hud').innerHTML =
    `<b>t</b> ${t.toFixed(1)} / ${D.dur.toFixed(0)} s &nbsp;&nbsp; `
    +`<b>alt</b> ${(p[3]).toFixed(1)} m &nbsp;&nbsp; <b>gs</b> ${gs.toFixed(1)} m/s `
    +`<span class="dim">(want ${SP.speed})</span><br>`
    +`<b>track</b> ${trk.toFixed(1)}° <span class="dim">(want ${SP.angle}°)</span> &nbsp;&nbsp; `
    +`<b>along</b> ${along.toFixed(0)} m &nbsp;&nbsp; <b>cross</b> ${cross.toFixed(0)} m<br>`
    +`<span class="dim">${sline}</span> ${tline}`;
  document.getElementById('jump')?.addEventListener('click',()=>{ if(SC){t=SC[0].t; playing=false;} });

  // ---- minimap (top-down, equal scale)
  const mw=mm.width, mh=mm.height;
  mx.fillStyle='#0d1017'; mx.fillRect(0,0,mw,mh);
  const sc=Math.min(mw/(S.maxx-S.minx+200), mh/(S.maxy-S.miny+200));
  const ox=(mw-(S.maxx-S.minx)*sc)/2-S.minx*sc, oy=(mh+(S.maxy-S.miny)*sc)/2+S.miny*sc;
  const m2=(x,y)=>[ox+x*sc, oy-y*sc];
  mx.strokeStyle='rgba(248,150,30,0.8)'; mx.lineWidth=1.5;
  let q0=m2(cor[0][0],cor[0][1]), q1=m2(cor[1][0],cor[1][1]);
  mx.beginPath(); mx.moveTo(q0[0],q0[1]); mx.lineTo(q1[0],q1[1]); mx.stroke();
  if(T.length>1){ mx.strokeStyle='rgba(155,140,255,0.7)'; mx.lineWidth=1.5; mx.beginPath();
    for(let i=0;i<T.length;i++){const q=m2(T[i][0],T[i][1]); i?mx.lineTo(q[0],q[1]):mx.moveTo(q[0],q[1]);}
    mx.stroke(); }
  mx.strokeStyle='rgba(120,150,190,0.25)'; mx.lineWidth=1; mx.beginPath();
  for(let i=0;i<A.length;i++){const q=m2(A[i][0],A[i][1]); i?mx.lineTo(q[0],q[1]):mx.moveTo(q[0],q[1]);}
  mx.stroke();
  mx.strokeStyle='#4cc9f0'; mx.lineWidth=2.2; mx.beginPath();
  for(let i=0;i<=i0;i++){const q=m2(A[i][0],A[i][1]); i?mx.lineTo(q[0],q[1]):mx.moveTo(q[0],q[1]);}
  mx.stroke();
  let pp=m2(P[0][0],P[0][1]); mx.fillStyle='#f94144';
  mx.beginPath(); mx.arc(pp[0],pp[1],4,0,TAU); mx.fill();
  let tp2=m2(T[ti][0],T[ti][1]); mx.strokeStyle='#9b8cff'; mx.lineWidth=2;
  mx.beginPath(); mx.arc(tp2[0],tp2[1],5,0,TAU); mx.stroke();

  if (playing) { t += 0.016*speed*4; if (t>D.dur) t=0; document.getElementById('seek').value=t; }
  dirty=false;
}

// ---- interaction
let drag=null;
cv.addEventListener('mousedown',e=>{drag={x:e.clientX,y:e.clientY,pan:e.shiftKey};});
window.addEventListener('mouseup',()=>{drag=null;});
window.addEventListener('mousemove',e=>{
  if(!drag) return;
  const dx=e.clientX-drag.x, dy=e.clientY-drag.y; drag.x=e.clientX; drag.y=e.clientY;
  if(drag.pan){ panX-=dx*span*0.0015; panY+=dy*span*0.0015; }
  else { az-=dx*0.006; el=Math.max(-1.45,Math.min(1.45,el+dy*0.005)); }
});
cv.addEventListener('wheel',e=>{ e.preventDefault();
  dist=Math.max(0.35,Math.min(6,dist*(1+Math.sign(e.deltaY)*0.08))); },{passive:false});
document.getElementById('seek').addEventListener('input',e=>{t=+e.target.value;playing=false;syncPlay();});
function syncPlay(){ document.getElementById('play').textContent = playing?'❚❚ pause':'▶ play'; }
document.getElementById('play').addEventListener('click',()=>{playing=!playing;syncPlay();});
document.getElementById('spd').addEventListener('input',e=>{speed=+e.target.value;
  document.getElementById('spdv').textContent=speed.toFixed(1)+'x';});
document.getElementById('vex').addEventListener('input',e=>{vex=+e.target.value;
  document.getElementById('vexv').textContent=vex.toFixed(0)+'x';});
document.getElementById('view').addEventListener('dblclick',()=>{az=-0.9;el=0.42;dist=1.5;panX=panY=0;});
syncPlay(); frame();
"""

_HTML = r"""<!doctype html><meta charset="utf-8">
<title>__TITLE__</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#12151c;color:#e8eaf0;font:13px -apple-system,Segoe UI,Roboto,sans-serif}
#wrap{display:grid;grid-template-columns:1fr 300px;gap:0;height:100vh}
#left{display:flex;flex-direction:column}
#view{flex:1;display:block;cursor:grab;min-height:0}
#view:active{cursor:grabbing}
#bar{display:flex;align-items:center;gap:14px;padding:9px 14px;background:#181c25;
     border-top:1px solid #2a3040;flex-wrap:wrap}
#hud{flex:1;min-width:260px;font-family:ui-monospace,Menlo,monospace;font-size:12px;line-height:1.55}
b{color:#8b93a7;font-weight:600}
.dim{color:#6f7789}
.ok{color:#57cc99}
#right{border-left:1px solid #2a3040;background:#0f1219;display:flex;flex-direction:column}
#mini{display:block;width:100%;height:300px;border-bottom:1px solid #2a3040}
#meta{padding:14px 16px;font-size:12.5px;line-height:1.65;overflow:auto}
#meta h3{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#8b93a7}
#meta table{width:100%;border-collapse:collapse}
#meta td{padding:2px 0}
#meta td:last-child{text-align:right;font-family:ui-monospace,Menlo,monospace}
input[type=range]{accent-color:#4cc9f0;width:130px}
button{background:#232a36;color:#e8eaf0;border:1px solid #333c4d;border-radius:5px;
       padding:5px 12px;cursor:pointer;font-size:12px}
button:hover{background:#2c3444}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;margin-left:6px}
.tag.ok{background:rgba(87,204,153,.16)}
.tag.no{background:rgba(249,65,68,.18);color:#f94144}
</style>
<div id="wrap">
  <div id="left">
    <canvas id="view"></canvas>
    <div id="bar">
      <button id="play">▶ play</button>
      <input id="seek" type="range" min="0" max="100" value="0" step="0.05">
      <label>speed <input id="spd" type="range" min="0.1" max="6" value="1" step="0.1">
        <span id="spdv" class="dim">1.0x</span></label>
      <label>vert &times; <input id="vex" type="range" min="1" max="30" value="8" step="1">
        <span id="vexv" class="dim">8x</span></label>
      <span class="dim">drag to orbit · shift-drag to pan · wheel to zoom · double-click to reset</span>
    </div>
  </div>
  <div id="right">
    <canvas id="mini"></canvas>
    <div id="meta">
      <h3>the requirement</h3>
      <table id="spec"></table>
      <h3>at the gate</h3>
      <table id="score"></table>
      <h3>notes</h3>
      <div id="notes" class="dim"></div>
    </div>
  </div>
</div>
<script>window.__FLYBY__ = __DATA__;</script>
<script>__JS__</script>
<script>
(function(){
  const D=window.__FLYBY__, SP=D.spec, SC=D.score&&D.score[0];
  const row=(k,v,cls)=>`<tr><td>${k}</td><td class="${cls||''}">${v}</td></tr>`;
  document.getElementById('spec').innerHTML =
    row('point', SP.lat.toFixed(6)+', '+SP.lon.toFixed(6))
  + row('altitude', SP.alt.toFixed(1)+' m MSL')
  + row('track', SP.angle+'°')
  + row('ground speed', SP.speed+' m/s')
  + (SP.fp!==null&&SP.fp!==undefined ? row('flight path', SP.fp+'°') : '')
  + row('point motion', D.motion);
  document.getElementById('score').innerHTML = SC ?
      row('crossing at', SC.t.toFixed(2)+' s')
    + row('miss (2D)', SC.miss.toFixed(2)+' m', Math.abs(SC.miss)<5?'ok':'')
    + row('miss (3D)', SC.miss3.toFixed(2)+' m')
    + row('track error', (SC.ae>=0?'+':'')+SC.ae.toFixed(2)+'°', Math.abs(SC.ae)<5?'ok':'')
    + row('speed error', (SC.se>=0?'+':'')+SC.se.toFixed(2)+' m/s', Math.abs(SC.se)<1?'ok':'')
    + row('altitude error', (SC.alt>=0?'+':'')+SC.alt.toFixed(1)+' m')
    + (SC.fpe!==null&&SC.fpe!==undefined ? row('flight-path err',(SC.fpe>=0?'+':'')+SC.fpe.toFixed(2)+'°') : '')
    + row('law says', SC.ok?'flyable':'INFEASIBLE', SC.ok?'ok':'no')
    : '<tr><td colspan=2 class="dim">no crossing of the gate</td></tr>';
  document.getElementById('notes').innerHTML = D.notes||'';
  document.getElementById('seek').max = D.dur;
})();
</script>
"""


def _downsample(rows, n):
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(n)]


def write_3d_player(samples, spec, scores, path, notes="", motion="static", origin=None):
    """A single self-contained HTML file: orbit/zoom/pan 3-D, playback, scrub, a synced
    top-down minimap, and the scored gate crossing marked on the timeline."""
    if not samples:
        return None
    origin = origin or geo.Origin()

    # world frame for the viewer: x east, y north, z UP (metres above the NED origin).
    # Colour is by time; the extra fields carry the HUD's numbers.
    A, T = [], []
    for s in _downsample(samples, 2400):
        A.append([round(s.ned[1], 2), round(s.ned[0], 2), round(-s.ned[2], 2),
                  round(-s.ned[2] + origin.alt_m, 2),      # altitude MSL
                  round(s.vel[1], 3), round(s.vel[0], 3)])
        T.append([round(s.target_ned[1], 2), round(s.target_ned[0], 2), round(-s.target_ned[2], 2)])
    P = [[round(spec.ned[1], 2), round(spec.ned[0], 2), round(-spec.ned[2], 2)]]

    SC = None
    if scores:
        c = scores[0]
        i = min(range(len(samples)),
                key=lambda k: abs(samples[k].t - c.t_cross))
        SC = dict(t=c.t_cross, i=i,
                  p=[round(c.ned_at_cross[1], 2), round(c.ned_at_cross[0], 2),
                     round(-c.ned_at_cross[2], 2)],
                  miss=c.miss_2d_m, miss3=c.miss_3d_m, ae=c.angle_err_deg,
                  se=c.speed_err_mps, alt=c.alt_err_m, fpe=c.fp_angle_err_deg,
                  ok=c.feasible)

    data = dict(aircraft=A, target=T, point=P, dur=samples[-1].t, total=len(samples),
                notes=notes, motion=motion,
                spec=dict(angle=spec.angle_deg, speed=spec.speed_mps, alt=spec.alt_m,
                          fp=spec.fp_angle_deg, lat=spec.lat, lon=spec.lon),
                score=[SC] if SC else None)

    title = f"{spec.angle_deg:.0f}° @ {spec.speed_mps:.0f} m/s"
    html_src = (_HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
                     .replace("__JS__", _JS)
                     .replace("__TITLE__", html.escape(f"fly-through {title}")))
    with open(path, "w") as f:
        f.write(html_src)
    return path
