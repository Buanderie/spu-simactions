"""Running a scenario, offline or against PX4, and looking at it.

One entry point for both worlds. `run_offline` closes the law against the reference
plant in `guidance.py`; `run_live` closes it against PX4's real flight stack over
MAVLink. Both hand back the same thing -- a list of `score.Sample` and the scored
crossings -- so a number measured in simulation and a number measured in the rig are
produced by identical code and comparable by construction.

The live ASCII map is here rather than in `viz.py` because it has to be drawn from
inside the control loop, interleaved with telemetry, and it must never be able to stall
that loop: it is a debugging view for a person watching a terminal, not a deliverable.
"""
import math
import sys
import time

from . import geo, score
from .guidance import (PassSpec, TerminalProfileGuidance, GuidanceConfig, PlantConfig,
                        decompose, simulate_pass)


# ----------------------------------------------------------------------- offline

def run_offline(scenario, guidance_cfg=None, plant_cfg=None, dt=0.05, duration_s=None,
                duration_margin=1.6):
    """Close the law against the reference plant and score the crossing."""
    cfg = guidance_cfg or GuidanceConfig()
    plant = plant_cfg or PlantConfig()
    # Run long enough to actually cross: estimate the time from the along-track geometry
    # rather than hard-coding a duration per scenario. Project onto the PASS LINE, not
    # onto the track's north component -- a pass on 090 has no along-track component in
    # north at all, and the estimate then says "infinitely far" and the run is over before
    # the gate.
    tgt, tgt_vel = scenario.target_fn(0.0)
    if duration_s is None:
        _s0, _e0, s_dot0, _ed0 = decompose(
            (scenario.start_ned[0] - tgt[0], scenario.start_ned[1] - tgt[1]),
            (scenario.start_vel_ned[0] - tgt_vel[0], scenario.start_vel_ned[1] - tgt_vel[1]),
            scenario.spec.angle_deg)
        closing = max(1e-3, min(max(s_dot0, 0.25 * scenario.spec.speed_mps),
                                scenario.spec.speed_mps))
        duration_s = abs(_s0) / closing * duration_margin + 25.0
    traj = simulate_pass(
        TerminalProfileGuidance(cfg), scenario.start_ned, scenario.start_vel_ned,
        scenario.spec, scenario.target_fn, dt=dt, duration_s=duration_s, plant=plant)

    samples = [score.Sample(t=t, ned=p, vel=v, target_ned=pl.target_ned,
                            target_vel=pl.target_vel_ned, plan=pl)
               for t, p, v, pl in traj]
    return samples, score.score_run(samples, scenario.spec), traj


# ------------------------------------------------------------------- ascii map

class AsciiMap:
    """A live top-down view of the pass frame, drawn in the terminal.

    North-up, east-right, one character per cell, the aircraft drawn as its current
    heading wedge. The pass line is the vertical axis through the middle column, which
    is the whole point of the picture: you can see the offset being nulled and the line
    being met, and you cannot see either in a list of numbers.
    """

    def __init__(self, spec: PassSpec, half_span_m=700.0, cols=49, rows=25, origin=None):
        self.spec = spec
        self.origin = origin or geo.Origin()
        self.half = half_span_m
        self.cols, self.rows = cols, rows
        self.aspect = (2.0 * half_span_m) / cols / ((2.0 * half_span_m) / rows)
        self.trail = []

    def _rot(self, ned):
        """Pass-frame metres -> (col, row) integers. Row 0 is north."""
        d, r = self.spec.angle_deg, None
        dn = ned[0] - self._tgt[0]
        de = ned[1] - self._tgt[1]
        # rotate by -angle so the required track points "up" the picture
        a = math.radians(d)
        along = dn * math.cos(a) + de * math.sin(a)
        right = -dn * math.sin(a) + de * math.cos(a)
        col = self.cols // 2 + int(round(right / self.half * (self.cols // 2)))
        row = self.rows // 2 - int(round(along / self.half * (self.rows // 2)))
        return col, row

    def update(self, sample: score.Sample, trail_max=400):
        self._tgt = sample.target_ned
        self.trail.append(self._rot(sample.ned))
        if len(self.trail) > trail_max:
            del self.trail[: len(self.trail) - trail_max]

    def render(self, sample: score.Sample, header: str = "") -> str:
        self.update(sample)
        grid = [[" "] * self.cols for _ in range(self.rows)]
        # the pass line, and the point
        for row in range(self.rows):
            grid[row][self.cols // 2] = "|"
        grid[self.rows // 2][self.cols // 2] = "P"
        # the point's own track, if it has one
        if abs(sample.target_vel[0]) + abs(sample.target_vel[1]) > 0.2:
            for k in range(-self.rows, self.rows):
                tt = k * 2.0
                n = sample.target_ned[0] + sample.target_vel[0] * tt
                e = sample.target_ned[1] + sample.target_vel[1] * tt
                c, r = self._rot((n, e, 0.0))
                if 0 <= r < self.rows and 0 <= c < self.cols and grid[r][c] in " |":
                    grid[r][c] = "."
        for (c, r) in self.trail:
            if 0 <= r < self.rows and 0 <= c < self.cols and grid[r][c] in " |.":
                grid[r][c] = ":"
        # the aircraft, with a heading wedge
        c, r = self.trail[-1]
        if 0 <= r < self.rows and 0 <= c < self.cols:
            grid[r][c] = "A"
            gs = math.hypot(sample.vel[0], sample.vel[1])
            if gs > 0.5:
                trk = math.degrees(math.atan2(sample.vel[1], sample.vel[0]))
                for d in (10, 20):
                    dn = math.cos(math.radians(trk)) * gs * d * 0.1
                    de = math.sin(math.radians(trk)) * gs * d * 0.1
                    cc, rr = self._rot((sample.ned[0] + dn, sample.ned[1] + de, 0.0))
                    if 0 <= rr < self.rows and 0 <= cc < self.cols and grid[rr][cc] in " |.:P":
                        grid[rr][cc] = "^"
        lines = [header] if header else []
        lines.append("+" + "-" * self.cols + "+   ^ required track is UP, P is the point")
        for row in range(self.rows):
            lines.append("|" + "".join(grid[row]) + "|")
        lines.append("+" + "-" * self.cols + "+")
        return "\n".join(lines)


class Progress:
    """A one-line-per-tick status strip, for the terminal when a full map is overkill."""

    def __init__(self, enabled=True, every_s=1.0):
        self.enabled = enabled
        self.every = every_s
        self._last = -1e9

    def tick(self, t: float, sample: score.Sample, plan, extra: str = "") -> None:
        if not self.enabled or (t - self._last) < self.every:
            return
        self._last = t
        gs = math.hypot(sample.vel[0], sample.vel[1])
        trk = math.degrees(math.atan2(sample.vel[1], sample.vel[0])) % 360.0 if gs > 0.5 else float("nan")
        flag = "" if plan is None else ("" if plan.feasible else " !INFEASIBLE")
        sys.stderr.write(
            f"\r  t={t:6.1f}s  s={plan.along_track_m:8.1f}  e={plan.cross_track_m:7.1f}  "
            f"e'={plan.cross_rate_mps:6.2f}  gs={gs:5.1f}  trk={trk:5.1f}  "
            f"dAlt={sample.ned[2]-sample.target_ned[2]:+6.1f}  tgo={plan.t_go_s:5.1f}{flag} {extra}   ")
        sys.stderr.flush()

    def end(self):
        if self.enabled:
            sys.stderr.write("\n")
            sys.stderr.flush()
