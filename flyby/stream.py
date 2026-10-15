"""Append telemetry to a JSON-lines file while the flight is still happening.

The point is that the artifact exists DURING the run, not after it. Everything else in this
repo's visual output is written once `run_pass` returns, which means a 150 s approach that
goes wrong in its first ten seconds costs you 150 s before you can look at any of it.

The format is one JSON object per line, so a reader can attach mid-flight and read what
exists, and a killed run still leaves a playable trail. Nothing here is clever and nothing
here is on the control loop's critical path beyond one `write` of a small string per tick --
which is the whole design constraint: if streaming ever slowed the control loop down, it
would be measuring the wrong flight.
"""
import json
import math
import pathlib
import time


class StreamWriter:
    """One flight, appended as it flies.

    Not thread-safe and does not need to be: it is driven from the control loop, in order,
    on one thread. `flush=True` every tick is deliberate -- a buffered line the viewer cannot
    see is a line that is not live, and a live viewer that lags by a buffer is not one.
    """

    def __init__(self, path, spec=None, meta=None):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate: this file is one flight, not an archive. Appending to yesterday's run
        # is exactly the class of bug FINDINGS section 3 records about the CSV.
        self.f = self.path.open("w")
        self.n = 0
        self._gate = None
        self.t0 = time.monotonic()

    def mark(self) -> float:
        """Session-elapsed seconds, for a caller that keeps its OWN local clock.

        The control loop and the pass loop both count from their own start, so writing
        either one's raw tick into one file produces a timeline that runs forward, jumps
        backwards at the phase boundary, and then runs forward again -- which looks exactly
        like two flights interleaved and makes the viewer's playhead jump. The stream owns
        one clock; a later phase adds its offset to its local time and everything stays on
        one monotonic axis.
        """
        return time.monotonic() - self.t0
        if spec is not None or meta is not None:
            mp = self.path.with_suffix(".meta.json")
            mp.write_text(json.dumps({"spec": spec or {}, "meta": meta or {}}))
            self.meta_path = mp

    def spec_json(self, spec, motion="static"):
        """Overwrite the meta file with the pass requirement (called once the spec is known)."""
        mp = getattr(self, "meta_path", self.path.with_suffix(".meta.json"))
        fp = getattr(spec, "fp_angle_deg", None)
        ned = None
        try:
            ned = [round(v, 2) for v in spec.ned]
        except Exception:
            ned = None
        self._brg = round(float(spec.angle_deg), 3)
        mp.write_text(json.dumps({"spec": {
            "angle": float(spec.angle_deg), "speed": float(spec.speed_mps),
            "fp": None if fp is None else float(fp),
            "ned": ned or [0, 0, 0],
            "gate": getattr(spec, "gate", "line"),
        }, "meta": {"motion": motion, "airframe": str(ROOT)}}))
        self.meta_path = mp

    def mark_gate(self, t, ned, miss_m=None):
        """Record where the pass was actually scored, so the viewer can mark it.

        A terminal descent ends on the ground and the law then flies ON along the line for
        the rest of the run, so the trajectory continues hundreds of metres past the point
        that was being aimed at. Without a marker the viewer shows that tail as if the
        approach had missed, which is the opposite of what happened.
        """
        self._gate = {"t": round(float(t), 3), "ned": [round(float(v), 2) for v in ned],
                      "miss": None if miss_m is None else round(float(miss_m), 3)}

    def write(self, t, ned, vel, target_ned=None, plan=None):
        # `target_ned` is optional: during climb-out and transit there is no pass in
        # progress, and the trajectory is still worth watching -- which is the point, since
        # those two phases are a third of the wall clock and are where a rig problem shows
        # up first. The along/cross errors are simply absent then rather than zeroed, which
        # would be a measurement that does not exist.
        tn = target_ned or ned
        # Distinct keys, one meaning each: `east` is a coordinate, `cross` is an error
        # against the pass line, and conflating them (or reusing one name for both) is
        # invisible until the viewer draws the trail somewhere absurd.
        rec = {
            "t": round(t, 3),
            "east": round(ned[1], 2),
            "north": round(ned[0], 2),
            "alt": round(-ned[2], 2),                 # metres above the NED origin
            "amsl": round(-ned[2], 2),                # altitude MSL is offset at load
            "gs": round(math.hypot(vel[0], vel[1]), 3),
            "vd": round(vel[2], 3),                   # NED down-positive
            "along": None if target_ned is None else round(ned[0] - target_ned[0], 2),
            "cross": None if target_ned is None else round(ned[1] - target_ned[1], 2),
            "cvz": None,
            "ok": None,
            "gate": (1 if (getattr(self, "_gate", None)
                           and abs(self._gate["t"] - t) < 0.06) else 0),
            # The point and the pass bearing ride on every row, so a reader can tell which
            # case a sample belongs to without trusting the meta file to still describe it.
            # The meta file is rewritten per case and a viewer that drew every case against
            # the FIRST case's point showed approaches "missing" a point they never aimed at.
            "tu": None if target_ned is None else round(-target_ned[2], 2),
            "brg": getattr(self, "_brg", None),
        }
        if plan is not None and plan.vel_ned is not None:
            rec["cvz"] = round(plan.vel_ned[2], 3)
            rec["ok"] = bool(plan.feasible)
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()
        self.n += 1

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


ROOT = pathlib.Path(__file__).resolve().parent.parent