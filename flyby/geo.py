"""Geodetic frame for the flyby sandbox.

Same equirectangular approximation as swarmpu-simuav's `engine/geo.py` (R = 6371000 m,
metres -> degrees of latitude/longitude), deliberately re-derived rather than imported:
the point of this repo is to find algorithms that later get written into swarmpu's
firmware, where nothing is importable. Keeping the arithmetic identical matters more
than sharing the file, and `tests/test_geo_consistency.py` checks that it *is*
identical rather than trusting the comment.
"""
import math
from dataclasses import dataclass

EARTH_RADIUS_M = 6371000.0

# swarmpu-simuav's scene origin (config/scene.yaml), which is also PX4 SITL's factory
# home. Every PX4 instance in this rig reports lat/lon against it, so it is the only
# correct thing to use unless the rig was booted with ORIGIN_* overrides.
DEFAULT_LAT_DEG = 47.397742
DEFAULT_LON_DEG = 8.545594
DEFAULT_ALT_M = 488.0


@dataclass(frozen=True)
class Origin:
    lat_deg: float = DEFAULT_LAT_DEG
    lon_deg: float = DEFAULT_LON_DEG
    alt_m: float = DEFAULT_ALT_M


def ned_to_latlon(north_m: float, east_m: float, origin: Origin = Origin()):
    return (
        origin.lat_deg + math.degrees(north_m / EARTH_RADIUS_M),
        origin.lon_deg + math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(origin.lat_deg)))),
    )


def latlon_to_ned(lat_deg: float, lon_deg: float, origin: Origin = Origin()):
    north = math.radians(lat_deg - origin.lat_deg) * EARTH_RADIUS_M
    east = math.radians(lon_deg - origin.lon_deg) * EARTH_RADIUS_M * math.cos(math.radians(origin.lat_deg))
    return north, east


def latlon_alt_to_ned(lat_deg: float, lon_deg: float, alt_m: float, origin: Origin = Origin()):
    """WGS84 -> local NED (north, east, down). Altitude is MSL, matching what a GPS
    reports and what hil.py synthesises (`HOME_ALT_M + altitude_m`)."""
    north, east = latlon_to_ned(lat_deg, lon_deg, origin)
    return north, east, origin.alt_m - alt_m


def ned_to_latlon_alt(north: float, east: float, down: float, origin: Origin = Origin()):
    lat, lon = ned_to_latlon(north, east, origin)
    return lat, lon, origin.alt_m - down


def offset_latlon(lat_deg: float, lon_deg: float, north_m: float, east_m: float):
    """Displace a lat/lon by a local north/east metre offset."""
    dlat = math.radians(north_m / EARTH_RADIUS_M)
    dlon = math.radians(east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat_deg))))
    return lat_deg + math.degrees(dlat), lon_deg + math.degrees(dlon)


def wrap_deg(a: float) -> float:
    """Wrap to [0, 360)."""
    return a % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Signed smallest difference a - b, in (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


# ---- pass-frame helpers --------------------------------------------------------
# The pass frame is the only coordinate system this guidance cares about: everything
# is expressed relative to the point we have to fly through, along the line we have to
# fly it on. d = (cos t, sin t) on (north, east) with t the compass bearing; r is d
# rotated 90 deg clockwise on a north-up map, i.e. "right of the track".

def track_unit(angle_deg: float) -> tuple:
    t = math.radians(angle_deg)
    return math.cos(t), math.sin(t)


def right_unit(angle_deg: float) -> tuple:
    """Unit vector 90 deg to the right of the track heading (north-up, east-right)."""
    t = math.radians(angle_deg)
    return -math.sin(t), math.cos(t)


def bearing_deg(from_ned: tuple, to_ned: tuple) -> float:
    """Compass bearing from one (north, east) point to another."""
    dn = to_ned[0] - from_ned[0]
    de = to_ned[1] - from_ned[1]
    if dn == 0.0 and de == 0.0:
        return 0.0
    return math.degrees(math.atan2(de, dn)) % 360.0
