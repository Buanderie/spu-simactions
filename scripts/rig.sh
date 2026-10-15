#!/usr/bin/env bash
# Start / stop / inspect the simulated aircraft, natively -- no Docker.
#
# swarmpu-simuav's own `docker compose up world uav-0` is the reference rig, but for
# iterating on a guidance law it is the wrong shape: every experiment pays a container
# start, and the PX4 binary it needs (extdeps/PX4-Autopilot/build/px4_sitl_default, a
# ~53 MB PATCHED build nobody has re-seeded here) is a bind mount, not a build product.
#
# The two processes are the same ones the compose file runs, and they talk exactly the same
# protocol -- hil.py owns the TCP listener on 4560+instance and PX4 dials in, and PX4's
# onboard MAVLink stream lands on udp:14540:
#
#   engine/fdm/hil.py   the flight dynamics model (engine/fdm/{fixedwing,multicopter}.py)
#   px4                 a real PX4 SITL, flying its whole stack against that plant
#
# Which PX4 binary: PX4_SITL_BUILD below. The default is the stock tree at
# /home/nis/dev/PX4-Autopilot, whose `px4_sitl_nolockstep` build exists on this host. That
# build has NOT had swarmpu's fixed-wing OFFBOARD patch applied (the patch is in
# swarmpu-simuav/extdeps/patches/), so OFFBOARD works fully for the quadrotor and the
# fixed-wing path is the one that needs a patched build -- see docs/FINDINGS.md.
set -uo pipefail

SIMUAV="${SIMUAV:-/home/nis/dev/swarmpu-simuav}"
PX4_SITL_BUILD="${PX4_SITL_BUILD:-/home/nis/dev/PX4-Autopilot/build/px4_sitl_nolockstep}"
PX4_AUTOPILOT_SRC="${PX4_AUTOPILOT_SRC:-/home/nis/dev/PX4-Autopilot}"
AIRFRAME="${SIMUAV_AIRFRAME:-quad}"
IDX="${IDX:-0}"
RUN="${RUN:-/tmp/spu-flyby-rig}"
PHYSICS_HZ="${PHYSICS_HZ:-250}"
TRUTH_PORT="${TRUTH_PORT:-8099}"
# Any aircraft profile simuav knows (config/profiles/*.json) can be flown: PROFILE=geran2 ./rig.sh
# start. Left unset, the kind comes from AIRFRAME and simuav's own default profile applies
# (iris for quad, trainer for plane). A profile that does not exist is a hard error here
# rather than a silent fallback to the default aircraft -- flying a different airframe than
# the one asked for and reporting the numbers as if it were the right one is the exact
# failure this whole rig exists to prevent.
PROFILE="${SIMUAV_PROFILE:-${PROFILE:-}}"
# PX4 writes parameters.bson on shutdown and reloads it on the next boot, so a limit changed
# by one experiment is still in place for the next one. That is fine in production and fatal
# in a benchmark: "run against the stock autopilot" silently measures against whatever the
# previous run left behind. So the working directory starts empty of parameters by default
# and every run begins from PX4's own defaults. Set RESET_PARAMS=0 to keep them.
RESET_PARAMS="${RESET_PARAMS:-1}"

# The scene origin has to agree with PX4's, or the FDM synthesises GPS for one place while
# the controller converts positions against another. swarmpu-simuav ships config/scene.yaml
# with PX4's factory Zurich coordinates explicitly set; passing ORIGIN_* here overrides both.
ORIGIN_ARGS=()
for v in ORIGIN_LAT ORIGIN_LON ORIGIN_ALT; do
  [ -n "${!v:-}" ] && ORIGIN_ARGS+=(--origin_lat "${ORIGIN_LAT}" --origin_lon "${ORIGIN_LON}" --origin_alt "${ORIGIN_ALT}")
done

profile_args() {
  [ -z "${PROFILE:-}" ] && return 0
  local p="$SIMUAV/config/profiles/$PROFILE.json"
  [ -f "$p" ] || { echo "no simuav profile at $p (PROFILE=$PROFILE)" >&2; exit 2; }
  echo "--profile $PROFILE"
}

# The PX4 airframe to boot is a property of the PROFILE, not of AIRFRAME, so a profile that
# is not an iris-class multirotor gets its own PX4_SIM_MODEL instead of the one AIRFRAME
# implies. Read the same file the FDM reads so the two cannot disagree.
model_for() {
  if [ -n "${PROFILE:-}" ]; then
    ( cd "$SIMUAV" && PYTHONPATH="$SIMUAV" python3 -c "
import sys
from engine.fdm import profiles
p = profiles.load_profile('$PROFILE')
sys.stdout.write(p['px4_sim_model'])
" ) && return 0
    echo "  (could not read px4_sim_model for profile $PROFILE)" >&2
  fi
  case "$1" in
    quad)  echo gazebo-classic_iris ;;
    plane) echo gazebo-classic_plane ;;
    *)      echo "unknown airframe: $1 (want quad|plane)" >&2; exit 2 ;;
  esac
}

# simuav's own name for the airframe kind, which is not the same string in both places.
kind_for() {
  if [ -n "${PROFILE:-}" ]; then
    ( cd "$SIMUAV" && PYTHONPATH="$SIMUAV" python3 -c "
import sys
from engine.fdm import profiles
sys.stdout.write(profiles.hil_kind(profiles.load_profile('$PROFILE')))
" ) && return 0
  fi
  case "$1" in
    quad)  echo quad ;;
    plane) echo plane ;;
    *)      echo "unknown airframe: $1 (want quad|plane)" >&2; exit 2 ;;
  esac
}

wait_port() {  # port, seconds, label
  local port="$1" secs="$2" label="$3" i=0
  while [ "$i" -lt "$((secs * 10))" ]; do
    ss -ltn "sport = :$port" 2>/dev/null | grep -q ":$port" && return 0
    sleep 0.1; i=$((i + 1))
  done
  echo "  timed out waiting for $label on port $port" >&2
  return 1
}

start() {
  stop >/dev/null 2>&1
  local kind; kind="$(kind_for "$AIRFRAME")"
  local tag="${PROFILE:-$AIRFRAME}"
  mkdir -p "$RUN/px4-$tag-$IDX" "$RUN/logs"
  local model; model="$(model_for "$kind")"

  if [ ! -x "$PX4_SITL_BUILD/bin/px4" ]; then
    echo "no PX4 SITL binary at $PX4_SITL_BUILD/bin/px4" >&2
    echo "set PX4_SITL_BUILD, or build one: see the header of this script" >&2
    exit 2
  fi

  local profile_flag; profile_flag="$(profile_args)"

  if [ "$RESET_PARAMS" = "1" ]; then
    rm -f "$RUN/px4-$tag-$IDX"/parameters.bson "$RUN/px4-$tag-$IDX"/parameters_backup.bson
    echo "  parameters: reset to PX4 defaults (RESET_PARAMS=1)"
  fi

  echo "  FDM: engine/fdm/hil.py, kind=$kind (profile: ${PROFILE:-$kind default}), physics ${PHYSICS_HZ} Hz, truth :$TRUTH_PORT"
  # setsid + full redirection: the shell tool (and any CI) must not be left holding the
  # pipe of a process that is meant to outlive the command that started it.
  ( cd "$SIMUAV" && PYTHONPATH="$SIMUAV" setsid python3 -u -m engine.fdm.hil \
      --num_uavs 1 --airframe "$kind" --initial_yaw_deg 0 \
      ${profile_flag:+$profile_flag} \
      --physics_hz "$PHYSICS_HZ" --status_period_s "${STATUS_PERIOD_S:-5}" \
      --scene_config config/scene.yaml --truth_port "$TRUTH_PORT" \
      ${ORIGIN_ARGS[@]+"${ORIGIN_ARGS[@]}"} \
      >"$RUN/logs/fdm.log" 2>&1 < /dev/null & echo $! > "$RUN/fdm.pid" )

  wait_port $((4560 + IDX)) 30 "the FDM's simulator_mavlink listener" || { stop; exit 1; }
  echo "  FDM listening on tcp:$((4560 + IDX))"

  echo "  PX4: $model, instance $IDX, against 127.0.0.1:$((4560 + IDX))"
  ( cd "$RUN/px4-$tag-$IDX" && \
    PX4_SIM_MODEL="$model" PX4_SIM_HOSTNAME=127.0.0.1 \
    PX4_PARAM_COM_VFR_HUD_FREQ=10 \
    setsid "$PX4_SITL_BUILD/bin/px4" -i "$IDX" -d "$PX4_SITL_BUILD/etc" \
      >"$RUN/logs/px4.log" 2>&1 < /dev/null & echo $! > "$RUN/px4.pid" ) >/dev/null 2>&1

  # PX4 boots, runs its startup script and only then opens the mavlink streams. Wait for
  # the onboard one specifically -- the GCS link (14550) comes up first and is a false
  # positive for "PX4 is ready".
  sleep 3
  for _ in $(seq 1 60); do
    grep -q "Startup script returned successfully" "$RUN/logs/px4.log" 2>/dev/null && break
    sleep 0.5
  done
  grep -q "Startup script returned successfully" "$RUN/logs/px4.log" 2>/dev/null || {
    echo "  PX4 did not finish startup; tail of its log:" >&2
    tail -20 "$RUN/logs/px4.log" >&2; stop; exit 1; }

  echo "  PX4 up; onboard MAVLink on udp:14540 (nothing listens there until the controller does)"
  echo "  logs: $RUN/logs/{fdm,px4}.log     stop: $0 stop"
  status
}

stop() {
  for f in "$RUN/fdm.pid" "$RUN/px4.pid"; do
    [ -f "$f" ] || continue
    local p; p="$(cat "$f" 2>/dev/null)"
    [ -n "$p" ] && kill -TERM "-$p" 2>/dev/null
    pkill -TERM -P "$p" 2>/dev/null
    rm -f "$f"
  done
  pkill -f "engine.fdm.hil" 2>/dev/null
  pkill -f "px4_sitl.*bin/px4" 2>/dev/null
  sleep 0.4
  return 0
}

status() {
  printf '  FDM  pid %s   %s\n' "$(cat "$RUN/fdm.pid" 2>/dev/null || echo -)" \
    "$(ss -ltn 2>/dev/null | grep -c ":$((4560 + IDX))" | sed 's/^0$/not listening/;s/^[1-9].*/listening/')"
  printf '  PX4  pid %s   %s\n' "$(cat "$RUN/px4.pid" 2>/dev/null || echo -)" \
    "$(grep -c 'Startup script returned successfully' "$RUN/logs/px4.log" 2>/dev/null | sed 's/^0$/booting/;s/^[1-9].*/up/')"
  if command -v curl >/dev/null && [ -n "${TRUTH_PORT:-}" ]; then
    curl -s --max-time 2 "http://127.0.0.1:$TRUTH_PORT/truth" 2>/dev/null | head -c 400
    echo
  fi
}

case "${1:-status}" in
  start)  start ;;
  stop)   echo "  stopped"; stop ;;
  status) status ;;
  logs)   tail -n "${2:-25}" "$RUN/logs/fdm.log" "$RUN/logs/px4.log" ;;
  *)      echo "usage: $0 {start|stop|status|logs [n]}" >&2; exit 2 ;;
esac
