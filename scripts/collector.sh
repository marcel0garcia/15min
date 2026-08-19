#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  24/7 corpus collector
#
#  The bottleneck on this project is not thinking, it is settled markets.
#  KXBTC15M lists exactly ONE 15-minute market at a time — 96 per day — and
#  each one is a single near-coin-flip. Reaching the R1 bar of 300 settled
#  observations therefore takes ~75 hours of runtime no matter how clever
#  the analysis is. This script's only job is to manufacture that runtime
#  unattended, and to capture each session's official outcomes before Kalshi
#  drops them (settled 15-minute markets 404 within weeks — verified
#  2026-08-19 against the June recordings, which are now unrecoverable).
#
#  It runs the bot in fixed-length segments rather than one endless process:
#    * a crash costs one segment, not the whole run
#    * each segment's recording is a self-contained, enrichable unit
#    * `replay enrich` runs between segments, while outcomes still exist
#
#  Usage:
#     scripts/collector.sh                  # default 6h segments, forever
#     SEGMENT_SEC=3600 scripts/collector.sh # 1h segments
#     TRADE=0 scripts/collector.sh          # observe only, no paper fills
#
#  Stop with:  pkill -f collector.sh   (then pkill -f 'main.py run')
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

SEGMENT_SEC="${SEGMENT_SEC:-21600}"      # 6 hours
RESTART_DELAY_SEC="${RESTART_DELAY_SEC:-15}"
MAX_RESTART_DELAY_SEC="${MAX_RESTART_DELAY_SEC:-300}"
TRADE="${TRADE:-1}"
TAG="${TAG:-collector}"
PY="${PY:-$ROOT/.venv/bin/python}"
LOG="$ROOT/logs/collector.log"
LOCK="$ROOT/logs/collector.lock"
STATUS="$ROOT/logs/collector_status.json"

mkdir -p "$ROOT/logs"

# Single instance. Two collectors would interleave recordings and double the
# venue WS load for no extra data — there is still only one market open.
if [ -e "$LOCK" ]; then
  if kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "collector already running (pid $(cat "$LOCK")); exiting" >&2
    exit 1
  fi
  rm -f "$LOCK"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"; pkill -P $$ -f "main.py run" 2>/dev/null; exit 0' INT TERM EXIT

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

TRADE_FLAG=""
[ "$TRADE" = "1" ] && TRADE_FLAG="--trade"

segment=0
delay="$RESTART_DELAY_SEC"

log "collector starting: segment=${SEGMENT_SEC}s trade=${TRADE} tag=${TAG}"

while true; do
  segment=$((segment + 1))
  before="$(ls -1 data/recordings 2>/dev/null | wc -l | tr -d ' ')"
  log "segment $segment starting (${SEGMENT_SEC}s)"

  start_ts=$(date +%s)
  "$PY" main.py run --headless $TRADE_FLAG \
      --duration "$SEGMENT_SEC" --tag "$TAG" >> "$LOG" 2>&1
  rc=$?
  elapsed=$(( $(date +%s) - start_ts ))
  log "segment $segment exited rc=$rc after ${elapsed}s"

  # Capture official settlement results for the session just recorded. This
  # is time-critical: Kalshi purges these markets, and once purged the only
  # settlement left is our own BRTI TWAP reconstruction.
  newest="$(ls -1t data/recordings 2>/dev/null | grep -v '^sessions.json$' | head -1)"
  if [ -n "$newest" ]; then
    log "enriching $newest"
    "$PY" main.py replay enrich "$newest" >> "$LOG" 2>&1 || log "enrich failed for $newest"
  fi

  after="$(ls -1 data/recordings 2>/dev/null | wc -l | tr -d ' ')"
  cat > "$STATUS" <<EOF
{
  "updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "segments_completed": $segment,
  "last_segment_seconds": $elapsed,
  "last_exit_code": $rc,
  "last_session": "${newest:-}",
  "recordings_before": ${before:-0},
  "recordings_after": ${after:-0},
  "segment_seconds": $SEGMENT_SEC,
  "trade": $TRADE
}
EOF

  # A segment that ran nearly its full duration is a normal rollover, so
  # restart promptly. One that died early is a real failure — back off, so a
  # persistent problem (expired credentials, network down) does not turn into
  # a tight reconnect loop against Kalshi.
  if [ "$elapsed" -ge $(( SEGMENT_SEC * 9 / 10 )) ]; then
    delay="$RESTART_DELAY_SEC"
  else
    log "segment ended early — backing off ${delay}s"
    sleep "$delay"
    delay=$(( delay * 2 ))
    [ "$delay" -gt "$MAX_RESTART_DELAY_SEC" ] && delay="$MAX_RESTART_DELAY_SEC"
    continue
  fi
  sleep "$RESTART_DELAY_SEC"
done
