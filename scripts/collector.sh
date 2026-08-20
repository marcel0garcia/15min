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

  # Capture official settlement results. Time-critical: Kalshi purges settled
  # 15-minute markets within weeks, and once purged the only settlement left
  # is our own BRTI TWAP reconstruction.
  #
  # Enrich the last few sessions, not just the one that ended, for two
  # reasons found the hard way on 2026-08-20:
  #
  #   1. `ls -t` on data/recordings sorts by DIRECTORY mtime, which does not
  #      advance while a session merely appends to files it already created.
  #      It named a 12-hour-old session as "newest" and a whole 6-hour
  #      segment went unenriched (16 markets, 1 outcome captured).
  #   2. A segment's last markets have not settled when it ends, and Kalshi
  #      finalizes results with a lag. One pass at the wrong moment misses
  #      them permanently.
  #
  # `replay enrich` skips tickers already finalized in the cache, so
  # re-running it over recent sessions is cheap and idempotent. Session ids
  # come from sessions.json (ordered by start_ts), which the recorder writes
  # — an authoritative source rather than an inference from the filesystem.
  recent="$("$PY" - <<'PYEOF' 2>/dev/null
import json
from pathlib import Path
p = Path("data/recordings/sessions.json")
try:
    rows = json.loads(p.read_text())
except Exception:
    rows = []
seen, out = set(), []
for m in sorted(rows, key=lambda r: r.get("start_ts") or 0, reverse=True):
    sid = m.get("session_id")
    if sid and sid not in seen and (Path("data/recordings") / sid).is_dir():
        seen.add(sid)
        out.append(sid)
    if len(out) >= 4:
        break
print("\n".join(out))
PYEOF
)"
  for sid in $recent; do
    log "enriching $sid"
    "$PY" main.py replay enrich "$sid" >> "$LOG" 2>&1 || log "enrich failed for $sid"
  done
  newest="$(echo "$recent" | head -1)"

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
