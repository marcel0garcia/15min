"""v2 Phase 1 telemetry recorder.

Ambient capture layer — co-runs with the strategy engine and records:
  - Kalshi raw WS frames (orderbook + trades)
  - Venue top-of-book ticks (Coinbase / Kraken / Bitstamp)
  - Decision log (every scan, including no-action branches)

Live writes are JSONL sidecars under `data/recordings/{session_id}/`. Run
`python main.py replay convert <session_id>` to consolidate into Parquet for
offline analysis.

Never places orders. Gate via `recording.enabled: false` in config.yaml to
disable entirely.
"""

from btc15.recording.session import SessionRecorder, BRAIN_VERSION

__all__ = ["SessionRecorder", "BRAIN_VERSION"]
