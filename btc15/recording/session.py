"""Session recorder — owns session_id, JSONL writers, and meta index.

Sync line-buffered writes with periodic flush. At ~50-100 msg/s aggregate
across all streams, sync I/O on the engine's event loop is negligible
(microseconds per write). If profiling later shows blocking, swap to a
background asyncio.Queue + writer task per stream.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# Marker for "which brain produced these decisions". Bump when the strategy's
# probability source changes (e.g. when Phase 3 swaps the ensemble for the
# fair-value engine). Lets replay/analyze segregate sessions by brain era.
BRAIN_VERSION = "ensemble-v1-phase1"


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode().strip()
    except Exception:
        return None


_SECRET_KEYS = {"api_key", "rsa_key_path", "password", "email"}


def _config_hash(cfg) -> str:
    """Stable 12-char hash of the loaded config minus secrets — tags decision rows
    so we can group by 'same config era' when comparing sessions."""

    def _to_dict(obj: Any) -> Any:
        if is_dataclass(obj):
            return {k: _to_dict(v) for k, v in asdict(obj).items()}
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items() if k not in _SECRET_KEYS}
        if isinstance(obj, (list, tuple)):
            return [_to_dict(x) for x in obj]
        return obj

    blob = json.dumps(_to_dict(cfg), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


class _JSONLWriter:
    """Append-only JSONL writer with periodic flush."""

    def __init__(self, path: Path, flush_every_sec: float = 2.0):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(path, "a", buffering=1)  # line-buffered
        self._last_flush = time.time()
        self._flush_every = flush_every_sec
        self.lines_written = 0

    def write(self, record: dict) -> None:
        try:
            self._fp.write(json.dumps(record, separators=(",", ":"), default=str))
            self._fp.write("\n")
            self.lines_written += 1
            now = time.time()
            if now - self._last_flush >= self._flush_every:
                self._fp.flush()
                self._last_flush = now
        except Exception as e:
            log.debug(f"JSONL write failed for {self.path.name}: {e}")

    def close(self) -> None:
        try:
            self._fp.flush()
            self._fp.close()
        except Exception:
            pass


class SessionRecorder:
    """Ambient telemetry recorder.

    Subscribes as an additional handler to existing WS streams (Kalshi via
    the engine's KalshiWebSocket; venues via its own connectors). Owns
    JSONL writers and the per-session meta.json. Writes are no-ops when
    `recording.enabled` is False.
    """

    def __init__(self, cfg, session_label: str):
        self.cfg = cfg
        self.enabled = bool(cfg.recording.enabled)
        self.session_label = session_label
        self.session_id: Optional[str] = None
        self.root: Optional[Path] = None
        self.kalshi_writer: Optional[_JSONLWriter] = None
        self.venue_writer: Optional[_JSONLWriter] = None
        self.decision_writer: Optional[_JSONLWriter] = None
        self.start_ts: float = 0.0
        self.config_hash: str = ""
        self.git_commit: Optional[str] = None
        self.brain_version: str = BRAIN_VERSION

        if not self.enabled:
            log.info("[REC] Recording disabled (recording.enabled=false)")
            return

        ts_suffix = str(int(time.time()))
        self.session_id = f"{session_label}_{ts_suffix}"
        self.root = Path(cfg.recording.path) / self.session_id
        self.root.mkdir(parents=True, exist_ok=True)

        self.kalshi_writer = _JSONLWriter(self.root / "kalshi_frames.jsonl")
        self.venue_writer = _JSONLWriter(self.root / "venue_ticks.jsonl")
        self.decision_writer = _JSONLWriter(self.root / "decisions.jsonl")

        self.start_ts = time.time()
        self.config_hash = _config_hash(cfg)
        self.git_commit = _git_commit()

        mode = (
            "live"
            if (cfg.strategy.auto_trade and not cfg.strategy.paper_trade)
            else "paper"
            if cfg.strategy.auto_trade
            else "signal-only"
        )
        meta = {
            "session_id": self.session_id,
            "session_label": session_label,
            "start_ts": self.start_ts,
            "start_iso": datetime.fromtimestamp(self.start_ts, timezone.utc).isoformat(),
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
            "brain_version": self.brain_version,
            "mode": mode,
            "recording_path": str(self.root),
        }
        (self.root / "meta.json").write_text(json.dumps(meta, indent=2))
        self._append_session_index(meta)
        log.info(
            f"[REC] Session {self.session_id} mode={mode} brain={self.brain_version} "
            f"git={self.git_commit} cfg_hash={self.config_hash} path={self.root}"
        )

    # ── Public write API (called by tap / venue connectors / decision log) ──

    def write_kalshi(self, record: dict) -> None:
        if self.enabled and self.kalshi_writer is not None:
            self.kalshi_writer.write(record)

    def write_venue(self, record: dict) -> None:
        if self.enabled and self.venue_writer is not None:
            self.venue_writer.write(record)

    def write_decision(self, record: dict) -> None:
        if self.enabled and self.decision_writer is not None:
            self.decision_writer.write(record)

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def close(self) -> None:
        if not self.enabled:
            return
        end_ts = time.time()
        lines = {}
        for name, w in (
            ("kalshi", self.kalshi_writer),
            ("venue", self.venue_writer),
            ("decisions", self.decision_writer),
        ):
            if w is not None:
                lines[name] = w.lines_written
                w.close()
        if self.root is not None:
            meta_path = self.root / "meta.json"
            try:
                meta = json.loads(meta_path.read_text())
                meta["end_ts"] = end_ts
                meta["end_iso"] = datetime.fromtimestamp(end_ts, timezone.utc).isoformat()
                meta["duration_sec"] = round(end_ts - self.start_ts, 1)
                meta["lines"] = lines
                meta_path.write_text(json.dumps(meta, indent=2))
                self._append_session_index(meta, replace=True)
            except Exception as e:
                log.debug(f"Could not finalize session meta: {e}")
        log.info(
            f"[REC] Session {self.session_id} closed — "
            f"K={lines.get('kalshi', 0)} V={lines.get('venue', 0)} "
            f"D={lines.get('decisions', 0)}"
        )

    # ── Internals ───────────────────────────────────────────────────────────

    def _append_session_index(self, meta: dict, replace: bool = False) -> None:
        index_path = Path(self.cfg.recording.path) / "sessions.json"
        try:
            data = json.loads(index_path.read_text()) if index_path.exists() else []
        except Exception:
            data = []
        if not isinstance(data, list):
            data = []
        if replace:
            data = [d for d in data if d.get("session_id") != meta["session_id"]]
        data.append(meta)
        try:
            index_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.debug(f"Could not write sessions index: {e}")
