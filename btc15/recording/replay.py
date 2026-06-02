"""Phase-1 replay & analysis tooling.

Two commands, both operating on a recorded session under data/recordings/{session_id}/:

  convert  Walk kalshi_frames.jsonl and reconstruct the Kalshi orderbook
           state per ticker. Round-trip test for the raw-frame capture.
           Also builds index_grid.jsonl from venue_ticks.jsonl (1Hz median
           of venue mids — Phase 2 will replace this with the real BRTI).

  analyze  Two validation analyses against decisions.jsonl:
           (a) decisions × reason_code × phase — count and prob distribution
           (b) counterfactual P&L for action='none' rows using the cached
               market settlement results (data/market_results_cache.json)
"""
from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger(__name__)


# ── JSONL loaders ────────────────────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ── Kalshi book reconstruction ───────────────────────────────────────────────

def reconstruct_books(session_dir: Path) -> dict[str, dict]:
    """Fold kalshi_frames.jsonl into per-ticker books. Returns final state."""
    books: dict[str, dict] = {}
    stats = {"snapshots": 0, "deltas": 0, "trades": 0, "skipped": 0}
    for frame in _iter_jsonl(session_dir / "kalshi_frames.jsonl"):
        kind = frame.get("kind")
        raw = frame.get("raw") or {}
        data = raw.get("msg") or {}
        ticker = data.get("market_ticker")
        if kind == "snapshot":
            stats["snapshots"] += 1
            if not ticker:
                stats["skipped"] += 1
                continue
            book = {"yes_bids": {}, "yes_asks": {}}
            for price_dollars, size_fp in data.get("yes_dollars_fp", []) or []:
                try:
                    price = int(round(float(price_dollars) * 100))
                    size = float(size_fp)
                except (TypeError, ValueError):
                    continue
                if size > 0:
                    book["yes_bids"][price] = size
            for price_dollars, size_fp in data.get("no_dollars_fp", []) or []:
                try:
                    price = int(round(float(price_dollars) * 100))
                    size = float(size_fp)
                except (TypeError, ValueError):
                    continue
                if size > 0:
                    book["yes_asks"][100 - price] = size
            books[ticker] = book
        elif kind == "delta":
            stats["deltas"] += 1
            if not ticker:
                stats["skipped"] += 1
                continue
            price_dollars = data.get("price_dollars")
            delta_fp = data.get("delta_fp")
            side = data.get("side")
            if price_dollars is None or delta_fp is None or side not in ("yes", "no"):
                stats["skipped"] += 1
                continue
            try:
                price = int(round(float(price_dollars) * 100))
                delta = float(delta_fp)
            except (TypeError, ValueError):
                stats["skipped"] += 1
                continue
            book = books.setdefault(ticker, {"yes_bids": {}, "yes_asks": {}})
            target = book["yes_bids"] if side == "yes" else book["yes_asks"]
            key = price if side == "yes" else 100 - price
            new_size = target.get(key, 0.0) + delta
            if new_size <= 0:
                target.pop(key, None)
            else:
                target[key] = new_size
        elif kind == "trade":
            stats["trades"] += 1
    books["__stats__"] = stats
    return books


def top_of_book(book: dict) -> tuple[Optional[int], Optional[int]]:
    """Return (yes_bid_cents, yes_ask_cents) — the highest YES bid and lowest YES ask."""
    yes_bids = book.get("yes_bids") or {}
    yes_asks = book.get("yes_asks") or {}
    bid = max(yes_bids.keys()) if yes_bids else None
    ask = min(yes_asks.keys()) if yes_asks else None
    return bid, ask


# ── Index grid (Phase 2 BRTI reconstruction) ─────────────────────────────────

def build_index_grid(
    session_dir: Path,
    interval_sec: float = 1.0,
    staleness_sec: float = 5.0,
    n_min: int = 2,
    k_mad: float = 3.0,
) -> list[dict]:
    """Build a 1Hz BRTI reconstruction grid from venue_ticks.jsonl.

    Delegates the algorithm to btc15.recording.brti (CF Benchmarks-style
    median-of-venue-mids with MAD outlier rejection). Writes both
    index_grid.jsonl (per-row) and stability_report.json (summary) next to
    the session.
    """
    from btc15.recording.brti import build_grid, stability_report

    venue_path = session_dir / "venue_ticks.jsonl"
    if not venue_path.exists():
        return []

    grid_rows = build_grid(
        _iter_jsonl(venue_path),
        grid_interval_sec=interval_sec,
        staleness_sec=staleness_sec,
        n_min=n_min,
        k_mad=k_mad,
    )

    out_path = session_dir / "index_grid.jsonl"
    with open(out_path, "w") as f:
        for row in grid_rows:
            f.write(json.dumps(row.to_dict(), separators=(",", ":")) + "\n")

    report = stability_report(grid_rows)
    from dataclasses import asdict
    (session_dir / "stability_report.json").write_text(
        json.dumps(asdict(report), indent=2)
    )

    return [r.to_dict() for r in grid_rows]


# ── Convert subcommand ──────────────────────────────────────────────────────

def cmd_convert(session_id: str, recordings_root: Path) -> dict:
    session_dir = recordings_root / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    log.info(f"[REPLAY-CONVERT] {session_id}")

    books = reconstruct_books(session_dir)
    stats = books.pop("__stats__", {})

    book_summary = {}
    for ticker, book in books.items():
        bid, ask = top_of_book(book)
        book_summary[ticker] = {
            "yes_bid": bid,
            "yes_ask": ask,
            "n_yes_bid_levels": len(book.get("yes_bids", {})),
            "n_yes_ask_levels": len(book.get("yes_asks", {})),
        }

    grid = build_index_grid(session_dir)

    summary = {
        "session_id": session_id,
        "kalshi": {
            "snapshots": stats.get("snapshots", 0),
            "deltas": stats.get("deltas", 0),
            "trades": stats.get("trades", 0),
            "skipped": stats.get("skipped", 0),
            "tickers_reconstructed": len(book_summary),
        },
        "top_of_book_per_ticker": book_summary,
        "index_grid": {
            "rows": len(grid),
            "first_ts": grid[0]["ts"] if grid else None,
            "last_ts": grid[-1]["ts"] if grid else None,
            "venues_seen": sorted({v for r in grid for v in r["venues"]}),
        },
    }
    (session_dir / "convert_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ── Analyze subcommand ──────────────────────────────────────────────────────

def _load_results_cache(path: Path) -> dict[str, str]:
    """ticker -> 'yes'|'no' (only finalized markets)"""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    out = {}
    for ticker, rec in raw.items():
        if isinstance(rec, dict) and rec.get("status") == "finalized":
            result = rec.get("result")
            if result in ("yes", "no"):
                out[ticker] = result
    return out


# ── Settlement enrichment (cmd_enrich) ───────────────────────────────────────
# Populates market_results_cache.json with settlement outcomes for every
# ticker observed in decisions.jsonl. The engine's _check_settlements loop
# only fetches results for markets the bot held positions in; counterfactual
# analysis on `action='none'` rows needs results for the SUPERSET of tickers
# the bot saw but skipped.

async def cmd_enrich_results(
    session_id: str,
    recordings_root: Path,
    cache_path: Path,
    cfg,
) -> dict:
    """Fetch Kalshi settlement results for any tickers in decisions.jsonl
    that aren't already finalized in the cache. Updates cache_path in place.
    """
    from btc15.kalshi.client import KalshiClient

    session_dir = recordings_root / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    log.info(f"[REPLAY-ENRICH] {session_id}")

    # Load existing cache (defensive — missing/corrupt → empty).
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
            if not isinstance(cache, dict):
                cache = {}
        except json.JSONDecodeError:
            cache = {}
    else:
        cache = {}

    # Collect unique tickers from decisions.
    tickers: set[str] = set()
    for d in _iter_jsonl(session_dir / "decisions.jsonl"):
        t = d.get("ticker")
        if t:
            tickers.add(t)

    to_fetch = sorted(
        t for t in tickers
        if cache.get(t, {}).get("status") != "finalized"
    )

    fetched = 0
    finalized = 0
    failed = 0

    async with KalshiClient(cfg.kalshi) as client:
        for ticker in to_fetch:
            try:
                m = await client.get_market(ticker)
            except Exception as e:
                log.debug(f"enrich: get_market({ticker}) failed: {e}")
                failed += 1
                continue
            fetched += 1
            status = m.status.value if hasattr(m.status, "value") else str(m.status)
            close_time = (
                m.close_time.isoformat()
                if hasattr(m.close_time, "isoformat") else None
            )
            cache[ticker] = {
                "result": m.result if isinstance(m.result, str) else None,
                "status": status,
                "close_time": close_time,
                "fetched_at": datetime.now().isoformat(),
            }
            if status == "finalized":
                finalized += 1

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))

    return {
        "session_id": session_id,
        "unique_tickers_in_decisions": len(tickers),
        "already_finalized_in_cache": len(tickers) - len(to_fetch),
        "fetched": fetched,
        "newly_finalized": finalized,
        "failed": failed,
        "cache_size_total": len(cache),
        "cache_path": str(cache_path),
    }


def cmd_analyze(
    session_id: str,
    recordings_root: Path,
    results_cache_path: Path,
) -> dict:
    session_dir = recordings_root / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_dir}")

    log.info(f"[REPLAY-ANALYZE] {session_id}")

    decisions = list(_iter_jsonl(session_dir / "decisions.jsonl"))
    results = _load_results_cache(results_cache_path)

    by_code_phase: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in decisions:
        key = (d.get("reason_code", "UNKNOWN"), d.get("phase", "?"))
        by_code_phase[key].append(d)

    code_summary = {}
    for (code, phase), rows in sorted(by_code_phase.items()):
        probs = [r["prob_yes"] for r in rows if r.get("prob_yes") is not None]
        confs = [r["confidence"] for r in rows if r.get("confidence") is not None]
        code_summary[f"{code}|{phase}"] = {
            "count": len(rows),
            "mean_prob_yes": round(statistics.mean(probs), 4) if probs else None,
            "mean_confidence": round(statistics.mean(confs), 4) if confs else None,
            "n_with_recommended_side": sum(1 for r in rows if r.get("recommended_side")),
        }

    cf_buckets: dict[str, dict] = defaultdict(lambda: {
        "n_total": 0, "n_resolved": 0, "wins": 0, "losses": 0,
        "hypothetical_pnl_cents": 0.0,
    })
    for d in decisions:
        if d.get("action") != "none":
            continue
        rec = d.get("recommended_side")
        ticker = d.get("ticker")
        mid = d.get("kalshi_mid")
        if rec not in ("yes", "no") or not ticker or mid is None:
            continue
        bucket = cf_buckets[d.get("reason_code", "UNKNOWN")]
        bucket["n_total"] += 1
        outcome = results.get(ticker)
        if outcome is None:
            continue
        bucket["n_resolved"] += 1
        if outcome == rec:
            bucket["wins"] += 1
            bucket["hypothetical_pnl_cents"] += (100.0 - float(mid))
        else:
            bucket["losses"] += 1
            bucket["hypothetical_pnl_cents"] += -float(mid)

    cf_summary = {}
    for code, b in sorted(cf_buckets.items()):
        n = b["n_resolved"]
        cf_summary[code] = {
            "n_decisions": b["n_total"],
            "n_resolved": n,
            "win_rate": round(b["wins"] / n, 3) if n else None,
            "mean_hypothetical_pnl_cents": round(b["hypothetical_pnl_cents"] / n, 2) if n else None,
            "total_hypothetical_pnl_dollars": round(b["hypothetical_pnl_cents"] / 100.0, 2),
        }

    summary = {
        "session_id": session_id,
        "decisions_total": len(decisions),
        "by_reason_code_phase": code_summary,
        "counterfactual_none": cf_summary,
    }
    (session_dir / "analyze_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
