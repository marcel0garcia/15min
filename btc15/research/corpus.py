"""Loading recorded sessions into a form the replay engine can re-price.

A decision row is the unit of the corpus: one market, one scan, carrying
the spot, the strike, the seconds remaining, and (since 2026-08-19) the
top of both sides of the book. That is everything needed to re-derive a
decision under a different configuration — which is the whole point.

## What the corpus can and cannot contain

Rows must carry `spot` and `strike` to be re-priceable. Sessions recorded
before those fields existed are unusable no matter how large they are,
and Kalshi purges settled 15-minute markets within weeks, so their
outcomes cannot be recovered either. Both were checked on 2026-08-19: the
one historical session with cached outcomes (01JUN23:24, 12 markets)
predates `strike`, and every later session's tickers 404 on the API. The
corpus therefore starts empty and is built by running the bot.

## Outcomes

Kalshi's official result is authoritative and comes from the results
cache (`replay enrich`). When it is missing we fall back to the same
instrument the contract actually settles on — the mean of the final 60
seconds of BRTI, computed from our own recorded ticks. That fallback is
honest but it is *our* reconstruction, so every observation records which
source it used and the scorer can be asked to use official results only.

**Enrich promptly.** Official outcomes are only fetchable for a few weeks
after settlement. A recording whose outcomes were never fetched degrades
to TWAP-only once Kalshi drops the market.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

SETTLEMENT_WINDOW_SEC = 60.0
# Below this many BRTI samples in the final minute we refuse to synthesize
# an outcome — a thin window is a reconstruction artifact, not a settlement.
MIN_TWAP_SAMPLES = 20

# How close to the strike our own TWAP may land before we stop trusting it
# to decide an outcome.
#
# Measured 2026-08-19 against 16 markets with official Kalshi results: our
# 3-venue reconstruction agreed on 15 (94%). The single miss settled $1.94
# from the strike — a reconstruction error of ~0.003% on a $68.7k index,
# which is excellent accuracy and still enough to flip the answer. Every
# market that settled more than $21 from its strike agreed.
#
# So near-strike TWAP outcomes are coin flips, and a coin flip used as a
# training label is worse than no label: it teaches the sweep noise with
# the authority of data. Inside this band we return no outcome and leave
# the market unscored unless Kalshi's official result is available.
MIN_TWAP_MARGIN_USD = 25.0


@dataclass
class Frame:
    """One market, one scan — replayable."""
    ts: float
    ticker: str
    strike: float
    spot: float
    secs: float
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    book: Optional[dict] = None      # {"yes_bids": [[px, qty]...], "yes_asks": [...]}

    def book_dicts(self) -> dict:
        """The {price: qty} shape PaperBroker walks.

        With no recorded depth we fall back to top-of-book with a nominal
        size. That is *not* equivalent: a depth-less replay cannot show
        slippage or partial fills, so it flatters large orders. Sessions
        recorded this way are flagged so results can say so out loud.
        """
        if self.book:
            return {
                "yes_bids": {float(px): float(q) for px, q in self.book.get("yes_bids", [])},
                "yes_asks": {float(px): float(q) for px, q in self.book.get("yes_asks", [])},
            }
        return {
            "yes_bids": {float(self.yes_bid): 1e9} if self.yes_bid else {},
            "yes_asks": {float(self.yes_ask): 1e9} if self.yes_ask else {},
        }


@dataclass
class LoadedSession:
    session_id: str
    frames: list[Frame]
    ticks: list[tuple[float, float]]       # (ts, BRTI spot), ascending — the vol history
    close_ts: dict[str, float]             # ticker -> settlement timestamp
    strikes: dict[str, float]
    has_depth: bool
    n_rows: int
    n_skipped: int
    # 'brti' means the recording carried the engine-cadence tick stream, so a
    # replay sees exactly the series the live vol nowcast saw. 'decisions'
    # means we fell back to one spot per scan, which reads sigma lower
    # because microstructure noise is sampled away.
    tick_source: str = "decisions"
    meta: dict = field(default_factory=dict)

    @property
    def tickers(self) -> list[str]:
        return sorted(self.close_ts)

    @property
    def span_sec(self) -> float:
        return (self.frames[-1].ts - self.frames[0].ts) if len(self.frames) > 1 else 0.0


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


def load_session(session_dir: Path, *, tick_resolution_sec: float = 0.5) -> LoadedSession:
    """Read one recorded session into replayable frames.

    Rows lacking `spot` or `strike` are counted and dropped: they come from
    a pre-v3 schema and cannot be re-priced at all.
    """
    frames: list[Frame] = []
    ticks_by_bucket: dict[int, tuple[float, float]] = {}
    close_samples: dict[str, list[float]] = defaultdict(list)
    strikes: dict[str, list[float]] = defaultdict(list)
    n_rows = n_skipped = 0
    has_depth = False

    for row in _iter_jsonl(session_dir / "decisions.jsonl"):
        n_rows += 1
        ticker = row.get("ticker")
        spot = row.get("spot")
        strike = row.get("strike")
        ts = row.get("ts")
        secs = row.get("secs")
        if secs is None:
            secs = row.get("secs_remaining")      # pre-v3 field name
        if not ticker or ts is None or secs is None or not spot or not strike:
            n_skipped += 1
            continue

        book = row.get("book")
        if book:
            has_depth = True
        frames.append(Frame(
            ts=float(ts), ticker=ticker, strike=float(strike), spot=float(spot),
            secs=float(secs),
            yes_bid=(float(row["yes_bid"]) if row.get("yes_bid") is not None else None),
            yes_ask=(float(row["yes_ask"]) if row.get("yes_ask") is not None else None),
            book=book,
        ))
        # One BRTI series for the whole session: only one market is open at
        # a time, so overlapping rows carry the same spot.
        bucket = int(float(ts) / tick_resolution_sec)
        ticks_by_bucket.setdefault(bucket, (float(ts), float(spot)))
        close_samples[ticker].append(float(ts) + float(secs))
        strikes[ticker].append(float(strike))

    frames.sort(key=lambda f: f.ts)

    # Prefer the engine-cadence BRTI stream when the recording has it: the
    # vol nowcast is sensitive to sampling rate, so replaying off a 1 Hz
    # reconstruction of a 4 Hz series changes sigma and therefore changes
    # decisions.
    tick_source = "decisions"
    brti_path = session_dir / "brti_ticks.jsonl"
    brti_ticks: list[tuple[float, float]] = []
    for row in _iter_jsonl(brti_path):
        ts_, mid = row.get("ts"), row.get("mid")
        if ts_ is not None and mid:
            brti_ticks.append((float(ts_), float(mid)))
    if len(brti_ticks) > len(ticks_by_bucket):
        brti_ticks.sort()
        ticks = brti_ticks
        tick_source = "brti"
    else:
        ticks = [ticks_by_bucket[b] for b in sorted(ticks_by_bucket)]

    meta_path = session_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}

    return LoadedSession(
        session_id=session_dir.name,
        frames=frames,
        ticks=ticks,
        # Median, not mean: a single scan delayed by a GC pause or a slow
        # REST call would drag the mean and shift the settlement window.
        close_ts={t: statistics.median(v) for t, v in close_samples.items()},
        strikes={t: statistics.median(v) for t, v in strikes.items()},
        has_depth=has_depth,
        n_rows=n_rows,
        n_skipped=n_skipped,
        tick_source=tick_source,
        meta=meta,
    )


def current_core_fingerprint() -> str:
    """Fingerprint of the decision code running right now."""
    from btc15.recording.session import _core_fingerprint
    return _core_fingerprint()


def load_results_cache(path: Path) -> dict[str, str]:
    """ticker -> 'yes'|'no' for markets Kalshi has finalized."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return {}
    return {
        t: rec["result"]
        for t, rec in raw.items()
        if isinstance(rec, dict) and rec.get("status") == "finalized" and rec.get("result")
    }


def settlement_twap(
    ticks: list[tuple[float, float]], close_ts: float,
    window_sec: float = SETTLEMENT_WINDOW_SEC,
) -> Optional[float]:
    """Mean BRTI over the final `window_sec` before close — the instrument
    KXBTC15M actually settles on."""
    window = [p for t, p in ticks if close_ts - window_sec <= t <= close_ts and p > 0]
    if len(window) < MIN_TWAP_SAMPLES:
        return None
    return sum(window) / len(window)


def resolve_outcomes(
    sess: LoadedSession,
    official: dict[str, str],
    *,
    allow_twap_fallback: bool = True,
    min_twap_margin_usd: float = MIN_TWAP_MARGIN_USD,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (ticker -> 'yes'/'no', ticker -> source).

    Source is 'official' or 'twap'. Markets that can be resolved by neither
    are absent — a replay must never invent a settlement, which is exactly
    the class of bug that produced the phantom -100% losses in the old logs.

    A TWAP landing within `min_twap_margin_usd` of the strike is treated as
    unresolved rather than guessed; see MIN_TWAP_MARGIN_USD for the
    measurement behind that number.
    """
    outcomes: dict[str, str] = {}
    source: dict[str, str] = {}
    for ticker, close in sess.close_ts.items():
        if ticker in official:
            outcomes[ticker] = official[ticker]
            source[ticker] = "official"
            continue
        if not allow_twap_fallback:
            continue
        twap = settlement_twap(sess.ticks, close)
        if twap is None:
            continue
        strike = sess.strikes[ticker]
        if abs(twap - strike) < min_twap_margin_usd:
            continue        # too close to call with our own reconstruction
        outcomes[ticker] = "yes" if twap >= strike else "no"
        source[ticker] = "twap"
    return outcomes, source


def session_start_ts(session_dir: Path) -> float:
    """When the session actually began.

    Ordering sessions by directory mtime is wrong, and wrong in a way that
    is easy to miss: a directory's mtime advances when a file is created or
    removed inside it, NOT when an existing file is appended to. A session
    that runs for six hours writing into files it opened at startup keeps
    the mtime it had in its first second, so a long old session can sort
    after a short new one. That mis-ordering silently corrupted the sweep's
    `--holdout` split, which is supposed to be chronological.

    meta.json's start_ts is written by the recorder and is authoritative.
    The first decision row is the fallback; mtime is the last resort.
    """
    meta = session_dir / "meta.json"
    if meta.exists():
        try:
            ts = json.loads(meta.read_text()).get("start_ts")
            if ts:
                return float(ts)
        except Exception:
            pass
    head = next(_iter_jsonl(session_dir / "decisions.jsonl"), None)
    if head and head.get("ts"):
        try:
            return float(head["ts"])
        except (TypeError, ValueError):
            pass
    try:
        return session_dir.stat().st_mtime
    except OSError:
        return 0.0


def discover_sessions(
    root: Path, *, min_rows: int = 1, require_depth: bool = False,
) -> list[Path]:
    """Session directories that hold re-priceable decision rows, oldest first.

    Chronological by session start — see session_start_ts for why that is
    not the same as sorting by mtime.
    """
    if not root.exists():
        return []
    out: list[tuple[float, Path]] = []
    for d in root.iterdir():
        if not d.is_dir():
            continue
        p = d / "decisions.jsonl"
        if not p.exists() or p.stat().st_size == 0:
            continue
        head = next(_iter_jsonl(p), None)
        if not head or head.get("spot") is None or head.get("strike") is None:
            continue                                  # pre-v3 schema
        if require_depth and head.get("book") is None:
            continue
        if sum(1 for _ in open(p)) < min_rows:
            continue
        out.append((session_start_ts(d), d))
    return [d for _, d in sorted(out, key=lambda kv: kv[0])]
