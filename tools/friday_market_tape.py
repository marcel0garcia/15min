"""Pull and analyze Kalshi market trade tapes for 15-min BTC markets.

For any settled market, fetches the full second-level trade history from
Kalshi's public /markets/trades endpoint and reconstructs the YES-side
price curve over the market's 15-minute lifetime. Overlays the bot's
entry/exit timestamps from logs/trades.csv so we can see what the price
did before, during, and after each position.

Cached to logs/friday_snapshot/market_tapes/<ticker>.json so re-runs are
free. Re-fetches only tickers that aren't cached.

Usage:
  # Single ticker — timeline + bot overlay
  python tools/friday_market_tape.py --ticker KXBTC15M-26MAY190245-45

  # All tickers in a session — summary stats
  python tools/friday_market_tape.py --session 18MAY22:30 --summary

  # Just shaken-out tickers — find the would-have-recovered moments
  python tools/friday_market_tape.py --session 18MAY22:30 --filter shaken_out

  # Force re-fetch (default reads cache when present)
  python tools/friday_market_tape.py --ticker KXBTC15M-... --refresh
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import re

import httpx


def _parse_iso(s: str) -> datetime:
    """Robust ISO parse — Kalshi sometimes returns 4-5 microsecond digits."""
    s = s.replace("Z", "+00:00")
    # Normalize microseconds to 6 digits (fromisoformat is strict)
    s = re.sub(r"\.(\d{1,6})(?=[+-])", lambda m: "." + m.group(1).ljust(6, "0"), s)
    return datetime.fromisoformat(s)

ROOT = Path(__file__).resolve().parent.parent
TRADES_CSV = ROOT / "logs" / "friday_snapshot" / "trades.csv"
CROSS_VALIDATE_JSON = ROOT / "logs" / "friday_snapshot" / "cross_validate.json"
TAPE_DIR = ROOT / "logs" / "friday_snapshot" / "market_tapes"
TAPE_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
REQ_DELAY_SEC = 0.2  # 5 req/s
PAGE_LIMIT = 1000  # Kalshi's max per page


# ─── Ticker parsing ───────────────────────────────────────────────────────────

def parse_close_time_utc(ticker: str) -> datetime | None:
    """Decode close time from 15-min BTC ticker. Ticker time is EDT — add 4h."""
    try:
        suffix = ticker.split("-")[1]  # e.g. "26MAY190245" in KXBTC15M-26MAY190245-45
        yy = 2000 + int(suffix[:2])
        mon = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
               "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}[suffix[2:5]]
        dd = int(suffix[5:7])
        hh = int(suffix[7:9])
        mm = int(suffix[9:11])
        return (datetime(yy, mon, dd, hh, mm, tzinfo=timezone.utc)
                + timedelta(hours=4))
    except Exception:
        return None


# ─── Trade tape fetch ─────────────────────────────────────────────────────────

def fetch_tape(ticker: str, refresh: bool = False) -> list[dict]:
    """Fetch the full trade tape for a ticker, paginating through Kalshi.
    Returns a list of trades sorted oldest → newest. Caches to JSON.
    """
    cache_path = TAPE_DIR / f"{ticker}.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())["trades"]

    client = httpx.Client(timeout=15.0, headers={"accept": "application/json"})
    trades: list[dict] = []
    cursor: str | None = None
    pages = 0
    try:
        while True:
            params: dict = {"ticker": ticker, "limit": PAGE_LIMIT}
            if cursor:
                params["cursor"] = cursor
            r = client.get(f"{BASE}/markets/trades", params=params)
            if r.status_code != 200:
                print(f"  ! {ticker}: HTTP {r.status_code}")
                break
            data = r.json()
            page = data.get("trades", [])
            trades.extend(page)
            cursor = data.get("cursor") or None
            pages += 1
            if not cursor or not page:
                break
            time.sleep(REQ_DELAY_SEC)
    finally:
        client.close()

    # Sort oldest → newest
    trades.sort(key=lambda t: t.get("created_time", ""))

    cache_path.write_text(json.dumps({
        "ticker": ticker,
        "fetched_at": datetime.utcnow().isoformat(),
        "page_count": pages,
        "trade_count": len(trades),
        "trades": trades,
    }, indent=2))
    return trades


# ─── Bot trade overlay ────────────────────────────────────────────────────────

def load_bot_trades_for_ticker(ticker: str) -> list[dict]:
    """Pull the bot's entries/exits for a ticker from trades.csv."""
    rows = []
    with open(TRADES_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("ticker") != ticker:
                continue
            try:
                ts = _parse_iso(row["timestamp"])
            except (ValueError, KeyError):
                continue
            rows.append({
                "ts": ts,
                "side": row["side"],
                "contracts": int(row["contracts"]),
                "price_cents": int(row["price_cents"]),
                "source": row["source"],
                "session": row["session"],
                "trade_id": row["trade_id"],
            })
    rows.sort(key=lambda r: r["ts"])
    return rows


# ─── Tape → minute-bucket aggregation ─────────────────────────────────────────

def bucket_by_minute(trades: list[dict], close_utc: datetime) -> dict[int, dict]:
    """Group trades by minute-bucket (seconds remaining at minute boundary).
    Returns {minute_bucket: {prices: [int], volume: float, count: int}}.
    """
    buckets: dict[int, dict] = defaultdict(
        lambda: {"prices": [], "volume": 0.0, "count": 0}
    )
    for t in trades:
        try:
            t_ts = _parse_iso(t["created_time"])
            yes_cents = int(round(float(t["yes_price_dollars"]) * 100))
            size = float(t["count_fp"])
        except (ValueError, KeyError):
            continue
        secs_left = (close_utc - t_ts).total_seconds()
        if secs_left < -60 or secs_left > 1200:
            continue  # outside the 15-min window (with small buffer)
        bucket = int(secs_left // 60) * 60  # bucket at minute boundaries
        b = buckets[bucket]
        b["prices"].append(yes_cents)
        b["volume"] += size
        b["count"] += 1
    return buckets


# ─── Single-ticker timeline render ────────────────────────────────────────────

def render_timeline(
    ticker: str,
    trades: list[dict],
    bot_events: list[dict],
    settlement_result: str | None,
) -> str:
    """Plain-text timeline of YES price by minute, with bot overlay."""
    close_utc = parse_close_time_utc(ticker)
    if not close_utc:
        return f"  ! Unable to parse close time for {ticker}"

    buckets = bucket_by_minute(trades, close_utc)

    lines = []
    lines.append("=" * 78)
    lines.append(f"MARKET TAPE: {ticker}")
    lines.append(f"  Close: {close_utc.isoformat()}  |  Settled: {settlement_result or '?'}"
                 f"  |  {len(trades):,} trades")
    lines.append("=" * 78)

    # Bot summary
    if bot_events:
        lines.append("\n[BOT activity]")
        for ev in bot_events:
            t_minus = (close_utc - ev["ts"]).total_seconds()
            lines.append(f"  {ev['ts'].isoformat()}  t-{t_minus:>4.0f}s  "
                         f"{ev['side']:14} ×{ev['contracts']:<3} @ {ev['price_cents']:>3}¢  "
                         f"  {ev['source']}")

    # Build per-minute table with bot markers
    lines.append("\n[YES price by minute]")
    lines.append(f"  {'t-':>6}  {'open':>5} {'high':>5} {'low':>5} {'close':>5}  "
                 f"{'vol':>7}  {'n':>4}   bot")

    bot_by_minute = defaultdict(list)
    for ev in bot_events:
        t_minus = (close_utc - ev["ts"]).total_seconds()
        bucket = int(t_minus // 60) * 60
        bot_by_minute[bucket].append(ev)

    for bucket in sorted(buckets.keys(), reverse=True):
        b = buckets[bucket]
        if not b["prices"]:
            continue
        prices = b["prices"]
        markers = ""
        for ev in bot_by_minute.get(bucket, []):
            side_short = "ENT" if not ev["side"].endswith("_exit") else "EXT"
            markers += f"  ←{side_short} {ev['side']:<10}@{ev['price_cents']}¢"
        lines.append(
            f"  t-{bucket:>4}s  {prices[0]:>4}¢ {max(prices):>4}¢ {min(prices):>4}¢ "
            f"{prices[-1]:>4}¢  {b['volume']:>7.1f}  {b['count']:>4}{markers}"
        )

    # Verdict for this market
    if bot_events:
        entries = [ev for ev in bot_events if not ev["side"].endswith("_exit")]
        exits = [ev for ev in bot_events if ev["side"].endswith("_exit")]
        if entries and exits:
            ent_side = entries[0]["side"]  # "yes" or "no"
            ent_qty = sum(e["contracts"] for e in entries)
            ent_cost = sum(e["contracts"] * e["price_cents"] / 100 for e in entries)
            avg_entry = ent_cost / max(ent_qty, 1) * 100  # back to cents
            exit_qty = sum(e["contracts"] for e in exits)
            exit_value = sum(e["contracts"] * e["price_cents"] / 100 for e in exits)
            realized = exit_value - ent_cost
            # If held to settlement instead, what would PnL have been?
            if settlement_result == ent_side:
                hold_value = ent_qty * 1.00
            elif settlement_result in ("yes", "no"):
                hold_value = 0.0
            else:
                hold_value = None
            lines.append("\n[Verdict]")
            lines.append(f"  Bot was on side: {ent_side.upper()}  "
                         f"(market settled {settlement_result})")
            lines.append(f"  Avg entry: {avg_entry:.0f}¢  ×{ent_qty}  "
                         f"cost ${ent_cost:.2f}")
            lines.append(f"  Realized PnL: ${realized:+.2f}")
            if hold_value is not None:
                hypothetical = hold_value - ent_cost
                lines.append(
                    f"  Held-to-settle PnL: ${hypothetical:+.2f}"
                    f"  → shake-out cost ${hypothetical - realized:+.2f}"
                )

    return "\n".join(lines)


# ─── Session-level analysis ───────────────────────────────────────────────────

def analyze_session(
    session_tag: str,
    filter_class: str | None = None,
    refresh: bool = False,
) -> None:
    """Pull tapes for every position in a session and produce aggregate stats."""
    if not CROSS_VALIDATE_JSON.exists():
        print(f"  ! {CROSS_VALIDATE_JSON} missing — run friday_cross_validate.py first")
        return
    data = json.loads(CROSS_VALIDATE_JSON.read_text())
    if session_tag not in data:
        print(f"  ! session {session_tag} not in cross_validate.json")
        print(f"    Available: {', '.join(sorted(data.keys()))}")
        return

    positions = data[session_tag]["positions"]
    if filter_class:
        positions = [p for p in positions if p.get("class") == filter_class]
    if not positions:
        print(f"  ! no positions matching filter")
        return

    print(f"\nSession {session_tag} — analyzing {len(positions)} positions"
          + (f" (filter: {filter_class})" if filter_class else "")
          + "\n")

    # Fetch tapes for all unique tickers
    tickers = sorted(set(p["ticker"] for p in positions))
    print(f"  Fetching tapes for {len(tickers)} tickers...")
    tape_data = {}
    for i, tk in enumerate(tickers, 1):
        cache_path = TAPE_DIR / f"{tk}.json"
        was_cached = cache_path.exists() and not refresh
        trades = fetch_tape(tk, refresh=refresh)
        tape_data[tk] = trades
        if i % 10 == 0 or i == len(tickers):
            print(f"    [{i}/{len(tickers)}] last={tk} cached={was_cached} "
                  f"trades={len(trades):,}")

    # Aggregate stats — focus on shake-out recovery patterns
    print(f"\n[Aggregate — drawdown / recovery patterns]")
    print(f"  Examining {len(positions)} positions...")

    drawdown_recovery = []  # list of (entry_price, low_after_entry, settle_price)
    for p in positions:
        tk = p["ticker"]
        trades = tape_data.get(tk, [])
        close_utc = parse_close_time_utc(tk)
        if not close_utc or not trades:
            continue
        try:
            entry_ts = _parse_iso(p["ts_open"])
        except (ValueError, KeyError, TypeError):
            continue
        entry_side = p["side"]  # "yes" or "no"
        entry_price_yes = None  # YES-price at entry
        if entry_side == "yes":
            entry_price_yes = p["entry_avg"]
        else:
            entry_price_yes = 100 - p["entry_avg"]

        # Walk forward from entry_ts to settlement
        post_entry = [
            t for t in trades
            if _parse_iso(t["created_time"]) >= entry_ts
        ]
        if not post_entry:
            continue

        prices_yes = [int(round(float(t["yes_price_dollars"]) * 100)) for t in post_entry]
        min_yes_after = min(prices_yes)
        max_yes_after = max(prices_yes)
        settle_yes = 100 if p["result"] == "yes" else (0 if p["result"] == "no" else None)
        if settle_yes is None:
            continue

        # Drawdown from entry, in the direction OUR side cares about
        if entry_side == "yes":
            max_adverse_pct = (entry_price_yes - min_yes_after) / max(entry_price_yes, 1)
        else:
            our_entry = 100 - entry_price_yes  # NO-side entry
            our_min = 100 - max_yes_after       # NO-side at max-adverse YES move
            max_adverse_pct = (our_entry - our_min) / max(our_entry, 1)
        drawdown_recovery.append({
            "ticker": tk, "side": entry_side, "class": p["class"],
            "entry_yes": entry_price_yes,
            "min_yes_after": min_yes_after, "max_yes_after": max_yes_after,
            "settle_yes": settle_yes, "max_adverse_pct": max_adverse_pct,
            "realized_pnl": p["pnl"],
            "exit_bucket": p["exit_bucket"],
        })

    # Drawdown buckets
    print(f"  Positions analyzed: {len(drawdown_recovery)}")
    by_class = defaultdict(list)
    for r in drawdown_recovery:
        by_class[r["class"]].append(r)

    for cls in ("correct_win", "shaken_out", "wrong_loss", "saved_by_exit"):
        rs = by_class.get(cls, [])
        if not rs:
            continue
        avg_drawdown = sum(r["max_adverse_pct"] for r in rs) / len(rs)
        deep_drawdown = sum(1 for r in rs if r["max_adverse_pct"] > 0.40)
        print(f"  {cls:15}: n={len(rs):3d}  "
              f"avg max-adverse drawdown={avg_drawdown:5.1%}  "
              f">40% adverse: {deep_drawdown}/{len(rs)}")

    # The big question: of trades that hit -40% drawdown, how many recovered?
    print(f"\n[Recovery from deep drawdown — would the cool-off have worked longer?]")
    deep = [r for r in drawdown_recovery if r["max_adverse_pct"] > 0.40]
    print(f"  Positions that hit >40% adverse drawdown: {len(deep)}")
    for cls in ("shaken_out", "correct_win", "wrong_loss"):
        rs = [r for r in deep if r["class"] == cls]
        if rs:
            recovered = sum(1 for r in rs
                            if (r["side"] == "yes" and r["settle_yes"] == 100)
                            or (r["side"] == "no" and r["settle_yes"] == 0))
            print(f"  {cls:15}: n={len(rs):3d}  "
                  f"settled in our favor: {recovered}/{len(rs)} "
                  f"({recovered/max(len(rs),1)*100:.0f}%)")

    # Shaken-out specific: how much was left on the table?
    so = [r for r in drawdown_recovery if r["class"] == "shaken_out"]
    if so:
        # If held to settle: each SO trade would have settled in our favor (by definition)
        # so PnL would have been (qty * 1.00 - cost). The qty isn't in this view, but we
        # can compute the per-contract gain.
        total_realized = sum(r["realized_pnl"] for r in so)
        # Lookup positions for full info
        so_full = [p for p in positions if p["class"] == "shaken_out"]
        total_hypothetical = sum(
            p["qty"] * 1.00 - (p["qty"] * p["entry_avg"] / 100) for p in so_full
        )
        print(f"\n[Shaken-out salvage]")
        print(f"  Realized:           ${total_realized:+.2f}")
        print(f"  If held to settle:  ${total_hypothetical:+.2f}")
        print(f"  Total left on table: ${total_hypothetical - total_realized:+.2f}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Kalshi market tape analyzer")
    ap.add_argument("--ticker", help="Single ticker to pull and render")
    ap.add_argument("--session", help="Session tag (e.g. 18MAY22:30) for aggregate")
    ap.add_argument("--filter", choices=["correct_win", "shaken_out",
                                          "saved_by_exit", "wrong_loss"],
                    help="Only positions matching this class")
    ap.add_argument("--summary", action="store_true",
                    help="Session-level summary (default if --session given)")
    ap.add_argument("--refresh", action="store_true",
                    help="Force re-fetch even if cache exists")
    args = ap.parse_args()

    if args.ticker:
        trades = fetch_tape(args.ticker, refresh=args.refresh)
        bot_events = load_bot_trades_for_ticker(args.ticker)
        # Lookup settlement result
        result = None
        if CROSS_VALIDATE_JSON.exists():
            cv = json.loads(CROSS_VALIDATE_JSON.read_text())
            for sess in cv.values():
                for p in sess.get("positions", []):
                    if p["ticker"] == args.ticker:
                        result = p.get("result")
                        break
                if result: break
        print(render_timeline(args.ticker, trades, bot_events, result))
    elif args.session:
        analyze_session(args.session, filter_class=args.filter, refresh=args.refresh)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
