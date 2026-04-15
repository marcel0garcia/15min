"""
Session post-mortem — pulls public Kalshi trade tapes for every ticker traded
in the most recent session and evaluates each entry against the final result
and the tape's live price around our fill.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).parent
TRADES = ROOT / "logs" / "trades.csv"
import os
SESSION_TAGS = tuple(os.environ.get("SESSIONS", "12APR20:43,12APR20:20").split(","))

BASE = "https://api.elections.kalshi.com/trade-api/v2"
_client = httpx.Client(timeout=10.0, headers={"accept": "application/json"})


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_session():
    rows = []
    with open(TRADES) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                continue
            if row[9] in SESSION_TAGS:
                rows.append({
                    "order_id": row[0],
                    "ts": parse_ts(row[1]),
                    "ticker": row[2],
                    "side": row[3],
                    "contracts": int(row[4]),
                    "price_cents": int(row[5]),
                    "cost_usd": float(row[6]),
                    "source": row[7],
                    "mode": row[8],
                    "session": row[9],
                })
    return rows


def fetch_market(ticker: str):
    r = _client.get(f"{BASE}/markets/{ticker}")
    if r.status_code != 200:
        return None
    return r.json().get("market")


def fetch_tape(ticker: str, max_trades: int = 5000):
    trades = []
    cursor = None
    while True:
        params = {"ticker": ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        r = _client.get(f"{BASE}/markets/trades", params=params)
        if r.status_code != 200:
            break
        data = r.json()
        trades.extend(data.get("trades", []))
        cursor = data.get("cursor")
        if not cursor or len(trades) >= max_trades:
            break
        time.sleep(0.08)
    # normalize: add parsed time and yes_cents
    out = []
    for t in trades:
        try:
            yp = float(t.get("yes_price_dollars", 0)) * 100
            out.append({
                "t": parse_ts(t["created_time"]),
                "yes": yp,
                "count": float(t.get("count_fp", 0)),
                "taker_side": t.get("taker_side"),
            })
        except Exception:
            continue
    out.sort(key=lambda x: x["t"])
    return out


def tape_stats_around(tape, t0, t1):
    """Return (min_yes, max_yes, n_trades, volume) in [t0, t1]."""
    window = [x for x in tape if t0 <= x["t"] <= t1]
    if not window:
        return None
    ys = [x["yes"] for x in window]
    return {
        "n": len(window),
        "min": min(ys),
        "max": max(ys),
        "first": window[0]["yes"],
        "last": window[-1]["yes"],
        "vol": sum(x["count"] for x in window),
    }


def closest_price(tape, t):
    """Nearest tape yes_price at time t."""
    best = None
    best_dt = None
    for x in tape:
        dt = abs((x["t"] - t).total_seconds())
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best = x
        if dt > 600 and best:  # we've walked past 10 min, stop
            break
    return best


def main():
    rows = load_session()
    print(f"Loaded {len(rows)} trade rows across sessions {SESSION_TAGS}\n")

    # Build per-ticker timeline + per-order lifecycle
    by_ticker = defaultdict(list)
    for r in rows:
        by_ticker[r["ticker"]].append(r)
    for t in by_ticker:
        by_ticker[t].sort(key=lambda r: r["ts"])

    # Fetch tapes + markets
    print("Fetching public Kalshi data…")
    tickers = sorted(by_ticker.keys())
    tapes = {}
    markets = {}
    for t in tickers:
        markets[t] = fetch_market(t)
        tapes[t] = fetch_tape(t)
        time.sleep(0.12)
        print(f"  {t}  result={markets[t].get('result') if markets[t] else '?':<4}  tape={len(tapes[t])}")
    print()

    # Build per-entry analysis
    analysis = []   # one row per *entry* (not exit)
    session_pnl = 0.0
    for ticker in tickers:
        mk = markets[ticker] or {}
        result = mk.get("result")   # 'yes' or 'no'
        tape = tapes[ticker]
        # Order flow: pair entries with their exit(s)
        orders = defaultdict(lambda: {"opens": [], "closes": []})
        for r in by_ticker[ticker]:
            if r["side"].endswith("_exit") or r["side"].endswith("_settled"):
                orders[r["order_id"]]["closes"].append(r)
            else:
                orders[r["order_id"]]["opens"].append(r)

        for oid, legs in orders.items():
            for open_leg in legs["opens"]:
                close_legs = legs["closes"]
                # sum matching close legs for this order
                if close_legs:
                    close_total = sum(c["cost_usd"] for c in close_legs)
                    close_contracts = sum(c["contracts"] for c in close_legs)
                    close_avg = (close_total / close_contracts * 100) if close_contracts else None
                    exit_time = max(c["ts"] for c in close_legs)
                    exit_source = close_legs[-1]["source"]
                else:
                    close_total = close_contracts = close_avg = None
                    exit_time = None
                    exit_source = "OPEN"

                open_price = open_leg["price_cents"]
                side = open_leg["side"]    # 'yes' / 'no'
                # Was the side correct given final result?
                correct_side = (side == result) if result else None

                # If we held → settled for $1 (if correct) or $0 (if wrong)
                # If we exited → realized $ = close_total - open_cost
                if close_total is not None:
                    realized = close_total - open_leg["cost_usd"]
                else:
                    realized = None

                # Tape context at entry: +/- 30s window
                t_open = open_leg["ts"]
                pre = tape_stats_around(
                    tape,
                    t_open.replace(microsecond=0) - datetime.resolution * 0,  # crude: 60s pre
                    t_open,
                )
                # Simpler: use seconds offset
                from datetime import timedelta
                pre_w = tape_stats_around(tape, t_open - timedelta(seconds=60), t_open)
                post_w = tape_stats_around(tape, t_open, t_open + timedelta(seconds=180))
                tape_at = closest_price(tape, t_open)

                analysis.append({
                    "ticker": ticker,
                    "result": result,
                    "open_ts": t_open.isoformat(),
                    "exit_ts": exit_time.isoformat() if exit_time else None,
                    "side": side,
                    "contracts": open_leg["contracts"],
                    "entry_cents": open_price,
                    "exit_cents": close_avg,
                    "correct_side": correct_side,
                    "source": open_leg["source"],
                    "exit_source": exit_source,
                    "realized_usd": round(realized, 2) if realized is not None else None,
                    "hold_secs": (exit_time - t_open).total_seconds() if exit_time else None,
                    # tape context
                    "tape_yes_at_open": round(tape_at["yes"], 1) if tape_at else None,
                    "tape_pre60_min": round(pre_w["min"], 0) if pre_w else None,
                    "tape_pre60_max": round(pre_w["max"], 0) if pre_w else None,
                    "tape_post180_min": round(post_w["min"], 0) if post_w else None,
                    "tape_post180_max": round(post_w["max"], 0) if post_w else None,
                })
                if realized is not None:
                    session_pnl += realized

    # Print a clean table
    print("\n═════════════════════════ PER-ENTRY BREAKDOWN ═════════════════════════\n")
    hdr = f"{'ticker':30} {'side':3} {'entry':>5} {'exit':>5} {'pnl$':>7} {'res':3} {'ok':2} {'tape@':>6} {'post180':>10} {'reason':38}"
    print(hdr)
    print("-" * len(hdr))
    for a in sorted(analysis, key=lambda r: r["open_ts"]):
        ok = "✓" if a["correct_side"] else ("✗" if a["correct_side"] is False else "?")
        tp = f"[{a['tape_post180_min']}-{a['tape_post180_max']}]" if a["tape_post180_min"] is not None else ""
        print(
            f"{a['ticker']:30} {a['side'][:3]:3} "
            f"{a['entry_cents']:>4}¢ "
            f"{str(a['exit_cents'] and int(a['exit_cents']))+'¢':>5} "
            f"{a['realized_usd'] if a['realized_usd'] is not None else '':>7} "
            f"{a['result'][:3] if a['result'] else '?':3} {ok:2} "
            f"{a['tape_yes_at_open']:>5}¢ "
            f"{tp:>10} "
            f"{a['source'][:38]:38}"
        )

    # Aggregate stats
    wins = [a for a in analysis if (a["realized_usd"] or 0) > 0]
    losses = [a for a in analysis if (a["realized_usd"] or 0) < 0]
    print(f"\nTotal entries: {len(analysis)}  | wins={len(wins)}  losses={len(losses)}")
    print(f"Session realized P&L (from entry/exit pairs): ${session_pnl:+.2f}")

    # How many entries were on the correct side of final result?
    correct = [a for a in analysis if a["correct_side"] is True]
    wrong = [a for a in analysis if a["correct_side"] is False]
    print(f"\nEntries on CORRECT side (by final settlement): {len(correct)} / {len(analysis)}")
    print(f"Entries on WRONG   side (by final settlement): {len(wrong)} / {len(analysis)}")

    # Of the losses, how many were on the correct side but got stopped?
    stopped_correct = [a for a in losses if a["correct_side"] is True]
    print(f"\nSTOPPED LOSSES that were on the CORRECT side (got shaken out!): {len(stopped_correct)}")
    for a in stopped_correct:
        print(f"  • {a['ticker']} {a['side'].upper()} entry={a['entry_cents']}¢ "
              f"exit={a['exit_cents']:.0f}¢ pnl=${a['realized_usd']:+.2f} ({a['exit_source'][:45]})")

    # Of the wins, how many on wrong side but cashed via exit?
    won_wrong = [a for a in wins if a["correct_side"] is False]
    print(f"\nWINS that were on the WRONG side (saved by exit before settlement): {len(won_wrong)}")
    for a in won_wrong:
        print(f"  • {a['ticker']} {a['side'].upper()} entry={a['entry_cents']}¢ "
              f"exit={a['exit_cents']:.0f}¢ pnl=${a['realized_usd']:+.2f} ({a['exit_source'][:45]})")

    out = ROOT / "session_analysis.json"
    out.write_text(json.dumps(analysis, indent=2, default=str))
    print(f"\nFull per-entry data → {out}")


if __name__ == "__main__":
    main()
