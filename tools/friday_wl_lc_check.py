"""Counterfactual: does the proposed cool-off ALSO hurt WL (wrong_loss) cuts?

For every WL trade that exited via loss_cut, fetch the Kalshi tape and look
at our-side price behavior in the 6s after cut (matching the max cool-off
window for >480s runway).

If WL loss_cuts also recover briefly within 6s, the cool-off would let some
losers bleed further before firing, eroding the SO gain. If WL loss_cuts
keep falling within 6s, the cool-off cleanly distinguishes flushes from
real adverse moves.

Also bins WL cuts by runway-at-cut so we can score the cool-off impact
under the actual proposed schedule (6s for >480s, 3s for 240-480s, 0s for <240s).
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
XVAL = ROOT / "logs" / "friday_snapshot" / "cross_validate.json"
TAPES_DIR = ROOT / "logs" / "friday_snapshot" / "tapes"
TAPES_DIR.mkdir(parents=True, exist_ok=True)
BASE = "https://api.elections.kalshi.com/trade-api/v2"
REQ_DELAY = 0.2


def parse_ts(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    m = re.match(r"^(.*\.)(\d{1,5})([+-]\d{2}:\d{2})$", s)
    if m:
        s = m.group(1) + m.group(2).ljust(6, "0") + m.group(3)
    return datetime.fromisoformat(s)


def load_wl_loss_cuts():
    data = json.loads(XVAL.read_text())
    out = []
    for sess, info in data.items():
        for p in info["positions"]:
            if p["class"] == "wrong_loss" and p["exit_bucket"] == "loss_cut":
                out.append({**p, "session": sess,
                            "ts_open": parse_ts(p["ts_open"]),
                            "ts_exit": parse_ts(p["ts_exit"])})
    return out


def fetch_tape(ticker: str, client: httpx.Client) -> list[dict]:
    path = TAPES_DIR / f"{ticker}.json"
    if path.exists():
        return json.loads(path.read_text())
    trades = []
    cursor = None
    while True:
        params = {"ticker": ticker, "limit": 1000}
        if cursor: params["cursor"] = cursor
        r = client.get(f"{BASE}/markets/trades", params=params)
        if r.status_code != 200: break
        data = r.json()
        trades.extend(data.get("trades", []))
        cursor = data.get("cursor")
        if not cursor or len(trades) >= 5000: break
        time.sleep(REQ_DELAY)
    out = []
    for t in trades:
        try:
            yp = float(t.get("yes_price_dollars", 0)) * 100
            out.append({"t": t["created_time"], "yes": round(yp, 1),
                        "count": float(t.get("count_fp", 0))})
        except Exception:
            continue
    out.sort(key=lambda x: x["t"])
    path.write_text(json.dumps(out))
    return out


def yes_to_our_side(yes_cents, side):
    return yes_cents if side == "yes" else (100 - yes_cents)


def window_max_our_side(tape, t0, t1, side):
    """Best (highest) price our side reached in [t0, t1]."""
    window = [x for x in tape if t0 <= parse_ts(x["t"]) <= t1]
    if not window: return None
    if side == "yes":
        return max(x["yes"] for x in window)
    else:
        return max(100 - x["yes"] for x in window)


def runway_band(secs):
    if secs > 480: return ">480s (cool=6s)"
    if secs > 240: return "240-480s (cool=3s)"
    return "<240s (cool=0s, no change)"


def main():
    events = load_wl_loss_cuts()
    print(f"Loaded {len(events)} WL loss_cut events")
    tickers = sorted({e["ticker"] for e in events})
    print(f"Unique tickers needed: {len(tickers)}")
    cached = sum(1 for t in tickers if (TAPES_DIR / f"{t}.json").exists())
    print(f"  Cached: {cached}/{len(tickers)} | Fetching: {len(tickers)-cached}")

    client = httpx.Client(timeout=15.0, headers={"accept": "application/json"})
    tapes = {}
    try:
        for i, t in enumerate(tickers, 1):
            tapes[t] = fetch_tape(t, client)
            if i % 25 == 0 or i == len(tickers):
                print(f"    [{i}/{len(tickers)}]")
    finally:
        client.close()

    # We need runway-at-cut. We don't have it directly in cross_validate.json,
    # but close_time is in the cache (market_results.json). secs_to_settle was
    # computed before — recompute here.
    market_results = json.loads((ROOT / "logs/friday_snapshot/market_results.json").read_text())
    for e in events:
        ct = market_results.get(e["ticker"], {}).get("close_time")
        e["secs_remaining_at_cut"] = (parse_ts(ct) - e["ts_exit"]).total_seconds() if ct else None

    # For each event, compute the "best price our side hit within 6s and 3s post-cut".
    # If that best price is meaningfully higher than the cut price, the cool-off would
    # have triggered a clear-pending and we'd hold a wrong-side loser.
    held_3s = held_6s = 0
    held_3s_extra_loss = held_6s_extra_loss = 0.0
    by_band = defaultdict(lambda: {"n": 0, "would_hold_at_band_cooloff": 0,
                                   "would_recover_to_within_5c_of_entry": 0,
                                   "extra_loss_if_held": 0.0})

    examples = []
    for e in events:
        tape = tapes.get(e["ticker"], [])
        cut_price = e["exit_avg"]
        entry_price = e["entry_avg"]
        side = e["side"]
        t_cut = e["ts_exit"]
        secs = e["secs_remaining_at_cut"]
        if secs is None: continue
        band = runway_band(secs)
        cool = 6.0 if "(cool=6s)" in band else (3.0 if "(cool=3s)" in band else 0.0)
        by_band[band]["n"] += 1
        if cool == 0:
            continue  # cool-off=0 in this band → no behavior change
        # Best our-side price in cool-off window
        best_in_cool = window_max_our_side(tape, t_cut, t_cut + timedelta(seconds=cool), side)
        # Cool-off pending only fires if condition is STILL true at end of window.
        # Condition = price ≤ cut+threshold. Approximation: if best price in window
        # > cut_price + 1¢, the model would likely have flipped back to "agree" or
        # pnl would have recovered above threshold → we'd hold (wrongly, since this is WL).
        if best_in_cool is not None and best_in_cool > cut_price + 1:
            by_band[band]["would_hold_at_band_cooloff"] += 1
            # If we held, the eventual pnl is somewhere between current loss and final settlement.
            # We settled on the wrong side, so the trade goes to ~$0 settlement on our side.
            # Extra loss vs cut = (close_price_at_settlement - cut_price) * qty/100
            # We don't know the exact final price — use 0¢ on our side for WL (worst case).
            extra = (0 - cut_price) * e["qty"] / 100
            by_band[band]["extra_loss_if_held"] += extra
            if best_in_cool >= entry_price - 5:
                by_band[band]["would_recover_to_within_5c_of_entry"] += 1
            if len(examples) < 6:
                examples.append({
                    "ticker": e["ticker"], "side": side, "entry": entry_price,
                    "cut": cut_price, "best_in_cool": best_in_cool,
                    "cool": cool, "secs_remaining": secs,
                    "qty": e["qty"], "extra_if_held": extra,
                })

    print()
    print("═" * 96)
    print(f"  WL LOSS_CUT CHECK — would the cool-off have held losers longer?")
    print("═" * 96)
    print(f"  {'runway band':<28} {'n':>4} {'would hold':>11} {'would recover':>14} {'extra loss if held':>18}")
    print(f"  {'─'*28} {'─'*4} {'─'*11} {'─'*14} {'─'*18}")
    grand = {"n": 0, "hold": 0, "recover": 0, "extra": 0.0}
    for band, s in sorted(by_band.items()):
        n = s["n"]; hold = s["would_hold_at_band_cooloff"]
        recover = s["would_recover_to_within_5c_of_entry"]; extra = s["extra_loss_if_held"]
        pct_hold = (hold / n * 100) if n else 0
        print(f"  {band:<28} {n:>4} {hold:>5} ({pct_hold:>4.0f}%)  "
              f"{recover:>5} ({recover/max(hold,1)*100:>4.0f}% of held)  "
              f"${extra:>+10.2f} (worst-case)")
        grand["n"] += n; grand["hold"] += hold; grand["recover"] += recover
        grand["extra"] += extra

    print(f"  {'─'*28} {'─'*4} {'─'*11} {'─'*14} {'─'*18}")
    print(f"  {'TOTAL':<28} {grand['n']:>4} {grand['hold']:>5} ({grand['hold']/max(grand['n'],1)*100:>4.0f}%)  "
          f"{grand['recover']:>5}        "
          f"${grand['extra']:>+10.2f} (worst-case)")
    print()
    print(f"  How to read 'extra loss if held': worst-case assumes the trade would")
    print(f"  go to $0 on our side at settlement. Real outcome usually less bad.")
    print()
    if examples:
        print("  ── Sample WL events that WOULD have been held by the cool-off ──")
        for ex in examples:
            print(f"  {ex['ticker']:32} {ex['side']:3}  entry={ex['entry']:.0f}c "
                  f"cut={ex['cut']:.0f}c best_in_{int(ex['cool'])}s={ex['best_in_cool']:.0f}c "
                  f"qty={ex['qty']}  extra_if_held=${ex['extra_if_held']:+.2f} "
                  f"({ex['secs_remaining']:.0f}s remaining at cut)")
    print()


if __name__ == "__main__":
    main()
