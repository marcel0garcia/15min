"""Cross-validate Friday paper-trading entries against Kalshi market outcomes.

For each (ticker, side) position in a session lasting >= MIN_SESSION_MINUTES,
fetch the final settlement result from Kalshi's public /markets/{ticker}
endpoint and report:
  - side accuracy (did we enter on the side that won?)
  - "shaken out" losses (correct side but stopped before settlement)
  - "saved by exit" wins (wrong side but exited profitably before settlement)
  - exit-reason mix conditioned on correctness
  - per-session and aggregate rollups

Rate-limited to ~5 req/s. Results cached to logs/friday_snapshot/market_results.json
so re-runs only fetch new tickers.
"""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
TRADES = ROOT / "logs" / "friday_snapshot" / "trades.csv"
CACHE = ROOT / "logs" / "friday_snapshot" / "market_results.json"
OUT_JSON = ROOT / "logs" / "friday_snapshot" / "cross_validate.json"

MIN_SESSION_MINUTES = 60
REQ_DELAY_SEC = 0.2  # 5 req/s — well under Kalshi's public-endpoint headroom
BASE = "https://api.elections.kalshi.com/trade-api/v2"


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def is_exit(side: str) -> bool:
    return side.endswith("_exit") or side.endswith("_settled")


def side_family(s: str) -> str:
    return "yes" if s.startswith("yes") else "no"


def exit_bucket(source: str) -> str:
    s = source.lower()
    if "loss_cut" in s: return "loss_cut"
    if "profit_take" in s: return "profit_take"
    if "reversal" in s: return "reversal"
    if "settled" in s or "settlement" in s: return "settled"
    if "time" in s or "expir" in s: return "time_stop"
    return "other"


def tag_sort_key(tag: str):
    day = int(tag[:2])
    mon = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
           "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}[tag[2:5]]
    hh, mm = tag[5:].split(":")
    return (mon, day, int(hh), int(mm))


def load_rows():
    rows = []
    with open(TRADES) as f:
        for row in csv.reader(f):
            if len(row) != 10 or row[0] == "trade_id":
                continue
            rows.append({
                "ts": parse_ts(row[1]),
                "ticker": row[2],
                "side": row[3],
                "contracts": int(row[4]),
                "price_cents": int(row[5]),
                "cost_usd": float(row[6]),
                "source": row[7],
                "session": row[9],
            })
    return rows


def filter_long_sessions(rows):
    """Return only rows belonging to sessions ≥ MIN_SESSION_MINUTES long."""
    by_sess = defaultdict(lambda: {"lo": None, "hi": None, "rows": []})
    for r in rows:
        s = by_sess[r["session"]]
        s["rows"].append(r)
        if s["lo"] is None or r["ts"] < s["lo"]: s["lo"] = r["ts"]
        if s["hi"] is None or r["ts"] > s["hi"]: s["hi"] = r["ts"]
    kept = {}
    dropped = []
    for tag, s in by_sess.items():
        dur_min = (s["hi"] - s["lo"]).total_seconds() / 60
        if dur_min >= MIN_SESSION_MINUTES:
            kept[tag] = {"rows": s["rows"], "dur_min": dur_min,
                         "lo": s["lo"], "hi": s["hi"]}
        else:
            dropped.append((tag, dur_min))
    return kept, dropped


def load_cache():
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    return {}


def save_cache(cache):
    CACHE.write_text(json.dumps(cache, indent=2, default=str))


def fetch_market_results(tickers, cache):
    """Fetch market.result for tickers, skipping any already in cache.
    Returns cache (mutated). Rate-limited to ~5 req/s."""
    todo = [t for t in tickers if t not in cache]
    if not todo:
        print(f"  All {len(tickers)} markets already cached")
        return cache

    print(f"  Cached: {len(tickers) - len(todo)} | Fetching: {len(todo)} "
          f"(~{len(todo) * REQ_DELAY_SEC:.0f}s @ {1/REQ_DELAY_SEC:.0f} req/s)")
    client = httpx.Client(timeout=10.0, headers={"accept": "application/json"})
    fetched = errors = 0
    t0 = time.time()
    try:
        for i, ticker in enumerate(todo, 1):
            try:
                r = client.get(f"{BASE}/markets/{ticker}")
                if r.status_code == 200:
                    mk = r.json().get("market") or {}
                    cache[ticker] = {
                        "result": mk.get("result"),         # 'yes' / 'no' / '' / None
                        "status": mk.get("status"),
                        "close_time": mk.get("close_time"),
                        "fetched_at": datetime.utcnow().isoformat(),
                    }
                    fetched += 1
                else:
                    cache[ticker] = {"result": None, "http_status": r.status_code}
                    errors += 1
            except Exception as e:
                cache[ticker] = {"result": None, "error": str(e)[:120]}
                errors += 1
            if i % 50 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"    [{i}/{len(todo)}] fetched={fetched} errors={errors} "
                      f"rate={rate:.1f} req/s")
                save_cache(cache)  # checkpoint every 50
            time.sleep(REQ_DELAY_SEC)
    finally:
        client.close()
        save_cache(cache)
    return cache


def build_positions_per_session(kept_sessions):
    """Aggregate to (ticker, side_family) per session — same approach as
    friday_session_summary.py for correct pyramid handling."""
    per_session = {}
    for tag, info in kept_sessions.items():
        legs = defaultdict(lambda: {"opens": [], "closes": []})
        for r in sorted(info["rows"], key=lambda x: x["ts"]):
            key = (r["ticker"], side_family(r["side"]))
            (legs[key]["closes"] if is_exit(r["side"]) else
             legs[key]["opens"]).append(r)
        positions = []
        for (ticker, fam), lg in legs.items():
            if not lg["opens"]:
                continue
            open_cost = sum(o["cost_usd"] for o in lg["opens"])
            open_qty = sum(o["contracts"] for o in lg["opens"])
            close_cost = sum(c["cost_usd"] for c in lg["closes"])
            close_qty = sum(c["contracts"] for c in lg["closes"])
            ts_open = min(o["ts"] for o in lg["opens"])
            exit_ts = max((c["ts"] for c in lg["closes"]), default=None)
            exit_src = lg["closes"][-1]["source"] if lg["closes"] else "OPEN"
            fully_closed = bool(lg["closes"]) and close_qty >= open_qty
            pnl = (close_cost - open_cost) if fully_closed else None
            positions.append({
                "ticker": ticker,
                "side": fam,
                "qty": open_qty,
                "entry_avg": (open_cost / open_qty * 100) if open_qty else 0,
                "exit_avg": (close_cost / close_qty * 100) if close_qty else None,
                "pnl": pnl,
                "ts_open": ts_open,
                "ts_exit": exit_ts,
                "exit_bucket": exit_bucket(exit_src) if lg["closes"] else "OPEN",
            })
        per_session[tag] = {
            "lo": info["lo"], "hi": info["hi"], "dur_min": info["dur_min"],
            "positions": positions,
        }
    return per_session


def classify(positions, market_results):
    """Decorate positions with correctness vs. final settlement."""
    out = []
    for p in positions:
        mr = market_results.get(p["ticker"], {}) or {}
        result = mr.get("result")  # 'yes' / 'no' / '' / None
        if result in ("yes", "no"):
            correct = (p["side"] == result)
        else:
            correct = None  # market unsettled or unknown
        cls = "open"
        if p["pnl"] is None:
            cls = "open"
        elif correct is None:
            cls = "unknown"
        elif correct and p["pnl"] > 0:
            cls = "correct_win"          # right side, exited at profit (or held to settle)
        elif correct and p["pnl"] <= 0:
            cls = "shaken_out"           # right side, but stopped out
        elif (not correct) and p["pnl"] > 0:
            cls = "saved_by_exit"        # wrong side, but exited profitably
        elif (not correct) and p["pnl"] <= 0:
            cls = "wrong_loss"           # wrong side, stopped/settled as loss
        out.append({**p, "result": result, "correct": correct, "class": cls})
    return out


def print_table(sessions):
    print("═" * 130)
    print(f"  FRIDAY CROSS-VALIDATION — sessions ≥ {MIN_SESSION_MINUTES} min")
    print("═" * 130)
    print(f"  {'session':12} {'dur':>5}  {'pos':>3} {'cls':>3} "
          f"{'W':>3} {'L':>3}  {'P&L $':>8}  "
          f"{'side%':>6}  {'CW':>3} {'SO':>3} {'SE':>3} {'WL':>3}  "
          f"{'exit-mix on correct entries':36}")
    print("─" * 130)
    grand = defaultdict(int)
    grand_pnl = 0.0
    grand_pnl_cw = 0.0
    grand_pnl_so = 0.0
    grand_pnl_se = 0.0
    grand_pnl_wl = 0.0

    for tag in sorted(sessions.keys(), key=tag_sort_key):
        s = sessions[tag]
        positions = s["enriched"]
        closed = [p for p in positions if p["pnl"] is not None]
        wins = [p for p in closed if p["pnl"] > 0]
        losses = [p for p in closed if p["pnl"] <= 0]
        # Side accuracy uses only positions with a known settlement
        rated = [p for p in positions if p["correct"] is not None]
        correct = [p for p in rated if p["correct"]]
        side_pct = (len(correct) / len(rated) * 100) if rated else 0

        cw = [p for p in closed if p["class"] == "correct_win"]
        so = [p for p in closed if p["class"] == "shaken_out"]
        se = [p for p in closed if p["class"] == "saved_by_exit"]
        wl = [p for p in closed if p["class"] == "wrong_loss"]
        pnl = sum(p["pnl"] for p in closed)

        # Exit-reason mix on correct-side entries that closed
        exit_mix = defaultdict(int)
        for p in [x for x in closed if x["correct"]]:
            exit_mix[p["exit_bucket"]] += 1
        mix_str = ", ".join(f"{k}:{v}" for k, v in
                            sorted(exit_mix.items(), key=lambda x: -x[1]))[:36]

        marker = "▲" if pnl > 0 else ("▼" if pnl < 0 else "·")
        print(f"  {tag:12} {s['dur_min']:>4.0f}m {len(positions):>3} {len(closed):>3} "
              f"{len(wins):>3} {len(losses):>3}  "
              f"{marker}${pnl:>+7.2f}  "
              f"{side_pct:>5.1f}%  "
              f"{len(cw):>3} {len(so):>3} {len(se):>3} {len(wl):>3}  "
              f"{mix_str:36}")

        grand["pos"] += len(positions)
        grand["cls"] += len(closed)
        grand["W"] += len(wins)
        grand["L"] += len(losses)
        grand["rated"] += len(rated)
        grand["correct"] += len(correct)
        grand["cw"] += len(cw)
        grand["so"] += len(so)
        grand["se"] += len(se)
        grand["wl"] += len(wl)
        grand_pnl += pnl
        grand_pnl_cw += sum(p["pnl"] for p in cw)
        grand_pnl_so += sum(p["pnl"] for p in so)
        grand_pnl_se += sum(p["pnl"] for p in se)
        grand_pnl_wl += sum(p["pnl"] for p in wl)

    print("─" * 130)
    side_pct = (grand["correct"] / grand["rated"] * 100) if grand["rated"] else 0
    print(f"  {'TOTAL':12} {'':>5}  {grand['pos']:>3} {grand['cls']:>3} "
          f"{grand['W']:>3} {grand['L']:>3}  "
          f"${grand_pnl:>+7.2f}  {side_pct:>5.1f}%  "
          f"{grand['cw']:>3} {grand['so']:>3} {grand['se']:>3} {grand['wl']:>3}")
    print()
    print(f"  Legend: CW=correct_win  SO=shaken_out (right side, stopped)  "
          f"SE=saved_by_exit (wrong side, exited at profit)  WL=wrong_loss")
    print()
    print(f"  P&L breakdown by class:")
    print(f"    correct_win  (CW): {grand['cw']:>3}  ${grand_pnl_cw:>+8.2f}   "
          f"avg ${grand_pnl_cw/max(grand['cw'],1):>+5.2f}")
    print(f"    shaken_out   (SO): {grand['so']:>3}  ${grand_pnl_so:>+8.2f}   "
          f"avg ${grand_pnl_so/max(grand['so'],1):>+5.2f}  "
          f"← would have been wins if held to settlement")
    print(f"    saved_by_exit(SE): {grand['se']:>3}  ${grand_pnl_se:>+8.2f}   "
          f"avg ${grand_pnl_se/max(grand['se'],1):>+5.2f}  "
          f"← active management saved wrong-side entries")
    print(f"    wrong_loss   (WL): {grand['wl']:>3}  ${grand_pnl_wl:>+8.2f}   "
          f"avg ${grand_pnl_wl/max(grand['wl'],1):>+5.2f}")
    print()


def main():
    rows = load_rows()
    kept, dropped = filter_long_sessions(rows)
    print(f"Loaded {len(rows)} rows. Sessions ≥{MIN_SESSION_MINUTES}m: {len(kept)}. "
          f"Dropped: {len(dropped)} ({', '.join(t for t,_ in dropped)})")

    per_session = build_positions_per_session(kept)
    all_tickers = sorted({p["ticker"] for s in per_session.values() for p in s["positions"]})
    print(f"Unique tickers across kept sessions: {len(all_tickers)}")

    cache = load_cache()
    cache = fetch_market_results(all_tickers, cache)

    # How many actually settled?
    settled = sum(1 for t in all_tickers if (cache.get(t) or {}).get("result") in ("yes", "no"))
    print(f"  Settled: {settled}/{len(all_tickers)}  "
          f"({len(all_tickers)-settled} still open / unknown)")
    print()

    for tag in per_session:
        per_session[tag]["enriched"] = classify(per_session[tag]["positions"], cache)

    print_table(per_session)

    # Persist
    snap = {tag: {"dur_min": s["dur_min"],
                  "positions": [
                      {**p,
                       "ts_open": p["ts_open"].isoformat(),
                       "ts_exit": p["ts_exit"].isoformat() if p["ts_exit"] else None,
                      } for p in s["enriched"]
                  ]}
            for tag, s in per_session.items()}
    OUT_JSON.write_text(json.dumps(snap, indent=2, default=str))
    print(f"  Full per-position data → {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
