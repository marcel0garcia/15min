"""Verify the panic-flush hypothesis on the 29 SO loss_cut events.

For each event:
  1. Fetch Kalshi public trade tape for the ticker, ±10 min around exit.
  2. Compute price trajectory on OUR side at: cut moment, +30s, +60s, +120s.
  3. Find the post-cut min/max on our side over the next 5 min.
  4. Classify the cut as: flush / partial_flush / real_adverse.
  5. From bot.log, pull SIGNAL/SUPPRESSED entries in the 90s pre-cut window
     to see the model's conf/edge trajectory.

Caches Kalshi tapes to logs/friday_snapshot/tapes/<ticker>.json.
Rate-limited to ~5 req/s.
"""
from __future__ import annotations

import json
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
XVAL = ROOT / "logs" / "friday_snapshot" / "cross_validate.json"
BOTLOG = ROOT / "logs" / "friday_snapshot" / "bot.log"
TAPES_DIR = ROOT / "logs" / "friday_snapshot" / "tapes"
OUT_JSON = ROOT / "logs" / "friday_snapshot" / "so_lc_verification.json"

TAPES_DIR.mkdir(parents=True, exist_ok=True)
BASE = "https://api.elections.kalshi.com/trade-api/v2"
REQ_DELAY = 0.2  # 5 req/s


def parse_ts(s: str) -> datetime:
    # Kalshi tape sometimes emits 5-digit fractional seconds; pad to 6 for fromisoformat.
    s = s.replace("Z", "+00:00")
    m = re.match(r"^(.*\.)(\d{1,5})([+-]\d{2}:\d{2})$", s)
    if m:
        s = m.group(1) + m.group(2).ljust(6, "0") + m.group(3)
    return datetime.fromisoformat(s)


def load_so_loss_cuts():
    data = json.loads(XVAL.read_text())
    out = []
    for sess, info in data.items():
        for p in info["positions"]:
            if p["class"] == "shaken_out" and p["exit_bucket"] == "loss_cut":
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
        if r.status_code != 200:
            break
        data = r.json()
        trades.extend(data.get("trades", []))
        cursor = data.get("cursor")
        if not cursor or len(trades) >= 5000:
            break
        time.sleep(REQ_DELAY)
    # Normalize: keep parsed time and yes_cents
    out = []
    for t in trades:
        try:
            yp = float(t.get("yes_price_dollars", 0)) * 100
            out.append({
                "t": t["created_time"],
                "yes": round(yp, 1),
                "count": float(t.get("count_fp", 0)),
                "taker_side": t.get("taker_side"),
            })
        except Exception:
            continue
    out.sort(key=lambda x: x["t"])
    path.write_text(json.dumps(out))
    return out


def closest_price(tape, target_ts):
    """Find yes price at the moment closest to target_ts."""
    if not tape: return None
    best = None
    best_dt = None
    for x in tape:
        dt = abs((parse_ts(x["t"]) - target_ts).total_seconds())
        if best_dt is None or dt < best_dt:
            best_dt = dt; best = x
        if dt > 600 and best:
            break
    return best


def window_stats(tape, t0, t1):
    """Return min/max/last yes price in [t0, t1]."""
    window = [x for x in tape if t0 <= parse_ts(x["t"]) <= t1]
    if not window: return None
    ys = [x["yes"] for x in window]
    return {"n": len(window), "min": min(ys), "max": max(ys),
            "first": window[0]["yes"], "last": window[-1]["yes"]}


def yes_to_our_side(yes_cents: float, side: str) -> float:
    """Convert YES price to OUR-side price."""
    return yes_cents if side == "yes" else (100 - yes_cents)


def classify_cut(p, tape):
    """Did our side recover after we cut?"""
    if not tape: return {"class": "no_tape"}
    t_cut = p["ts_exit"]
    our_cut = p["exit_avg"]
    # Trajectory windows
    p30 = window_stats(tape, t_cut, t_cut + timedelta(seconds=30))
    p60 = window_stats(tape, t_cut, t_cut + timedelta(seconds=60))
    p120 = window_stats(tape, t_cut, t_cut + timedelta(seconds=120))
    p300 = window_stats(tape, t_cut, t_cut + timedelta(seconds=300))

    def our_side(stats):
        if not stats: return None
        return {
            "n": stats["n"],
            "our_min": yes_to_our_side(stats["max"], p["side"])  # max yes = min our side if side=='no'
                       if p["side"] == "no" else stats["min"],
            "our_max": yes_to_our_side(stats["min"], p["side"]) if p["side"] == "no" else stats["max"],
            "our_last": yes_to_our_side(stats["last"], p["side"]),
        }

    s30, s60, s120, s300 = (our_side(p30), our_side(p60),
                            our_side(p120), our_side(p300))

    # Did our side recover to ≥ cut price within window?
    recover_30 = s30 and s30["our_max"] >= our_cut + 1     # +1¢ tolerance
    recover_60 = s60 and s60["our_max"] >= our_cut + 1
    recover_120 = s120 and s120["our_max"] >= our_cut + 1
    # Strong recovery: returned to within 10¢ of entry
    entry = p["entry_avg"]
    strong_recover = s120 and s120["our_max"] >= entry - 10

    if recover_30:        klass = "fast_flush"   # bounced within 30s
    elif recover_60:      klass = "flush"        # bounced within 60s
    elif recover_120:     klass = "slow_flush"   # bounced within 2min
    elif s300 and s300["our_max"] >= our_cut + 5:  klass = "delayed_recover"
    else:                  klass = "real_adverse"

    return {
        "class": klass,
        "strong_recover_in_120s": bool(strong_recover),
        "our_cut": round(our_cut, 1),
        "our_entry": round(entry, 1),
        "p30": s30, "p60": s60, "p120": s120, "p300": s300,
    }


# Parse the bot.log for the 90s preceding a cut to extract conf/edge trajectory
PAT_SIGNAL = re.compile(
    r"^(\S+ \S+).*SIGNAL.*?(\S+-\d+)\s+([A-Z]+)\s+\|\s+conf=(\d+)%\s+edge=([+-]?\d+\.?\d*)%"
)
PAT_LOSS = re.compile(
    r"^(\S+ \S+).*LOSS CUT.*?(\S+-\d+)\s+([A-Z]+).*pnl=([+-]?\d+\.?\d*)%"
)
PAT_SUPPRESSED = re.compile(
    r"^(\S+ \S+).*STOP SUPPRESSED.*?(\S+-\d+)\s+([A-Z]+).*pnl=([+-]?\d+\.?\d*)%"
)
PAT_PYRAMID = re.compile(
    r"^(\S+ \S+).*PYRAMID eligible.*?(\S+-\d+)\s+([A-Z]+)\s+\|\s+pnl=([+-]?\d+\.?\d*)%\s+conf=(\d+)%\s+edge=([+-]?\d+\.?\d*)%"
)
PAT_REVERSAL = re.compile(
    r"^(\S+ \S+).*REVERSAL EXIT.*?(\S+-\d+)\s+([A-Z]+)"
)


def parse_botlog_ts(s: str) -> datetime:
    # Format: 2026-05-10 04:51:08,862 — bot.log uses Pi local time (EDT).
    # All other timestamps in our data are UTC. We'll convert.
    t = datetime.strptime(s.split(",")[0], "%Y-%m-%d %H:%M:%S")
    # EDT = UTC - 4. Adjust to UTC.
    from datetime import timezone, timedelta
    return t.replace(tzinfo=timezone(timedelta(hours=-4)))


def extract_log_context(events):
    """For each SO loss_cut, find pre-cut model state in bot.log."""
    # Group target events by ticker for efficiency
    by_ticker = defaultdict(list)
    for e in events:
        by_ticker[e["ticker"]].append(e)

    # Walk bot.log once, accumulating relevant lines
    print("  Scanning bot.log...")
    lines = BOTLOG.read_text(errors="ignore").splitlines()
    for line in lines:
        # Only care about lines containing one of our SO tickers
        # (cheap pre-filter)
        m = None
        for ticker in by_ticker.keys():
            if ticker not in line:
                continue
            for pat, kind in [(PAT_SIGNAL, "signal"),
                              (PAT_LOSS, "loss_cut"),
                              (PAT_SUPPRESSED, "suppressed"),
                              (PAT_PYRAMID, "pyramid"),
                              (PAT_REVERSAL, "reversal")]:
                m = pat.search(line)
                if m:
                    ts = parse_botlog_ts(m.group(1))
                    for e in by_ticker[ticker]:
                        # within 5 min of the exit
                        if abs((ts - e["ts_exit"]).total_seconds()) < 300:
                            entry = {"ts": ts.isoformat(), "kind": kind, "raw": line.strip()[:180]}
                            if kind == "signal":
                                entry["conf"] = int(m.group(4))
                                entry["edge"] = float(m.group(5))
                            elif kind == "pyramid":
                                entry["pnl"] = float(m.group(4))
                                entry["conf"] = int(m.group(5))
                                entry["edge"] = float(m.group(6))
                            e.setdefault("log_context", []).append(entry)
                    break  # one pattern is enough
            break  # one ticker is enough
    for e in events:
        e.setdefault("log_context", [])
        e["log_context"].sort(key=lambda x: x["ts"])
    print(f"  Done — found context for {sum(1 for e in events if e['log_context'])} of {len(events)} events")


def main():
    events = load_so_loss_cuts()
    print(f"Loaded {len(events)} SO loss_cut events")
    tickers = sorted({e["ticker"] for e in events})
    print(f"Unique tickers: {len(tickers)}")

    cached = sum(1 for t in tickers if (TAPES_DIR / f"{t}.json").exists())
    print(f"  Tapes cached: {cached}/{len(tickers)} | Fetching: {len(tickers)-cached}")
    client = httpx.Client(timeout=15.0, headers={"accept": "application/json"})
    try:
        tapes = {}
        for i, t in enumerate(tickers, 1):
            tapes[t] = fetch_tape(t, client)
            if i % 5 == 0 or i == len(tickers):
                print(f"    [{i}/{len(tickers)}] cached")
    finally:
        client.close()

    # Classify each cut
    for e in events:
        e["verdict"] = classify_cut(e, tapes[e["ticker"]])

    # Add log context
    extract_log_context(events)

    # Summary
    print()
    print("═" * 100)
    print("  PANIC-FLUSH VERIFICATION — what happened on our side AFTER we cut?")
    print("═" * 100)
    klass_count = defaultdict(int)
    klass_pnl = defaultdict(float)
    strong_recover = 0
    for e in events:
        klass_count[e["verdict"]["class"]] += 1
        klass_pnl[e["verdict"]["class"]] += e["pnl"]
        if e["verdict"].get("strong_recover_in_120s"):
            strong_recover += 1

    print(f"  {'class':<20} {'n':>3} {'%':>5}   {'pnl':>9}   meaning")
    print(f"  {'─'*20} {'─'*3} {'─'*5}   {'─'*9}   ─────────────────────────────────────────────")
    desc = {
        "fast_flush":      "our side recovered within 30s of cut",
        "flush":           "our side recovered within 60s",
        "slow_flush":      "our side recovered within 2 min",
        "delayed_recover": "our side recovered within 5 min (≥+5¢ from cut)",
        "real_adverse":    "kept moving against us — eventually settled correct on time decay",
        "no_tape":         "no tape data",
    }
    n_total = len(events)
    for k in ["fast_flush", "flush", "slow_flush", "delayed_recover", "real_adverse", "no_tape"]:
        if klass_count[k] == 0: continue
        print(f"  {k:<20} {klass_count[k]:>3} {klass_count[k]/n_total*100:>4.1f}%   "
              f"${klass_pnl[k]:>+7.2f}   {desc[k]}")
    print()
    flush_total = klass_count["fast_flush"] + klass_count["flush"] + klass_count["slow_flush"]
    flush_pnl = klass_pnl["fast_flush"] + klass_pnl["flush"] + klass_pnl["slow_flush"]
    print(f"  Any flush within 2 min:    {flush_total}/{n_total} ({flush_total/n_total*100:.0f}%)   "
          f"P&L ${flush_pnl:+.2f}")
    print(f"  Strong recover (≥-10¢ of entry within 2min): {strong_recover}/{n_total} "
          f"({strong_recover/n_total*100:.0f}%)")
    print()

    # ── Show a few examples with log context ─────────────────────────────────
    print("  ── EXAMPLES (5 of each class with model trajectory) " + "─" * 32)
    for cls in ["fast_flush", "flush", "slow_flush", "real_adverse"]:
        sub = [e for e in events if e["verdict"]["class"] == cls][:3]
        if not sub: continue
        print(f"\n  ┌─ Class: {cls.upper()} ({len([e for e in events if e['verdict']['class']==cls])} total)")
        for e in sub:
            v = e["verdict"]
            print(f"  │  {e['ticker']:30} {e['side']:3}  "
                  f"entry={v['our_entry']:.0f}c cut={v['our_cut']:.0f}c "
                  f"pnl=${e['pnl']:+.2f}  result={e['result']}")
            for w, s in (("+30s", v.get("p30")), ("+60s", v.get("p60")),
                          ("+120s", v.get("p120"))):
                if s is None: continue
                print(f"  │     {w:>6}: our_max={s['our_max']:.0f}c "
                      f"our_last={s['our_last']:.0f}c  n={s['n']}")
            # Log context — last 3 entries before cut
            pre = [c for c in e["log_context"]
                   if parse_ts(c["ts"]) <= e["ts_exit"]][-3:]
            for c in pre:
                conf = c.get("conf", "")
                edge = c.get("edge", "")
                tag = f"conf={conf}% edge={edge}%" if conf != "" else ""
                age = (e["ts_exit"] - parse_ts(c["ts"])).total_seconds()
                print(f"  │     [-{age:>3.0f}s pre-cut] {c['kind']:<10} {tag}")
    print()

    OUT_JSON.write_text(json.dumps(events, indent=2, default=str))
    print(f"  Full data → {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
