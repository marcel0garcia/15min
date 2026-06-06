"""Diagnostic for the 67 'shaken_out' trades — test 4 hypotheses about why
the bot exits correct-side positions before they settle as winners.

Hypotheses tested:
  A. Exit-reason mix: reversal-dominated vs loss_cut-dominated?
  B. Stop tightness: SO entries clustered in low-conf / low-edge band?
  C. Late-window panic: SO exits cluster near settlement?
  D. Pyramid amplification: SO trades have n_legs > 1?
  E. (Bonus) Did the bot re-enter and recover after the shake-out?

Compares SO vs CW (correct_win) as the natural baseline — both picked the
right side; they only differ in how we exited.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRADES = ROOT / "logs" / "friday_snapshot" / "trades.csv"
CACHE = ROOT / "logs" / "friday_snapshot" / "market_results.json"
XVAL = ROOT / "logs" / "friday_snapshot" / "cross_validate.json"

MIN_SESSION_MIN = 60


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def side_family(s: str) -> str:
    return "yes" if s.startswith("yes") else "no"


def is_exit(side: str) -> bool:
    return side.endswith("_exit") or side.endswith("_settled")


def exit_bucket(source: str) -> str:
    s = source.lower()
    if "loss_cut" in s: return "loss_cut"
    if "profit_take" in s: return "profit_take"
    if "reversal" in s: return "reversal"
    if "settled" in s or "settlement" in s: return "settled"
    if "time" in s or "expir" in s: return "time_stop"
    return "other"


def parse_conf_edge(source: str):
    """Extract conf=NN% and edge=±NN.N% from a source string."""
    conf = re.search(r"conf=(\d+)%", source)
    edge = re.search(r"edge=([+-]?\d+\.?\d*)%", source)
    return (int(conf.group(1)) if conf else None,
            float(edge.group(1)) if edge else None)


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


def build_session_durations(rows):
    by_sess = defaultdict(lambda: [None, None])
    for r in rows:
        if by_sess[r["session"]][0] is None or r["ts"] < by_sess[r["session"]][0]:
            by_sess[r["session"]][0] = r["ts"]
        if by_sess[r["session"]][1] is None or r["ts"] > by_sess[r["session"]][1]:
            by_sess[r["session"]][1] = r["ts"]
    return {s: (hi - lo).total_seconds() / 60 for s, (lo, hi) in by_sess.items()}


def build_positions(rows, kept_sessions):
    """Group entries by (session, ticker, side_family) — same as cross_validate."""
    legs = defaultdict(lambda: {"opens": [], "closes": []})
    for r in sorted(rows, key=lambda r: r["ts"]):
        if r["session"] not in kept_sessions:
            continue
        key = (r["session"], r["ticker"], side_family(r["side"]))
        (legs[key]["closes"] if is_exit(r["side"]) else
         legs[key]["opens"]).append(r)
    positions = []
    for (sess, ticker, fam), lg in legs.items():
        if not lg["opens"]:
            continue
        open_cost = sum(o["cost_usd"] for o in lg["opens"])
        open_qty = sum(o["contracts"] for o in lg["opens"])
        close_cost = sum(c["cost_usd"] for c in lg["closes"])
        close_qty = sum(c["contracts"] for c in lg["closes"])
        ts_open = min(o["ts"] for o in lg["opens"])
        ts_exit = max((c["ts"] for c in lg["closes"]), default=None)
        first_entry = sorted(lg["opens"], key=lambda o: o["ts"])[0]
        conf, edge = parse_conf_edge(first_entry["source"])
        fully_closed = bool(lg["closes"]) and close_qty >= open_qty
        pnl = (close_cost - open_cost) if fully_closed else None
        exit_src = lg["closes"][-1]["source"] if lg["closes"] else "OPEN"
        positions.append({
            "session": sess,
            "ticker": ticker,
            "side": fam,
            "qty": open_qty,
            "n_legs": len(lg["opens"]),
            "entry_avg": (open_cost / open_qty * 100) if open_qty else 0,
            "exit_avg": (close_cost / close_qty * 100) if close_qty else None,
            "open_cost": open_cost,
            "pnl": pnl,
            "ts_open": ts_open,
            "ts_exit": ts_exit,
            "hold_secs": (ts_exit - ts_open).total_seconds() if ts_exit else None,
            "entry_conf": conf,
            "entry_edge": edge,
            "entry_src": first_entry["source"],
            "exit_src": exit_src,
            "exit_bucket": exit_bucket(exit_src) if lg["closes"] else "OPEN",
        })
    return positions


def classify(positions, market_results):
    for p in positions:
        mr = market_results.get(p["ticker"], {}) or {}
        result = mr.get("result")
        close_time = mr.get("close_time")
        p["result"] = result
        p["close_time"] = parse_ts(close_time) if close_time else None
        if p["pnl"] is None or result not in ("yes", "no"):
            p["correct"] = None
            p["class"] = "open" if p["pnl"] is None else "unknown"
            continue
        correct = (p["side"] == result)
        p["correct"] = correct
        if correct and p["pnl"] > 0:    p["class"] = "correct_win"
        elif correct:                    p["class"] = "shaken_out"
        elif p["pnl"] > 0:               p["class"] = "saved_by_exit"
        else:                            p["class"] = "wrong_loss"
        # Time from exit to settlement (None if no close_time)
        if p["ts_exit"] and p["close_time"]:
            p["secs_to_settle"] = (p["close_time"] - p["ts_exit"]).total_seconds()
        else:
            p["secs_to_settle"] = None
    return positions


def detect_re_entries(positions):
    """For each SO trade, look for a same-session entry on the same (ticker,side)
    AFTER our exit. If we re-entered on the correct side, did we recover?"""
    by_key = defaultdict(list)
    for p in positions:
        by_key[(p["session"], p["ticker"], p["side"])].append(p)
    for plist in by_key.values():
        plist.sort(key=lambda p: p["ts_open"])
        # If multiple entries-by-(ticker,side) exist within a session, they're
        # already merged by build_positions. So re-entry detection means: did the
        # bot trade the *opposite* side after a SO exit?
    # Different angle: did we open a NO position right after a YES SO (or vice versa)?
    for p in positions:
        if p["class"] != "shaken_out":
            continue
        opp = "no" if p["side"] == "yes" else "yes"
        opp_entries = [q for q in by_key.get((p["session"], p["ticker"], opp), [])
                       if q["ts_open"] >= p["ts_exit"]]
        if opp_entries:
            o = opp_entries[0]
            p["reentered_opp"] = True
            p["reentry_pnl"] = o["pnl"]
            p["reentry_correct"] = o.get("correct")
        else:
            p["reentered_opp"] = False
    return positions


def section(title):
    print(f"\n  ── {title} " + "─" * (80 - len(title)))


def pct_dist(items, key, all_vals=None):
    counts = defaultdict(int)
    for it in items:
        counts[it.get(key, "?")] += 1
    total = max(len(items), 1)
    rows = sorted(counts.items(), key=lambda x: -x[1])
    return [(k, v, v/total*100) for k, v in rows]


def quantiles(vals, qs=(0.25, 0.5, 0.75)):
    if not vals: return [None]*len(qs)
    s = sorted(vals)
    return [s[int(q * (len(s)-1))] for q in qs]


def main():
    rows = load_rows()
    sess_dur = build_session_durations(rows)
    kept = {s for s, d in sess_dur.items() if d >= MIN_SESSION_MIN}
    market_results = json.loads(CACHE.read_text())
    positions = build_positions(rows, kept)
    positions = classify(positions, market_results)
    positions = detect_re_entries(positions)

    so = [p for p in positions if p["class"] == "shaken_out"]
    cw = [p for p in positions if p["class"] == "correct_win"]
    wl = [p for p in positions if p["class"] == "wrong_loss"]

    print("═" * 86)
    print(f"  SO DIAGNOSTIC — {len(so)} shaken-out trades vs {len(cw)} correct-win baseline")
    print("═" * 86)
    print(f"  SO total P&L: ${sum(p['pnl'] for p in so):+.2f}   "
          f"avg ${sum(p['pnl'] for p in so)/len(so):+.2f}")

    # ── Hypothesis A: exit-reason mix ──────────────────────────────────────
    section("A. Exit-reason mix (SO vs CW)")
    print(f"  {'reason':<12} {'SO':>6} {'%':>6}     {'CW':>6} {'%':>6}")
    so_buckets = pct_dist(so, "exit_bucket")
    cw_buckets = {k: (v, p) for k, v, p in pct_dist(cw, "exit_bucket")}
    seen = set()
    for k, v, p in so_buckets:
        seen.add(k)
        cv, cp = cw_buckets.get(k, (0, 0.0))
        print(f"  {k:<12} {v:>6} {p:>5.1f}%   {cv:>6} {cp:>5.1f}%")
    for k, (cv, cp) in cw_buckets.items():
        if k not in seen:
            print(f"  {k:<12} {0:>6} {0.0:>5.1f}%   {cv:>6} {cp:>5.1f}%")

    # ── Hypothesis B: entry confidence + edge ──────────────────────────────
    section("B. Entry confidence and edge — were SO trades thinner setups?")
    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals: return "n/a"
        q1, med, q3 = quantiles(vals)
        return f"n={len(vals):>3}  med={med:>5.1f}  IQR=[{q1:.1f}, {q3:.1f}]  mean={statistics.mean(vals):.1f}"
    print(f"  SO entry_conf:   {stats([p['entry_conf'] for p in so])}")
    print(f"  CW entry_conf:   {stats([p['entry_conf'] for p in cw])}")
    print(f"  WL entry_conf:   {stats([p['entry_conf'] for p in wl])}")
    print()
    print(f"  SO entry_edge:   {stats([p['entry_edge'] for p in so])}")
    print(f"  CW entry_edge:   {stats([p['entry_edge'] for p in cw])}")
    print(f"  WL entry_edge:   {stats([p['entry_edge'] for p in wl])}")

    # ── Hypothesis C: time-to-settlement at exit ───────────────────────────
    section("C. Time from SO exit to settlement — late-window panic?")
    sts = [p["secs_to_settle"] for p in so if p["secs_to_settle"] is not None]
    print(f"  Distribution of seconds-to-settle at the moment we exited:")
    if sts:
        q1, med, q3 = quantiles(sts)
        print(f"    n={len(sts)}  median={med:>5.0f}s  IQR=[{q1:.0f}, {q3:.0f}]s  "
              f"max={max(sts):.0f}s")
    # Bucketed
    buckets = {"<60s": 0, "60-180s": 0, "180-450s": 0, "450-900s": 0, ">900s (early)": 0}
    for s in sts:
        if   s < 60:  buckets["<60s"] += 1
        elif s < 180: buckets["60-180s"] += 1
        elif s < 450: buckets["180-450s"] += 1
        elif s < 900: buckets["450-900s"] += 1
        else:         buckets[">900s (early)"] += 1
    print(f"  {'window':<18} {'n':>3} {'%':>6}   pnl_contribution")
    for k, v in buckets.items():
        pnl_in_bucket = sum(p["pnl"] for p in so
                            if p["secs_to_settle"] is not None
                            and _in_bucket(p["secs_to_settle"], k))
        print(f"  {k:<18} {v:>3} {v/max(len(sts),1)*100:>5.1f}%   ${pnl_in_bucket:+.2f}")

    # ── Hypothesis D: pyramid amplification ────────────────────────────────
    section("D. Pyramid amplification — were SO trades multi-leg pyramids?")
    def n_legs_dist(group):
        d = defaultdict(int)
        for p in group:
            d[p["n_legs"]] += 1
        return d
    so_legs = n_legs_dist(so)
    cw_legs = n_legs_dist(cw)
    print(f"  {'n_legs':>7} {'SO_n':>5} {'SO_%':>6} {'SO_pnl':>9}    {'CW_n':>5} {'CW_%':>6} {'CW_pnl':>9}")
    keys = sorted(set(so_legs) | set(cw_legs))
    for k in keys:
        so_n = so_legs.get(k, 0); cw_n = cw_legs.get(k, 0)
        so_pnl = sum(p["pnl"] for p in so if p["n_legs"] == k)
        cw_pnl = sum(p["pnl"] for p in cw if p["n_legs"] == k)
        print(f"  {k:>7} {so_n:>5} {so_n/max(len(so),1)*100:>5.1f}% ${so_pnl:>+7.2f}    "
              f"{cw_n:>5} {cw_n/max(len(cw),1)*100:>5.1f}% ${cw_pnl:>+7.2f}")

    # ── Hypothesis E: re-entry behavior after shake-out ────────────────────
    section("E. After the shake-out — did the bot flip and recover?")
    flipped = [p for p in so if p.get("reentered_opp")]
    recovered = [p for p in flipped if (p.get("reentry_pnl") or 0) > 0]
    flip_to_wrong = [p for p in flipped if p.get("reentry_correct") is False]
    print(f"  SO trades total: {len(so)}")
    print(f"  → flipped to opposite side after exit:  {len(flipped)} ({len(flipped)/max(len(so),1)*100:.1f}%)")
    print(f"     of those, flip was on the WRONG side: {len(flip_to_wrong)}  "
          f"(compounds the loss — original SO side was correct, then flipped wrong)")
    print(f"     of those, flip recovered profit:     {len(recovered)}")
    flip_pnl = sum((p.get("reentry_pnl") or 0) for p in flipped)
    print(f"  Net P&L from those re-entry flips:      ${flip_pnl:+.2f}")
    so_pnl = sum(p["pnl"] for p in so)
    print(f"  Original SO loss:  ${so_pnl:+.2f}")
    print(f"  Combined (SO + flip):  ${so_pnl + flip_pnl:+.2f}")

    # ── Hold time as cross-check ───────────────────────────────────────────
    section("F. Hold time — did we exit faster than winners?")
    so_hold = [p["hold_secs"] for p in so if p["hold_secs"] is not None]
    cw_hold = [p["hold_secs"] for p in cw if p["hold_secs"] is not None]
    if so_hold and cw_hold:
        q1, med_so, q3 = quantiles(so_hold)
        _, med_cw, _ = quantiles(cw_hold)
        print(f"  SO median hold: {med_so:>5.0f}s  (IQR {q1:.0f}-{q3:.0f}s, n={len(so_hold)})")
        print(f"  CW median hold: {med_cw:>5.0f}s  (n={len(cw_hold)})")

    print()


def _in_bucket(s, label):
    if label == "<60s": return s < 60
    if label == "60-180s": return 60 <= s < 180
    if label == "180-450s": return 180 <= s < 450
    if label == "450-900s": return 450 <= s < 900
    if label == ">900s (early)": return s >= 900
    return False


if __name__ == "__main__":
    main()
