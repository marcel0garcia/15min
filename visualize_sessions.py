"""
Side-by-side visualization of session 12APR06:15 (pre-fix) vs 12APR20:43 (post-fix).
Generates session_compare.png + prints an ASCII summary.
"""
from __future__ import annotations

import csv
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

ROOT = Path(__file__).parent
TRADES = ROOT / "logs" / "trades.csv"
BOTLOG = ROOT / "logs" / "bot.log"
BASE = "https://api.elections.kalshi.com/trade-api/v2"

SESSIONS = {
    "PRE-FIX  (12APR06:15)": ["12APR06:15", "12APR05:28"],
    "POST-FIX (12APR20:43)": ["12APR20:43", "12APR20:20"],
}

_client = httpx.Client(timeout=10.0, headers={"accept": "application/json"})


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load(tags):
    rows = []
    with open(TRADES) as f:
        for row in csv.reader(f):
            if len(row) < 10 or row[9] not in tags:
                continue
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
            })
    rows.sort(key=lambda r: r["ts"])
    return rows


def fetch_result(ticker):
    r = _client.get(f"{BASE}/markets/{ticker}")
    if r.status_code != 200:
        return None
    return (r.json().get("market") or {}).get("result")


def session_stats(rows):
    """Build per-ticker P&L + per-entry correct-side judgements.

    Returns (stats_dict, cumulative_timeline, entries_detail)
    """
    tickers = {r["ticker"] for r in rows}
    results = {t: fetch_result(t) for t in tickers}

    # Pair opens with matching closes by order_id
    orders = defaultdict(lambda: {"opens": [], "closes": []})
    for r in rows:
        if r["side"].endswith("_exit") or r["side"].endswith("_settled"):
            orders[r["order_id"]]["closes"].append(r)
        else:
            orders[r["order_id"]]["opens"].append(r)

    entries = []
    for oid, legs in orders.items():
        for op in legs["opens"]:
            close_cost = sum(c["cost_usd"] for c in legs["closes"]) or 0
            close_contracts = sum(c["contracts"] for c in legs["closes"]) or 0
            close_avg = (close_cost / close_contracts * 100) if close_contracts else None
            pnl = (close_cost - op["cost_usd"]) if legs["closes"] else None
            exit_ts = max((c["ts"] for c in legs["closes"]), default=None)
            entries.append({
                "ts_open": op["ts"],
                "ts_exit": exit_ts,
                "ticker": op["ticker"],
                "side": op["side"],
                "contracts": op["contracts"],
                "entry": op["price_cents"],
                "exit": close_avg,
                "pnl": pnl,
                "result": results.get(op["ticker"]),
                "source": op["source"],
            })

    # Cumulative P&L timeline (event-based)
    timeline = []
    cum = 0.0
    sorted_e = sorted([e for e in entries if e["pnl"] is not None], key=lambda e: e["ts_exit"] or e["ts_open"])
    for e in sorted_e:
        cum += e["pnl"]
        timeline.append((e["ts_exit"] or e["ts_open"], cum))

    # Aggregates
    wins = [e for e in entries if (e["pnl"] or 0) > 0]
    losses = [e for e in entries if (e["pnl"] or 0) < 0]
    correct = [e for e in entries if e["result"] and e["side"] == e["result"]]
    wrong = [e for e in entries if e["result"] and e["side"] != e["result"]]
    total_pnl = sum((e["pnl"] or 0) for e in entries)

    return {
        "n": len(entries),
        "wins": len(wins),
        "losses": len(losses),
        "correct_side": len(correct),
        "wrong_side": len(wrong),
        "pnl": total_pnl,
        "timeline": timeline,
        "entries": entries,
    }


def count_gate_events(tags):
    """Count post-fix gate interventions (my new log lines)."""
    if not BOTLOG.exists():
        return {"drift_skip": 0, "drift_halve": 0, "reversal_skip": 0}
    # Find the bot.log local-time range that corresponds to the session tags.
    # The session tag is MMDDhh:mm in UTC — we don't know bot.log tz exactly,
    # so just count occurrences in the whole file and the user can eyeball.
    text = BOTLOG.read_text(errors="ignore")
    return {
        "drift_skip": len(re.findall(r"GTC→IOC SKIPPED", text)),
        "drift_halve": len(re.findall(r"DRIFT HALVE", text)),
        "reversal_skip": len(re.findall(r"REVERSAL RE-ENTRY SKIPPED", text)),
    }


def render_ascii_comparison(results: dict):
    print()
    print("═" * 72)
    print("  SIDE-BY-SIDE SESSION COMPARISON")
    print("═" * 72)
    hdr = f"{'metric':32s}" + "".join(f"{k:>20s}" for k in results)
    print(hdr)
    print("─" * len(hdr))
    for metric, key, fmt in [
        ("Entries",              "n",            "{:>20d}"),
        ("Wins",                 "wins",         "{:>20d}"),
        ("Losses",               "losses",       "{:>20d}"),
        ("Correct side",         "correct_side", "{:>20d}"),
        ("Wrong side",           "wrong_side",   "{:>20d}"),
        ("Realized P&L ($)",     "pnl",          "{:>+20.2f}"),
    ]:
        row = f"{metric:32s}" + "".join(fmt.format(s[key]) for s in results.values())
        print(row)
    # Rates
    side_acc = lambda s: s["correct_side"] / max(s["correct_side"] + s["wrong_side"], 1) * 100
    win_rate = lambda s: s["wins"] / max(s["wins"] + s["losses"], 1) * 100
    print(f"{'Side accuracy (%)':32s}" +
          "".join(f"{side_acc(s):>19.1f}%" for s in results.values()))
    print(f"{'Win rate (%)':32s}" +
          "".join(f"{win_rate(s):>19.1f}%" for s in results.values()))
    print()


def render_ascii_timeline(label, timeline, width=68):
    if not timeline:
        print(f"  [{label}] no closed trades")
        return
    pnls = [p for _, p in timeline]
    lo, hi = min(0.0, *pnls), max(0.0, *pnls)
    span = hi - lo or 1.0
    zero_row = int(round((hi - 0) / span * 10))

    print(f"\n  CUMULATIVE P&L TIMELINE — {label}")
    print(f"  ${hi:+.2f} ┐")
    grid = [[" "] * width for _ in range(11)]
    for i, (_, pnl) in enumerate(timeline):
        col = int(i / max(len(timeline) - 1, 1) * (width - 1))
        row = int(round((hi - pnl) / span * 10))
        row = max(0, min(10, row))
        # Draw step
        grid[row][col] = "●"
        # Vertical connector to zero line (optional)
    # Zero line
    for c in range(width):
        if grid[zero_row][c] == " ":
            grid[zero_row][c] = "─"
    for r, line in enumerate(grid):
        marker = ""
        if r == 0:
            marker = ""
        elif r == 10:
            marker = ""
        elif r == zero_row:
            marker = "  $0.00"
        print("        │" + "".join(line) + "│" + marker)
    print(f"  ${lo:+.2f} ┘" + "─" * (width - 2) + " (last: ${:+.2f})".format(pnls[-1]))


def build_png(results, gate_counts, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("15-min BTC Bot — Session Comparison (pre-fix vs post-fix)",
                 fontsize=15, fontweight="bold")

    # ── (0,0) Cumulative P&L timeline ─────────────────────────
    ax = axes[0][0]
    colors = {"PRE-FIX  (12APR06:15)": "#c62828",
              "POST-FIX (12APR20:43)": "#1565c0"}
    for label, s in results.items():
        if not s["timeline"]:
            continue
        # normalize time as minutes-since-first-trade
        t0 = s["timeline"][0][0]
        xs = [(t - t0).total_seconds() / 60 for t, _ in s["timeline"]]
        ys = [p for _, p in s["timeline"]]
        # add origin point
        xs = [0] + xs
        ys = [0] + ys
        ax.plot(xs, ys, marker="o", linewidth=2, label=label,
                color=colors.get(label, "gray"))
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Cumulative realized P&L (entry/exit pairs)")
    ax.set_xlabel("Minutes into session")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # ── (0,1) Side accuracy bars ──────────────────────────────
    ax = axes[0][1]
    labels = list(results.keys())
    correct = [results[l]["correct_side"] for l in labels]
    wrong = [results[l]["wrong_side"] for l in labels]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, correct, w, label="Correct side", color="#2e7d32")
    ax.bar(x + w/2, wrong, w, label="Wrong side", color="#c62828")
    # Annotate %
    for i, l in enumerate(labels):
        total = correct[i] + wrong[i]
        pct = (correct[i] / total * 100) if total else 0
        ax.text(i, max(correct[i], wrong[i]) + 0.5,
                f"{pct:.0f}% correct", ha="center", fontweight="bold")
    ax.set_xticks(x, [l.split(" ")[0] for l in labels])
    ax.set_title("Directional side accuracy vs final settlement")
    ax.set_ylabel("Entries")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # ── (1,0) Win/loss counts and P&L ─────────────────────────
    ax = axes[1][0]
    wins = [results[l]["wins"] for l in labels]
    losses = [results[l]["losses"] for l in labels]
    ax.bar(x - w/2, wins, w, label="Wins", color="#2e7d32")
    ax.bar(x + w/2, losses, w, label="Losses", color="#c62828")
    for i, l in enumerate(labels):
        pnl = results[l]["pnl"]
        colour = "#2e7d32" if pnl >= 0 else "#c62828"
        ax.text(i, max(wins[i], losses[i]) + 0.4,
                f"P&L: ${pnl:+.2f}", ha="center", fontweight="bold", color=colour)
    ax.set_xticks(x, [l.split(" ")[0] for l in labels])
    ax.set_title("Wins vs Losses per session (closed legs)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # ── (1,1) Gate interventions ──────────────────────────────
    ax = axes[1][1]
    events = ["GTC→IOC\nSKIPPED", "DRIFT\nHALVE", "REVERSAL\nRE-ENTRY\nSKIPPED"]
    counts = [gate_counts["drift_skip"], gate_counts["drift_halve"], gate_counts["reversal_skip"]]
    bars = ax.bar(events, counts,
                  color=["#c62828", "#ef6c00", "#6a1b9a"])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, c + 0.05, str(c),
                ha="center", fontweight="bold", fontsize=12)
    ax.set_title("New gates — trades SKIPPED by the post-fix logic\n(total across all activity post-deploy)")
    ax.set_ylabel("Events")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"\n  PNG saved → {out_path}")


def main():
    all_results = {}
    for label, tags in SESSIONS.items():
        rows = load(tags)
        if not rows:
            print(f"[{label}] no trades for {tags}")
            continue
        print(f"\n● Analyzing {label} — {len(rows)} CSV rows …")
        all_results[label] = session_stats(rows)
        time.sleep(0.2)

    gate_counts = count_gate_events(None)

    render_ascii_comparison(all_results)

    for label, s in all_results.items():
        render_ascii_timeline(label, s["timeline"])

    # Extra: enumerate per-entry for the new session
    post = "POST-FIX (12APR20:43)"
    if post in all_results:
        print(f"\n  POST-FIX SESSION — PER-ENTRY DETAIL\n")
        print(f"  {'time':8s}  {'ticker':27s} {'side':3s} {'entry':>5} {'exit':>4} "
              f"{'pnl$':>6} {'res':3s} ok  source")
        for e in sorted(all_results[post]["entries"], key=lambda x: x["ts_open"]):
            ok = "✓" if e["result"] and e["side"] == e["result"] else ("✗" if e["result"] else "?")
            exit_str = f"{int(e['exit'])}¢" if e["exit"] is not None else "—"
            pnl_str = f"{e['pnl']:+.2f}" if e["pnl"] is not None else "—"
            print(f"  {e['ts_open'].strftime('%H:%M:%S')}  {e['ticker']:27s} "
                  f"{e['side'][:3]:3s} {e['entry']:>4}¢ {exit_str:>4} "
                  f"{pnl_str:>6} {(e['result'] or '?')[:3]:3s} {ok:2} {e['source'][:50]}")

    print(f"\n  GATE INTERVENTIONS (all-time in bot.log, since deploy):")
    print(f"    GTC→IOC SKIPPED (drift > 5¢):   {gate_counts['drift_skip']}")
    print(f"    DRIFT HALVE (drift > 2¢):       {gate_counts['drift_halve']}")
    print(f"    REVERSAL RE-ENTRY SKIPPED:      {gate_counts['reversal_skip']}")

    out = ROOT / "session_compare.png"
    build_png(all_results, gate_counts, out)


if __name__ == "__main__":
    main()
