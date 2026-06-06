"""Per-session summary for paper trading on Friday (Pi5).

Reads logs/friday_snapshot/trades.csv (10-col schema only — Friday is clean),
pairs entries with exits by trade_id, and emits a clean ASCII roll-up + JSON.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRADES = ROOT / "logs" / "friday_snapshot" / "trades.csv"
OUT_JSON = ROOT / "logs" / "friday_snapshot" / "session_summary.json"


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_rows():
    rows = []
    with open(TRADES) as f:
        for row in csv.reader(f):
            if len(row) != 10 or row[0] == "trade_id":
                continue
            rows.append({
                "trade_id": row[0],
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


def is_exit(side: str) -> bool:
    return side.endswith("_exit") or side.endswith("_settled")


def exit_bucket(source: str) -> str:
    s = source.lower()
    if "loss_cut" in s: return "loss_cut"
    if "profit_take" in s: return "profit_take"
    if "reversal" in s: return "reversal"
    if "settle" in s or "settled" in s: return "settled"
    if "time" in s or "expir" in s: return "time_stop"
    if "open" == s.strip(): return "OPEN"
    return "other"


def entry_bucket(source: str) -> str:
    s = source.lower()
    if "dir_early" in s: return "dir_early"
    if "dir_prime" in s: return "dir_prime"
    if "dir_late" in s: return "dir_late"
    if "scalp" in s: return "scalper"
    if "arb" in s: return "arb"
    if "snipe" in s: return "sniper"
    if "reconciled_gap" in s: return "reconciled_gap"
    if "reconciled" in s: return "reconciled"
    if "gtc_escalated" in s: return "gtc_escalated"
    if "mm_quote" in s: return "mm_quote"
    if "settlement_lock" in s: return "settlement_lock"
    return "other"


def summarize():
    rows = load_rows()
    by_session = defaultdict(list)
    for r in rows:
        by_session[r["session"]].append(r)

    sessions_out = []
    grand = {
        "entries": 0, "closed": 0, "open": 0, "wins": 0, "losses": 0, "flats": 0,
        "pnl": 0.0, "gross_cost": 0.0, "exit_mix": defaultdict(int),
        "entry_mix": defaultdict(int),
    }

    for tag in sorted(by_session.keys(),
                      key=lambda t: parse_ts("2026-" + t[:5].replace("MAY", "-05-") + "T" + t[5:] + ":00+00:00") if False else t):
        # Simple sort by parsed embedded date+time
        pass

    # We need a real chronological sort. The tag format is e.g. "10MAY05:43".
    def tag_sort_key(tag: str):
        # tag = DDMMMHH:MM  e.g. "10MAY05:43"
        day = int(tag[:2])
        mon = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
               "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}[tag[2:5]]
        hh, mm = tag[5:].split(":")
        return (mon, day, int(hh), int(mm))

    ordered_tags = sorted(by_session.keys(), key=tag_sort_key)

    for tag in ordered_tags:
        srows = sorted(by_session[tag], key=lambda r: r["ts"])
        # Group by (ticker, side_family) — handles pyramid trades where multiple
        # entries on the same ticker get exited under a single trade_id.
        # side_family: "yes" matches yes/yes_exit/yes_settled, similarly for "no".
        def side_family(s: str) -> str:
            return "yes" if s.startswith("yes") else "no"
        legs = defaultdict(lambda: {"opens": [], "closes": []})
        for r in srows:
            key = (r["ticker"], side_family(r["side"]))
            (legs[key]["closes"] if is_exit(r["side"]) else
             legs[key]["opens"]).append(r)

        entries = []
        for (ticker, fam), lg in legs.items():
            if not lg["opens"]:
                continue  # orphan exit (e.g., reconciled adoption)
            open_cost = sum(o["cost_usd"] for o in lg["opens"])
            open_qty = sum(o["contracts"] for o in lg["opens"])
            open_avg = (open_cost / open_qty * 100) if open_qty else 0
            close_cost = sum(c["cost_usd"] for c in lg["closes"])
            close_qty = sum(c["contracts"] for c in lg["closes"])
            close_avg = (close_cost / close_qty * 100) if close_qty else None
            ts_open = min(o["ts"] for o in lg["opens"])
            exit_ts = max((c["ts"] for c in lg["closes"]), default=None)
            exit_src = lg["closes"][-1]["source"] if lg["closes"] else "OPEN"
            entry_src = lg["opens"][0]["source"]
            # Position is closed only if total exit qty >= total entry qty.
            fully_closed = lg["closes"] and close_qty >= open_qty
            pnl = (close_cost - open_cost) if fully_closed else None
            n_pyramid = len(lg["opens"])
            entries.append({
                "ticker": ticker,
                "side": fam,
                "qty": open_qty,
                "n_legs": n_pyramid,
                "entry_cents": round(open_avg, 1),
                "exit_cents": round(close_avg, 1) if close_avg is not None else None,
                "entry_cost": round(open_cost, 2),
                "exit_proceeds": round(close_cost, 2) if lg["closes"] else None,
                "pnl": round(pnl, 2) if pnl is not None else None,
                "entry_src": entry_src,
                "exit_src": exit_src,
                "ts_open": ts_open.isoformat(),
                "ts_exit": exit_ts.isoformat() if exit_ts else None,
                "hold_secs": (exit_ts - ts_open).total_seconds() if exit_ts else None,
                "entry_bucket": entry_bucket(entry_src),
                "exit_bucket": exit_bucket(exit_src) if lg["closes"] else "OPEN",
            })

        wins = [e for e in entries if (e["pnl"] or 0) > 0]
        losses = [e for e in entries if (e["pnl"] or 0) < 0]
        flats = [e for e in entries if e["pnl"] is not None and e["pnl"] == 0]
        opens = [e for e in entries if e["pnl"] is None]
        closed = [e for e in entries if e["pnl"] is not None]
        pnl_total = sum(e["pnl"] for e in closed)
        gross_cost = sum(e["entry_cost"] for e in entries)
        first_ts = min(e["ts_open"] for e in entries)
        last_ts = max((e["ts_exit"] or e["ts_open"]) for e in entries)
        dur_min = (parse_ts(last_ts) - parse_ts(first_ts)).total_seconds() / 60

        exit_mix = defaultdict(int)
        entry_mix = defaultdict(int)
        for e in entries:
            entry_mix[e["entry_bucket"]] += 1
        for e in closed:
            exit_mix[e["exit_bucket"]] += 1

        sessions_out.append({
            "session": tag,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "duration_min": round(dur_min, 1),
            "entries": len(entries),
            "closed": len(closed),
            "open_unresolved": len(opens),
            "wins": len(wins),
            "losses": len(losses),
            "flats": len(flats),
            "win_rate": round(len(wins) / max(len(closed), 1) * 100, 1),
            "pnl_usd": round(pnl_total, 2),
            "gross_cost_usd": round(gross_cost, 2),
            "roi_pct": round(pnl_total / gross_cost * 100, 2) if gross_cost else 0.0,
            "best_trade": round(max((e["pnl"] for e in closed), default=0), 2),
            "worst_trade": round(min((e["pnl"] for e in closed), default=0), 2),
            "exit_mix": dict(exit_mix),
            "entry_mix": dict(entry_mix),
        })

        grand["entries"] += len(entries)
        grand["closed"] += len(closed)
        grand["open"] += len(opens)
        grand["wins"] += len(wins)
        grand["losses"] += len(losses)
        grand["flats"] += len(flats)
        grand["pnl"] += pnl_total
        grand["gross_cost"] += gross_cost
        for k, v in exit_mix.items():
            grand["exit_mix"][k] += v
        for k, v in entry_mix.items():
            grand["entry_mix"][k] += v

    return sessions_out, grand


def print_table(sessions, grand):
    print("═" * 110)
    print(f"  FRIDAY (Pi5) PAPER TRADING — {len(sessions)} SESSIONS  ({sessions[0]['first_ts'][:10]} → {sessions[-1]['last_ts'][:10]})")
    print("═" * 110)
    print(f"  {'session':12} {'dur':>5}  {'n':>3} {'cls':>3} {'op':>2} "
          f"{'W':>3} {'L':>3} {'F':>2} {'win%':>5}  {'P&L $':>8}  {'gross':>7}  {'ROI%':>6}  "
          f"{'best':>5} {'worst':>6}  exit mix")
    print("─" * 110)
    for s in sessions:
        mix = ", ".join(f"{k}:{v}" for k, v in sorted(s["exit_mix"].items(), key=lambda x: -x[1]))
        pnl = s["pnl_usd"]
        marker = "▲" if pnl > 0 else ("▼" if pnl < 0 else "·")
        print(f"  {s['session']:12} {s['duration_min']:>5.0f}m {s['entries']:>3} {s['closed']:>3} {s['open_unresolved']:>2} "
              f"{s['wins']:>3} {s['losses']:>3} {s['flats']:>2} {s['win_rate']:>4.0f}%  "
              f"{marker}${pnl:>+7.2f}  ${s['gross_cost_usd']:>5.0f}  {s['roi_pct']:>+5.1f}%  "
              f"${s['best_trade']:>+4.1f} ${s['worst_trade']:>+5.1f}  {mix}")
    print("─" * 110)
    print(f"  {'TOTAL':12} {'':>6} {grand['entries']:>3} {grand['closed']:>3} {grand['open']:>2} "
          f"{grand['wins']:>3} {grand['losses']:>3} {grand['flats']:>2} "
          f"{grand['wins']/max(grand['closed'],1)*100:>4.0f}%  "
          f"${grand['pnl']:>+7.2f}  ${grand['gross_cost']:>5.0f}  "
          f"{grand['pnl']/grand['gross_cost']*100:>+5.1f}%")
    print()
    print(f"  Exit-reason mix (all sessions): "
          + ", ".join(f"{k}:{v}" for k, v in sorted(grand['exit_mix'].items(), key=lambda x: -x[1])))
    print(f"  Entry-bucket mix (all sessions): "
          + ", ".join(f"{k}:{v}" for k, v in sorted(grand['entry_mix'].items(), key=lambda x: -x[1])))
    print()


def main():
    sessions, grand = summarize()
    print_table(sessions, grand)
    OUT_JSON.write_text(json.dumps({
        "sessions": sessions,
        "grand": {**grand,
                  "exit_mix": dict(grand["exit_mix"]),
                  "entry_mix": dict(grand["entry_mix"])},
    }, indent=2, default=str))
    print(f"  Full JSON → {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
