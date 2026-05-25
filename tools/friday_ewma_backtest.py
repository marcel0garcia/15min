"""EWMA α-sensitivity backtest against historical bot.log + cross_validate.

Reads bot.log files for SIGNAL PENDING and SIGNAL fire events. Each PENDING
line records the conf/edge values at that 1s scan (after the existing raw
threshold check). Each FIRE line is the moment a directional entry actually
went through. By replaying the PENDING-tick sequences with different EWMA
α values, we can answer:

  - At α = X, how many of the actual fires would have survived?
  - For survivors, what's the side-accuracy distribution? (joined with
    cross_validate.json outcomes)
  - What α optimizes the trade-off between trade count and quality?

Caveat: bot.log only records ticks where conf+edge raw values ALREADY
cleared the per-phase floor (that's when SIGNAL PENDING fires). We don't
see scans where conf was below threshold. So the backtest is a
"near-backtest" — directionally meaningful for relative α comparison,
but absolute fire counts assume the pre-floor filter is identical across
α values (which is approximately true).

Usage:
  python tools/friday_ewma_backtest.py                  # default friday_snapshot
  python tools/friday_ewma_backtest.py --log path.log
  python tools/friday_ewma_backtest.py --alphas 0.10,0.15,0.20,0.30
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CV_JSON = ROOT / "logs" / "friday_snapshot" / "cross_validate.json"
DEFAULT_LOGS = [ROOT / "logs" / "friday_snapshot" / "bot.log"]

PEND_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.\d]+\s+INFO\s+\S+:\s+"
    r"\[AUTO\] SIGNAL PENDING \[(\d+)/(\d+)\]:\s+(\S+)\s+(YES|NO) \| "
    r"conf=(\d+)% edge=([+-]?\d+\.\d+)%"
)
FIRE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.\d]+\s+INFO\s+\S+:\s+"
    r"\[AUTO\] SIGNAL \[([a-z]+)\|([A-Z]+)\]:\s+(\S+)\s+(YES|NO) \| "
    r"conf=(\d+)% edge=([+-]?\d+\.\d+)%"
)


def parse_log(path: Path) -> list[dict]:
    events = []
    if not path.exists():
        return events
    with open(path, errors="replace") as f:
        for line in f:
            m = PEND_RE.match(line)
            if m:
                ts, _, _, tk, side, conf, edge = m.groups()
                events.append({
                    "ts": datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"),
                    "kind": "pending",
                    "ticker": tk, "side": side.lower(),
                    "conf": int(conf) / 100, "edge": float(edge) / 100,
                })
                continue
            m = FIRE_RE.match(line)
            if m:
                ts, phase, _, tk, side, conf, edge = m.groups()
                events.append({
                    "ts": datetime.strptime(ts, "%Y-%m-%d %H:%M:%S"),
                    "kind": "fire",
                    "ticker": tk, "side": side.lower(),
                    "conf": int(conf) / 100, "edge": float(edge) / 100,
                    "phase": phase,
                })
    return events


def build_sequences(events: list[dict]) -> list[list[dict]]:
    """Group events into per-attempt sequences (PENDING ticks → FIRE).

    A sequence ends on FIRE, or on a >60s gap (attempt abandoned).
    """
    by_key = defaultdict(list)
    for ev in events:
        by_key[(ev["ticker"], ev["side"])].append(ev)
    sequences = []
    for key, lst in by_key.items():
        lst.sort(key=lambda e: e["ts"])
        cur, last_ts = [], None
        for ev in lst:
            if last_ts and (ev["ts"] - last_ts).total_seconds() > 60:
                if cur:
                    sequences.append(cur)
                cur = []
            cur.append(ev)
            last_ts = ev["ts"]
            if ev["kind"] == "fire":
                sequences.append(cur)
                cur, last_ts = [], None
        if cur:
            sequences.append(cur)
    return sequences


def replay_ewma(seq: list[dict], alpha: float, stale_sec: float = 5.0) -> dict:
    sm_conf = sm_edge = None
    last_ts = None
    for ev in seq:
        ts = ev["ts"]
        if (sm_conf is None
                or (last_ts and (ts - last_ts).total_seconds() > stale_sec)):
            sm_conf, sm_edge = ev["conf"], ev["edge"]
        else:
            sm_conf = alpha * ev["conf"] + (1 - alpha) * sm_conf
            sm_edge = alpha * ev["edge"] + (1 - alpha) * sm_edge
        last_ts = ts
    fired = seq[-1]["kind"] == "fire"
    return {
        "final_conf": sm_conf,
        "final_edge": sm_edge,
        "fired": fired,
        "fire_conf": seq[-1]["conf"] if fired else None,
        "fire_edge": seq[-1]["edge"] if fired else None,
        "ticker": seq[-1]["ticker"],
        "side": seq[-1]["side"],
        "phase": seq[-1].get("phase"),
        "n_pending": sum(1 for e in seq if e["kind"] == "pending"),
    }


def load_outcomes() -> dict[str, dict]:
    if not CV_JSON.exists():
        return {}
    cv = json.loads(CV_JSON.read_text())
    out = {}
    for sess in cv.values():
        for p in sess.get("positions", []):
            out[p["ticker"]] = {
                "result": p.get("result"),
                "side": p.get("side"),
                "class": p.get("class"),
                "pnl": p.get("pnl"),
            }
    return out


def survives(replay: dict, fire_conf_relax: float = 0.95) -> bool:
    """Would this sequence have fired under the given α?

    Approximation: yes if it ORIGINALLY fired AND the smoothed conf at the
    final tick is at least fire_conf_relax × original_fire_conf.
    """
    if not replay["fired"] or replay["final_conf"] is None:
        return False
    return replay["final_conf"] >= replay["fire_conf"] * fire_conf_relax


def main():
    ap = argparse.ArgumentParser(description="EWMA α-sensitivity backtest")
    ap.add_argument("--log", action="append",
                    help="Path to bot.log (repeatable). Default: friday_snapshot/bot.log")
    ap.add_argument("--alphas", default="0.00,0.10,0.15,0.20,0.25,0.30,0.40",
                    help="Comma-separated α values to compare")
    ap.add_argument("--stale-sec", type=float, default=5.0,
                    help="Reset smoothing after this many seconds of silence")
    ap.add_argument("--fire-conf-relax", type=float, default=0.95,
                    help="Relaxation factor against original fire conf "
                         "(default 0.95 = require smoothed >= 95%% of orig)")
    args = ap.parse_args()

    log_paths = [Path(p) for p in (args.log or DEFAULT_LOGS)]
    alphas = [float(a) for a in args.alphas.split(",")]

    all_events = []
    for p in log_paths:
        evs = parse_log(p)
        print(f"  Parsed {len(evs):,} signal events from {p.name}")
        all_events.extend(evs)
    print(f"  Total events: {len(all_events):,}")

    sequences = build_sequences(all_events)
    print(f"  Attempt-sequences: {len(sequences):,}")

    outcomes = load_outcomes()
    print(f"  Outcomes available for {len(outcomes)} tickers")
    print()

    actual_fires_all = [s for s in sequences if s[-1]["kind"] == "fire"]
    # Reversal re-entries and pyramid adds BYPASS the 3-tick gate, so their
    # SIGNAL fires have no PENDING ticks preceding them. EWMA matters only
    # for fires that went through the gate -- i.e., those with prior PENDING
    # observations. Filter to those.
    actual_fires = [s for s in actual_fires_all
                    if sum(1 for e in s if e["kind"] == "pending") >= 1]
    bypass = len(actual_fires_all) - len(actual_fires)
    print(f"  Of {len(actual_fires_all)} fires, {bypass} are reversal/pyramid bypasses")
    print(f"  (no preceding PENDING; EWMA can't affect them in this backtest).")
    print(f"  Analyzing the {len(actual_fires)} fires that DID go through the gate:")
    print()

    print("=" * 86)
    print(" EWMA α-sensitivity: how many gate-fires survive, outcome mix?")
    print("=" * 86)
    print(f"  {'α':>5}  {'fires':>6}  {'survive':>7}  {'pct':>5}  "
          f"{'CW':>4} {'SO':>4} {'SE':>4} {'WL':>4}  {'side%':>6}  "
          f"{'med_ticks':>9}")
    print("-" * 86)

    n_fires = len(actual_fires)

    rows = []
    for α in alphas:
        survived = []
        for seq in actual_fires:
            r = replay_ewma(seq, α, args.stale_sec)
            if survives(r, args.fire_conf_relax):
                survived.append(r)
        cls_counts = defaultdict(int)
        for r in survived:
            cls_counts[outcomes.get(r["ticker"], {}).get("class", "?")] += 1
        cw = cls_counts["correct_win"]
        so = cls_counts["shaken_out"]
        se = cls_counts["saved_by_exit"]
        wl = cls_counts["wrong_loss"]
        rated = cw + so + se + wl
        side_pct = (cw + so) / rated * 100 if rated else 0
        n_ticks = sorted(r["n_pending"] for r in survived)
        median_ticks = n_ticks[len(n_ticks) // 2] if n_ticks else 0
        pct = len(survived) / n_fires * 100 if n_fires else 0
        marker = "  ← current default" if abs(α - 0.20) < 0.001 else ""
        if abs(α - 0.00) < 0.001:
            marker = "  ← no smoothing (current pre-EWMA behavior)"
        print(f"  {α:>5.2f}  {n_fires:>6}  {len(survived):>7}  {pct:>4.0f}%  "
              f"{cw:>4} {so:>4} {se:>4} {wl:>4}  {side_pct:>5.1f}%  "
              f"{median_ticks:>9d}{marker}")
        rows.append((α, len(survived), side_pct, rated))

    print()
    print("Reading the table:")
    print("  - 'survive' = sequences whose EWMA-smoothed final conf still")
    print(f"     clears (original fire conf × {args.fire_conf_relax})")
    print("  - 'side%' = (CW + SO) / (CW + SO + SE + WL) of survivors")
    print("  - heavier smoothing (lower α) filters more fires but keeps the")
    print("     higher-conviction subset → typically higher side accuracy")
    print()

    # Recommendation
    good = [(α, n, s, r) for α, n, s, r in rows if r >= 30 and α > 0]
    if good:
        # Optimize a simple score: side_pct (penalize fewer survivors mildly)
        good.sort(key=lambda x: x[2], reverse=True)
        print(f"  Best α by side accuracy (≥30 survivors, excluding α=0):")
        for α, n, s, r in good[:3]:
            print(f"    α={α:.2f}: side%={s:.1f}%  survivors={n}")
    if not outcomes:
        print("  (No cross_validate.json found — outcomes joined columns will be 0)")


if __name__ == "__main__":
    main()
