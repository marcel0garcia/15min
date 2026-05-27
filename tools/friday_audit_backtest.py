"""Entry-strategy audit backtest.

Single comprehensive tool that answers the questions the iterative
filter-tweaking has been failing to answer:

  1. Is the ensemble's prob_yes calibrated? (Brier score, reliability diagram)
  2. Are our filters rejecting more good entries than bad? (per-filter
     contribution analysis using hypothetical outcomes of rejected fires)
  3. Are there regimes (phase, flow, conf bucket) where the model is
     reliably calibrated vs purely noise?
  4. How much edge are we leaving on the table per market vs capturing?
  5. Based on the data, which filters should we drop, keep, tighten, loosen?

Pure backtest — reads existing bot.log + cross_validate.json + market_tapes/
cache. No code or behavior changes. Designed to answer "do we even know
our probability signal is real" before we redesign anything.

Usage:
  python tools/friday_audit_backtest.py
  python tools/friday_audit_backtest.py --log path/to/bot.log
  python tools/friday_audit_backtest.py --section calibration
  python tools/friday_audit_backtest.py --quiet     # report only, no progress
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "logs" / "friday_snapshot"
BOT_LOG = SNAP / "bot.log"
CV_JSON = SNAP / "cross_validate.json"
TAPES_DIR = SNAP / "market_tapes"

# Phase boundaries — match the bot's min_confidence_by_phase / loss-cut tiers
PHASE_BOUNDARIES = [
    ("early", 540, 9999),   # >540s remaining (0-6 min from open)
    ("mid",   300,  540),   # 6-10 min remaining
    ("prime", 180,  300),   # 10-12 min
    ("late",    0,  180),   # last 3 min
]


# ─── Log parsing ──────────────────────────────────────────────────────────────

# SIGNAL fire line. Has optional (raw X%/Y%) and optional flow tag.
FIRE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.\d]+\s+INFO\s+\S+:\s+"
    r"\[AUTO\] SIGNAL \[([a-z]+)\|([A-Z]+)\]:\s+(\S+)\s+(YES|NO) \| "
    r"conf=(\d+)% edge=([+-]?\d+\.\d+)%"
    r"(?:\s*\(raw (\d+)%/([+-]?\d+\.\d+)%\))?"
    r".*?mid=([\d.]+)¢"
    r"(?:.*flow=yes:([\d.]+)/no:([\d.]+)\s+net=([+-]?\d+\.\d+))?"
)
# SIGNAL PENDING line
PEND_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})[,.\d]+\s+INFO\s+\S+:\s+"
    r"\[AUTO\] SIGNAL PENDING \[(\d+)/(\d+)\]:\s+(\S+)\s+(YES|NO) \| "
    r"conf=(\d+)% edge=([+-]?\d+\.\d+)%"
)
REJECT_RES = {
    "FLOW MISALIGNMENT": re.compile(
        r"\[AUTO\] FLOW MISALIGNMENT:\s+(\S+)\s+(YES|NO)"
    ),
    "SIGNAL FADED": re.compile(
        r"\[AUTO\] SIGNAL FADED:\s+(\S+)\s+(YES|NO)"
    ),
    "RAW FADED": re.compile(
        r"\[AUTO\] RAW FADED:\s+(\S+)"
    ),
    "ENTRY SUPPRESSED": re.compile(
        r"\[AUTO\] ENTRY SUPPRESSED:\s+(\S+)\s+(YES|NO)\s+"
        r"conf=(\d+)%\s+edge=([+-]?\d+\.\d+)%"
    ),
}


def _ts(s: str) -> datetime:
    """Parse bot.log timestamps. bot.log uses local EDT (UTC-4); convert to UTC.
    (trades.csv timestamps are already UTC with +00:00 suffix; this function
    is only for bot.log lines, which lack tz info and are local-time.)"""
    return (datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            .replace(tzinfo=timezone.utc)
            + timedelta(hours=4))


def parse_close_utc(ticker: str) -> datetime | None:
    """Decode close time from 15-min BTC ticker. Ticker time is EDT — add 4h."""
    try:
        suffix = ticker.split("-")[1]
        yy = 2000 + int(suffix[:2])
        mon = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
               "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}[suffix[2:5]]
        dd, hh, mm = int(suffix[5:7]), int(suffix[7:9]), int(suffix[9:11])
        return (datetime(yy, mon, dd, hh, mm, tzinfo=timezone.utc)
                + timedelta(hours=4))
    except Exception:
        return None


def phase_of(secs_left: float) -> str:
    for name, lo, hi in PHASE_BOUNDARIES:
        if lo <= secs_left < hi:
            return name
    return "unknown"


# ─── Event extraction ─────────────────────────────────────────────────────────

def parse_log(path: Path, quiet: bool = False) -> dict:
    """Extract structured event records from bot.log.

    Returns:
      {
        "fires":     list of fire records
        "pendings":  list of pending records (incl. those that never fired)
        "rejects":   list of explicit rejection events by filter
      }
    """
    fires, pendings, rejects = [], [], []
    if not path.exists():
        print(f"  ! log not found: {path}")
        return {"fires": [], "pendings": [], "rejects": []}

    n_lines = 0
    with open(path, errors="replace") as f:
        for line in f:
            n_lines += 1
            # Try fire first (most specific)
            m = FIRE_RE.search(line)
            if m:
                ts_s, phase_label, mode, tk, side, conf, edge, raw_c, raw_e, mid, fyv, fnv, fnet = m.groups()
                fires.append({
                    "ts": _ts(ts_s), "phase_label": phase_label, "mode": mode,
                    "ticker": tk, "side": side.lower(),
                    "conf": int(conf) / 100.0,
                    "edge": float(edge) / 100.0,
                    "raw_conf": (int(raw_c) / 100.0) if raw_c else None,
                    "raw_edge": (float(raw_e) / 100.0) if raw_e else None,
                    "mid_cents": float(mid),
                    "flow_yes_vol": float(fyv) if fyv else None,
                    "flow_no_vol":  float(fnv) if fnv else None,
                    "flow_net":     float(fnet) if fnet else None,
                })
                continue
            m = PEND_RE.search(line)
            if m:
                ts_s, k, K, tk, side, conf, edge = m.groups()
                pendings.append({
                    "ts": _ts(ts_s), "k": int(k), "K": int(K),
                    "ticker": tk, "side": side.lower(),
                    "conf": int(conf) / 100.0,
                    "edge": float(edge) / 100.0,
                })
                continue
            # Rejection patterns
            for reason, regex in REJECT_RES.items():
                m = regex.search(line)
                if m:
                    groups = m.groups()
                    # All reject lines start with timestamp at start of line
                    ts_m = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                    if not ts_m:
                        break
                    rec = {
                        "ts": _ts(ts_m.group(1)),
                        "reason": reason,
                        "ticker": groups[0],
                        "side": groups[1].lower() if len(groups) > 1 and groups[1] in ("YES","NO") else None,
                    }
                    if reason == "ENTRY SUPPRESSED" and len(groups) >= 4:
                        rec["conf"] = int(groups[2]) / 100.0
                        rec["edge"] = float(groups[3]) / 100.0
                    rejects.append(rec)
                    break

    if not quiet:
        print(f"  parsed {n_lines:,} log lines")
        print(f"  fires:     {len(fires):,}")
        print(f"  pendings:  {len(pendings):,}")
        print(f"  rejects:   {len(rejects):,}")
        for reason in REJECT_RES:
            n = sum(1 for r in rejects if r["reason"] == reason)
            print(f"    {reason:20}: {n:,}")
    return {"fires": fires, "pendings": pendings, "rejects": rejects}


def load_outcomes() -> dict[str, dict]:
    """Map ticker → outcome from cross_validate.json. Aggregates across sessions."""
    if not CV_JSON.exists():
        return {}
    cv = json.loads(CV_JSON.read_text())
    out = {}
    for sess in cv.values():
        for p in sess.get("positions", []):
            # If ticker has multiple positions (e.g., reversal), use the first one's outcome
            # We're going to also key by (ticker, side) for finer matching below
            out.setdefault(p["ticker"], p)
            out[(p["ticker"], p.get("side"))] = p
    return out


def settlement_from_tape(ticker: str) -> str | None:
    """Look up final settlement (yes/no) from cached tape. Last trades cluster
    at $0.99 or $0.01 depending on side."""
    tape_path = TAPES_DIR / f"{ticker}.json"
    if not tape_path.exists():
        return None
    try:
        data = json.loads(tape_path.read_text())
        trades = data.get("trades", [])
        if not trades:
            return None
        # Look at last few trades
        last_yes_prices = [
            float(t["yes_price_dollars"]) for t in trades[-10:]
            if "yes_price_dollars" in t
        ]
        if not last_yes_prices:
            return None
        median = sorted(last_yes_prices)[len(last_yes_prices) // 2]
        if median >= 0.95:
            return "yes"
        elif median <= 0.05:
            return "no"
        return None
    except Exception:
        return None


# ─── Derived computations per fire ────────────────────────────────────────────

def enrich_fire(fire: dict, outcomes: dict) -> dict:
    """Compute prob_yes, secs_remaining, phase, outcome class for a fire."""
    side = fire["side"]
    mid_yes = fire["mid_cents"] / 100.0
    # edge_yes = prob_yes - mid_yes  →  prob_yes = mid_yes + edge_yes
    # edge_no  = prob_no  - mid_no   →  prob_no  = (1 - mid_yes) + edge_no
    # We can only derive the OUR-SIDE prob, then convert to prob_yes.
    if side == "yes":
        our_prob = mid_yes + fire["edge"]
        prob_yes = max(0.0, min(1.0, our_prob))
    else:
        our_prob = (1 - mid_yes) + fire["edge"]
        prob_no = max(0.0, min(1.0, our_prob))
        prob_yes = 1.0 - prob_no
    fire["prob_yes"] = prob_yes
    fire["our_prob"] = our_prob  # P(our side settles)

    # Phase based on time-to-close
    close_utc = parse_close_utc(fire["ticker"])
    if close_utc:
        secs_left = (close_utc - fire["ts"]).total_seconds()
        fire["secs_left"] = secs_left
        fire["phase"] = phase_of(secs_left)
    else:
        fire["secs_left"] = None
        fire["phase"] = "unknown"

    # Cross_validate outcome (try both ticker+side then ticker-only)
    o = outcomes.get((fire["ticker"], side)) or outcomes.get(fire["ticker"])
    if o:
        fire["cv_class"] = o.get("class")
        fire["cv_result"] = o.get("result")
        fire["cv_pnl"] = o.get("pnl")
    else:
        fire["cv_class"] = None
        fire["cv_result"] = None
        fire["cv_pnl"] = None

    # Tape-derived settlement as a fallback / cross-check
    tape_settle = settlement_from_tape(fire["ticker"])
    fire["tape_settle"] = tape_settle

    # Authoritative "did our side win"
    final_settle = fire["cv_result"] or fire["tape_settle"]
    if final_settle in ("yes", "no"):
        fire["our_side_won"] = (final_settle == side)
    else:
        fire["our_side_won"] = None
    return fire


def build_rejected_attempts(events: dict) -> list[dict]:
    """For each rejection, reconstruct the attempted-entry context and look up
    the market's eventual settlement. Returns records like:
      {
        ticker, side, reason, ts, secs_left, phase, conf, edge,
        settle (yes/no/None), would_have_won (True/False/None),
      }
    """
    rejects = events["rejects"]
    # Index pendings + fires by (ticker, side) for time-nearest lookup
    pend_by_key = defaultdict(list)
    for p in events["pendings"]:
        pend_by_key[(p["ticker"], p["side"])].append(p)
    for k in pend_by_key:
        pend_by_key[k].sort(key=lambda x: x["ts"])

    results = []
    for r in rejects:
        tk, side = r["ticker"], r["side"]
        if not side:
            # RAW FADED log doesn't include side; try to recover from nearest pending
            cands = []
            for s in ("yes", "no"):
                for p in pend_by_key.get((tk, s), []):
                    delta = abs((p["ts"] - r["ts"]).total_seconds())
                    if delta < 10:
                        cands.append((delta, s, p))
            if not cands:
                continue
            cands.sort()
            _, side, nearest_pend = cands[0]
        else:
            # Find nearest preceding pending for context
            cands = [p for p in pend_by_key.get((tk, side), [])
                     if (r["ts"] - p["ts"]).total_seconds() < 10
                     and (r["ts"] - p["ts"]).total_seconds() >= 0]
            nearest_pend = cands[-1] if cands else None

        conf = r.get("conf") or (nearest_pend and nearest_pend["conf"]) or None
        edge = r.get("edge") or (nearest_pend and nearest_pend["edge"]) or None

        close_utc = parse_close_utc(tk)
        secs_left = (close_utc - r["ts"]).total_seconds() if close_utc else None
        phase = phase_of(secs_left) if secs_left is not None else "unknown"

        settle = settlement_from_tape(tk)
        would_have_won = (settle == side) if settle in ("yes", "no") else None

        results.append({
            "ts": r["ts"], "reason": r["reason"], "ticker": tk, "side": side,
            "secs_left": secs_left, "phase": phase,
            "conf": conf, "edge": edge,
            "settle": settle, "would_have_won": would_have_won,
        })
    return results


# ─── ANALYSIS 1: calibration ──────────────────────────────────────────────────

def calibration_analysis(fires: list[dict]) -> None:
    print("\n" + "=" * 86)
    print(" ANALYSIS 1: ENSEMBLE CALIBRATION")
    print(" Does P(YES from model) match observed settle rate?")
    print("=" * 86)
    scored = [f for f in fires if f["our_side_won"] is not None]
    if not scored:
        print("  ! no fires with known settlement — skipping")
        return
    print(f"  n closed fires with known settlement: {len(scored)}\n")

    # Bucket by prob_yes in 5pp buckets
    buckets = [(0.30, 0.40), (0.40, 0.50), (0.50, 0.55), (0.55, 0.60),
               (0.60, 0.65), (0.65, 0.70), (0.70, 0.80), (0.80, 1.01)]
    print(f"  {'bucket':16} {'n':>4} {'pred':>6} {'observed':>10} {'gap':>6}")
    print("  " + "-" * 60)
    total_brier = 0.0
    for lo, hi in buckets:
        in_b = [f for f in scored if lo <= f["prob_yes"] < hi]
        if not in_b:
            continue
        avg_pred = sum(f["prob_yes"] for f in in_b) / len(in_b)
        # observed = fraction settling YES (regardless of which side we picked)
        obs_yes = sum(1 for f in in_b
                      if (f["cv_result"] or f["tape_settle"]) == "yes") / len(in_b)
        gap = obs_yes - avg_pred
        marker = "✓" if abs(gap) < 0.05 else ("⚠" if abs(gap) < 0.10 else "✗")
        print(f"  {f'{lo:.2f}-{hi:.2f}':16} {len(in_b):>4} {avg_pred:>6.2f} "
              f"{obs_yes:>9.2%}  {gap:>+5.2f} {marker}")

    # Brier score: mean squared error between prob_yes and (1 if YES else 0)
    brier_terms = []
    for f in scored:
        actual = 1.0 if (f["cv_result"] or f["tape_settle"]) == "yes" else 0.0
        brier_terms.append((f["prob_yes"] - actual) ** 2)
    brier = sum(brier_terms) / len(brier_terms)
    # Reference: a constant 50% predictor has Brier = 0.25
    print(f"\n  Brier score: {brier:.4f}  (vs 0.25 = constant 50% predictor)")
    if brier < 0.22:
        print(f"  → model has meaningful predictive power")
    elif brier < 0.25:
        print(f"  → model marginally better than 50% baseline")
    else:
        print(f"  → model performing WORSE than constant 50%  — strong miscalibration signal")

    # OUR-SIDE perspective: when we predict our side wins with prob X, do we win X%?
    print(f"\n  Our-side calibration: when we said our side has P=p, did we win?")
    print(f"  {'bucket':16} {'n':>4} {'pred (our)':>10} {'observed':>10} {'gap':>6}")
    print("  " + "-" * 60)
    our_brier_terms = []
    for lo, hi in [(0.50,0.55),(0.55,0.60),(0.60,0.65),(0.65,0.70),(0.70,0.80),(0.80,1.01)]:
        in_b = [f for f in scored if lo <= f["our_prob"] < hi]
        if not in_b:
            continue
        avg_pred = sum(f["our_prob"] for f in in_b) / len(in_b)
        obs_win = sum(1 for f in in_b if f["our_side_won"]) / len(in_b)
        gap = obs_win - avg_pred
        marker = "✓" if abs(gap) < 0.05 else ("⚠" if abs(gap) < 0.10 else "✗")
        print(f"  {f'{lo:.2f}-{hi:.2f}':16} {len(in_b):>4} {avg_pred:>10.2f} "
              f"{obs_win:>9.2%}  {gap:>+5.2f} {marker}")
    for f in scored:
        actual = 1.0 if f["our_side_won"] else 0.0
        our_brier_terms.append((f["our_prob"] - actual) ** 2)
    our_brier = sum(our_brier_terms) / max(len(our_brier_terms), 1)
    print(f"\n  Our-side Brier: {our_brier:.4f}")


# ─── ANALYSIS 2: filter-stack contribution ────────────────────────────────────

def filter_contribution_analysis(rejected: list[dict]) -> None:
    print("\n" + "=" * 86)
    print(" ANALYSIS 2: FILTER-STACK CONTRIBUTION")
    print(" For each rejection reason: are we rejecting more wins or losses?")
    print(" Hypothetical: 'if we HAD entered, would our side have won?'")
    print("=" * 86)

    by_reason = defaultdict(list)
    for r in rejected:
        by_reason[r["reason"]].append(r)

    print(f"\n  {'reason':22} {'n':>5} {'w/ settle':>10} {'would_win':>10} {'win%':>6} {'verdict':>12}")
    print("  " + "-" * 80)
    for reason in REJECT_RES:
        rs = by_reason.get(reason, [])
        scored = [r for r in rs if r["would_have_won"] is not None]
        if not rs:
            continue
        wins = sum(1 for r in scored if r["would_have_won"])
        rate = wins / max(len(scored), 1) * 100
        if not scored:
            verdict = "no data"
        elif rate > 55:
            verdict = "HURTING"  # rejected good trades
        elif rate < 45:
            verdict = "helping"   # rejected losing trades
        else:
            verdict = "neutral"
        print(f"  {reason:22} {len(rs):>5} {len(scored):>10} "
              f"{wins:>10} {rate:>5.1f}%  {verdict:>12}")

    # Show context — what would the cost have been per rejection?
    print(f"\n  Per-rejection net-PnL (assuming entry @ recorded conditions):")
    print(f"  {'reason':22} {'n':>5} {'sum w-l':>9}   notes")
    print("  " + "-" * 80)
    for reason in REJECT_RES:
        rs = [r for r in by_reason.get(reason, []) if r["would_have_won"] is not None]
        if not rs:
            continue
        wins_minus_losses = sum(1 if r["would_have_won"] else -1 for r in rs)
        print(f"  {reason:22} {len(rs):>5} {wins_minus_losses:>+9}   "
              f"(positive = filter saving us; negative = filter costing us)")


# ─── ANALYSIS 3: per-condition slicing ────────────────────────────────────────

def slicing_analysis(fires: list[dict]) -> None:
    print("\n" + "=" * 86)
    print(" ANALYSIS 3: PER-CONDITION SLICING")
    print(" Where is the model reliable vs noisy?")
    print("=" * 86)

    scored = [f for f in fires if f["our_side_won"] is not None]
    if not scored:
        return

    def slice_by(key_fn, label, buckets, fmt):
        print(f"\n  {label}:")
        print(f"  {'bucket':14} {'n':>4} {'win%':>6} {'avg prob':>10} {'gap':>6}")
        for b_label, b_pred in buckets:
            in_b = [f for f in scored if key_fn(f) == b_label]
            if not in_b:
                continue
            win = sum(1 for f in in_b if f["our_side_won"])
            win_pct = win / len(in_b) * 100
            avg_prob = sum(f["our_prob"] for f in in_b) / len(in_b)
            gap = win_pct / 100 - avg_prob
            print(f"  {fmt(b_label):14} {len(in_b):>4} {win_pct:>5.1f}% "
                  f"{avg_prob:>10.2%} {gap:>+5.2f}")

    slice_by(
        lambda f: f["phase"], "By phase",
        [("early", None), ("mid", None), ("prime", None), ("late", None)],
        str,
    )

    def conf_bucket(f):
        c = f["conf"]
        if c < 0.55: return "<55"
        if c < 0.65: return "55-64"
        if c < 0.75: return "65-74"
        return "75+"
    slice_by(
        conf_bucket, "By confidence bucket",
        [("<55", None), ("55-64", None), ("65-74", None), ("75+", None)],
        lambda x: f"conf {x}%",
    )

    def edge_bucket(f):
        e = f["edge"]
        if e < 0.10: return "<10"
        if e < 0.15: return "10-15"
        if e < 0.20: return "15-20"
        if e < 0.30: return "20-30"
        return "30+"
    slice_by(
        edge_bucket, "By edge bucket",
        [("<10", None), ("10-15", None), ("15-20", None), ("20-30", None), ("30+", None)],
        lambda x: f"edge {x}%",
    )

    def flow_bucket(f):
        if f["flow_net"] is None:
            return "unknown"
        # Convert net to our-side-relative
        signed = f["flow_net"] if f["side"] == "yes" else -f["flow_net"]
        if signed >= 0.3: return "strong-with"
        if signed >= 0.0: return "neutral-with"
        if signed >= -0.3: return "neutral-against"
        return "strong-against"
    slice_by(
        flow_bucket, "By tape-flow alignment",
        [("strong-with", None), ("neutral-with", None),
         ("neutral-against", None), ("strong-against", None), ("unknown", None)],
        str,
    )


# ─── ANALYSIS 4: edge-on-the-table (tape-derived) ─────────────────────────────

def edge_on_table_analysis(fires: list[dict]) -> None:
    print("\n" + "=" * 86)
    print(" ANALYSIS 4: EDGE-ON-THE-TABLE (TAPE GROUND TRUTH)")
    print(" For each fire: how much of the available edge did we capture?")
    print("=" * 86)
    scored = [f for f in fires if f["cv_pnl"] is not None]
    if not scored:
        print("  ! no fires with realized PnL — skipping")
        return

    # For each fire with realized PnL, compute (1) actual realized vs (2)
    # hypothetical full-settlement PnL had we held.
    realized = sum(f["cv_pnl"] for f in scored)
    # Hypothetical hold-to-settle: cost is entry_price × contracts; settle value
    # is contracts × $1 (if right side wins) or $0 (if wrong). We approximate
    # via cv_class — CW/SO both right-side, WL wrong.
    held_total = 0.0
    for f in scored:
        # cv_pnl already encodes realized. We don't have exact contracts here
        # without parsing trades.csv; approximate via the side_correct +
        # whether held would have paid full vs partial.
        pass  # placeholder — exact calc requires contract counts

    print(f"  Total fires with PnL: {len(scored)}")
    print(f"  Realized PnL across all fires: ${realized:+.2f}")
    print(f"  Avg per fire: ${realized/len(scored):+.2f}")
    print()
    print(f"  Outcome class breakdown:")
    classes = defaultdict(list)
    for f in scored:
        classes[f["cv_class"]].append(f["cv_pnl"])
    for cls in ("correct_win", "shaken_out", "saved_by_exit", "wrong_loss"):
        if cls not in classes: continue
        ps = classes[cls]
        print(f"    {cls:16} n={len(ps):>4} total=${sum(ps):+7.2f} avg=${sum(ps)/len(ps):+5.2f}")

    print(f"\n  Held-to-settle hypothetical (per cross_validate's standard analysis):")
    so = classes.get("shaken_out", [])
    if so:
        # Each SO would have settled in our favor → contracts × ($1 - entry)
        # We don't have entry price here; just count
        print(f"    {len(so)} SO trades realized ${sum(so):+.2f}")
        print(f"    These would have settled in our favor if held to expiry.")
        print(f"    The 'edge on the table' is the SO leakage — exit-side problem.")


# ─── ANALYSIS 5: recommendation engine ────────────────────────────────────────

def recommendation_engine(fires: list[dict], rejected: list[dict]) -> None:
    print("\n" + "=" * 86)
    print(" ANALYSIS 5: RECOMMENDATION ENGINE")
    print(" Based on the data above, what should we drop/keep/tighten/loosen?")
    print("=" * 86)

    # 1. Filter recommendations
    print(f"\n  Filter actions ranked by impact:")
    print(f"  {'filter':22} {'rec':>10} {'evidence':>40}")
    print("  " + "-" * 80)
    by_reason = defaultdict(list)
    for r in rejected:
        if r["would_have_won"] is not None:
            by_reason[r["reason"]].append(r)
    for reason in REJECT_RES:
        rs = by_reason.get(reason, [])
        if not rs:
            print(f"  {reason:22} {'no-data':>10} {'(no settled rejected entries)':>40}")
            continue
        win_rate = sum(1 for r in rs if r["would_have_won"]) / len(rs)
        if win_rate > 0.55:
            rec = "DROP"
            ev = f"rejected {win_rate*100:.0f}% wins from {len(rs)} cases"
        elif win_rate < 0.40:
            rec = "KEEP"
            ev = f"rejected {(1-win_rate)*100:.0f}% losses from {len(rs)} cases"
        else:
            rec = "loosen?"
            ev = f"near-coinflip ({win_rate*100:.0f}% wins) from {len(rs)} cases"
        print(f"  {reason:22} {rec:>10} {ev:>40}")

    # 2. Calibration verdict
    print(f"\n  Calibration verdict:")
    scored = [f for f in fires if f["our_side_won"] is not None]
    if scored:
        brier_terms = []
        for f in scored:
            actual = 1.0 if (f["cv_result"] or f["tape_settle"]) == "yes" else 0.0
            brier_terms.append((f["prob_yes"] - actual) ** 2)
        brier = sum(brier_terms) / len(brier_terms)
        if brier < 0.22:
            print(f"    Brier {brier:.4f} → CALIBRATED enough to act on. "
                  f"Strip gates that are hurting; trust the signal.")
        elif brier < 0.25:
            print(f"    Brier {brier:.4f} → MARGINAL calibration. "
                  f"Per-component instrumentation needed to identify which model carries alpha.")
        else:
            print(f"    Brier {brier:.4f} → POOR calibration. "
                  f"Recommend recalibration layer (isotonic regression / Platt scaling) "
                  f"OR pivot strategy class (arbitrage / maker-only).")

    # 3. Per-condition signal strength
    print(f"\n  Where to focus / what to filter on:")
    if scored:
        by_phase = defaultdict(list)
        for f in scored:
            by_phase[f["phase"]].append(f)
        for ph, fs in sorted(by_phase.items()):
            if not fs: continue
            wr = sum(1 for f in fs if f["our_side_won"]) / len(fs)
            if wr > 0.60:
                print(f"    {ph:8}: side-win {wr*100:.0f}% (n={len(fs)}) → STRONG; "
                      f"loosen gates in this phase")
            elif wr < 0.45:
                print(f"    {ph:8}: side-win {wr*100:.0f}% (n={len(fs)}) → "
                      f"WEAK; tighten or skip this phase")
            else:
                print(f"    {ph:8}: side-win {wr*100:.0f}% (n={len(fs)}) → coinflip")

    print(f"\n  Per-component analysis would require new instrumentation. "
          f"If overall calibration is OK, defer. If overall calibration is poor, "
          f"per-component data may reveal one good model masked by 4 bad ones.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Entry-strategy audit backtest")
    ap.add_argument("--log", default=str(BOT_LOG), help="Path to bot.log")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    ap.add_argument("--section", choices=["calibration","filters","slicing",
                                          "edge","recommendations","all"],
                    default="all", help="Run only one analysis section")
    args = ap.parse_args()

    if not args.quiet:
        print("Parsing bot.log...")
    events = parse_log(Path(args.log), quiet=args.quiet)
    if not events["fires"]:
        print("\n! no SIGNAL fires found — cannot audit. Check log path.")
        return

    if not args.quiet:
        print("\nLoading outcomes from cross_validate.json...")
    outcomes = load_outcomes()
    if not args.quiet:
        print(f"  outcomes loaded for {len(outcomes)} (ticker, side) pairs")

    if not args.quiet:
        print("\nEnriching fires with derived fields...")
    for f in events["fires"]:
        enrich_fire(f, outcomes)

    if not args.quiet:
        print("Building rejected-entry hypothetical outcomes...")
    rejected = build_rejected_attempts(events)
    if not args.quiet:
        print(f"  {len(rejected)} rejection events with reconstructed context")
        scored_rej = sum(1 for r in rejected if r["would_have_won"] is not None)
        print(f"  {scored_rej} have tape-derived settlements available")

    sec = args.section
    if sec in ("calibration", "all"):  calibration_analysis(events["fires"])
    if sec in ("filters", "all"):      filter_contribution_analysis(rejected)
    if sec in ("slicing", "all"):      slicing_analysis(events["fires"])
    if sec in ("edge", "all"):         edge_on_table_analysis(events["fires"])
    if sec in ("recommendations", "all"): recommendation_engine(events["fires"], rejected)

    print("\n" + "=" * 86)
    print(" Done. Read the sections above before making any architectural decisions.")
    print("=" * 86)


if __name__ == "__main__":
    main()
