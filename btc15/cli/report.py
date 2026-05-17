"""Houston — command-center report for 15minbtc.

Snapshot-mode panoramic view across all sessions in logs/trades.csv,
optionally filtered by mode (paper/live) and start date.

Sections rendered (top to bottom):
  1.  Mission status         — bot uptime, current session, filters in effect
  2.  Headline P&L           — realized P&L, ROI, win rate, theoretical edge
  3.  Outcome classes        — CW / SO / SE / WL with class P&L contribution
  4.  Per-session table      — all kept sessions, sortable by date
  5.  Exit reason mix        — reversal / loss_cut / profit_take / settled / emergency
  6.  Entry source mix       — dir_early / dir_prime / arb / reconciled / mm_quote
  7.  Pyramid analysis       — by leg count, shows pyramid rope (Fix #3) effect
  8.  Cool-off effectiveness — LOSS CUT PENDING armed vs saved vs fired (Fix #1)
  9.  Reconciler activity    — placement-race guard skips, gap adoptions (commit 6801952)
  10. Notable trades         — top 5 wins and top 5 losses with full context
  11. System health          — bot.log errors and reconnects

Kalshi settlement results are cached in data/market_results_cache.json. Settled
markets are immutable, so only unsettled / unknown tickers are re-fetched on
subsequent runs. Use --no-fetch to skip the Kalshi roundtrip entirely.

All timestamps displayed in UTC. Color discipline matches the live dashboard.
"""
from __future__ import annotations

import csv
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns

# ── Paths and constants ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = ROOT / "logs"
DATA_DIR = ROOT / "data"
TRADES_CSV = LOGS_DIR / "trades.csv"
BOT_LOG = LOGS_DIR / "bot.log"
CACHE_FILE = DATA_DIR / "market_results_cache.json"

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
REQ_DELAY = 0.2   # 5 req/s — well under Kalshi's public-endpoint headroom

# bot.log uses Pi local time (EDT, UTC-4). Add 4h to get UTC.
BOTLOG_TO_UTC = timedelta(hours=4)

console = Console()


# ── Time parsing ─────────────────────────────────────────────────────────────
def parse_iso(s: str) -> datetime:
    """Parse ISO timestamps tolerantly (5-digit fractional seconds happen)."""
    s = s.replace("Z", "+00:00")
    m = re.match(r"^(.*\.)(\d{1,5})([+-]\d{2}:\d{2})$", s)
    if m:
        s = m.group(1) + m.group(2).ljust(6, "0") + m.group(3)
    return datetime.fromisoformat(s)


def parse_botlog_ts(s: str) -> datetime:
    """bot.log emits 'YYYY-MM-DD HH:MM:SS,mmm' in Pi local time (EDT).
    Convert to UTC by adding 4 hours."""
    t = datetime.strptime(s.split(",")[0], "%Y-%m-%d %H:%M:%S")
    return (t + BOTLOG_TO_UTC).replace(tzinfo=timezone.utc)


def fmt_dt(d: Optional[datetime]) -> str:
    if d is None:
        return "—"
    return d.strftime("%Y-%m-%d %H:%M UTC")


def fmt_dur_min(m: float) -> str:
    if m < 60:
        return f"{m:.0f}m"
    if m < 60 * 24:
        return f"{m/60:.1f}h"
    return f"{m/1440:.1f}d"


def tag_sort_key(tag: str):
    """Session tags are 'DDMMMHH:MM' in UTC (e.g. '15MAY01:14')."""
    day = int(tag[:2])
    mon = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
           "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}[tag[2:5]]
    hh, mm = tag[5:].split(":")
    return (mon, day, int(hh), int(mm))


# ── CSV parsing ──────────────────────────────────────────────────────────────
def fam(s: str) -> str:
    return "yes" if s.startswith("yes") else "no"


def is_exit_side(s: str) -> bool:
    return s.endswith("_exit") or s.endswith("_settled")


def exit_bucket(source: str) -> str:
    s = source.lower()
    if "emergency" in s:                          return "emergency_stop"
    if "after" in s and "cool-off" in s:          return "loss_cut_after_cooloff"
    if "loss_cut" in s:                           return "loss_cut"
    if "profit_take" in s:                        return "profit_take"
    if "reversal" in s:                           return "reversal"
    if "settled" in s or "settlement" in s:       return "settled"
    if "time" in s or "expir" in s:               return "time_stop"
    return "other"


def entry_bucket(source: str) -> str:
    s = source.lower()
    if "dir_early" in s:        return "dir_early"
    if "dir_prime" in s:        return "dir_prime"
    if "dir_late" in s:         return "dir_late"
    if "pure_arb" in s or "arb" in s: return "arb"
    if "reconciled_gap" in s:   return "reconciled_gap"
    if "reconciled" in s:       return "reconciled"
    if "gtc_escalated" in s:    return "gtc_escalated"
    if "mm_quote" in s:         return "mm_quote"
    if "settlement_lock" in s:  return "settlement_lock"
    if "snipe" in s:            return "sniper"
    if "scalp" in s:            return "scalper"
    if "manual" in s:           return "manual"
    return "other"


@dataclass
class Row:
    trade_id: str
    ts: datetime
    ticker: str
    side: str
    contracts: int
    price_cents: int
    cost_usd: float
    source: str
    mode: str
    session: str


def load_rows(trades_path: Path, mode_filter: Optional[str],
              since: Optional[datetime]) -> list[Row]:
    """Read trades.csv, filter by mode and start date. Only 10-col schema."""
    rows: list[Row] = []
    if not trades_path.exists():
        return rows
    with open(trades_path) as f:
        for r in csv.reader(f):
            if len(r) != 10 or r[0] == "trade_id":
                continue
            if mode_filter is not None and r[8] != mode_filter:
                continue
            try:
                ts = parse_iso(r[1])
            except Exception:
                continue
            if since is not None and ts < since:
                continue
            try:
                rows.append(Row(
                    trade_id=r[0], ts=ts, ticker=r[2], side=r[3],
                    contracts=int(r[4]), price_cents=int(r[5]),
                    cost_usd=float(r[6]), source=r[7], mode=r[8],
                    session=r[9],
                ))
            except (ValueError, TypeError):
                continue
    return rows


# ── Position aggregation (groups by (session, ticker, side_family)) ──────────
@dataclass
class Position:
    session: str
    ticker: str
    side: str               # "yes" / "no"
    qty: int
    n_legs: int             # 1=single, 2=pyramid+1, etc.
    entry_avg: float        # avg entry price in cents
    exit_avg: Optional[float]
    pnl: Optional[float]
    ts_open: datetime
    ts_exit: Optional[datetime]
    entry_src: str
    exit_src: str
    entry_bucket: str
    exit_bucket: str
    result: Optional[str] = None       # "yes" / "no" / None
    correct: Optional[bool] = None
    klass: str = "open"                # CW / SO / SE / WL / open / unknown


def build_positions(rows: list[Row]) -> list[Position]:
    legs: dict = defaultdict(lambda: {"opens": [], "closes": []})
    for r in sorted(rows, key=lambda r: r.ts):
        key = (r.session, r.ticker, fam(r.side))
        (legs[key]["closes"] if is_exit_side(r.side) else
         legs[key]["opens"]).append(r)

    positions: list[Position] = []
    for (sess, ticker, side), lg in legs.items():
        if not lg["opens"]:
            continue
        open_cost = sum(o.cost_usd for o in lg["opens"])
        open_qty = sum(o.contracts for o in lg["opens"])
        close_cost = sum(c.cost_usd for c in lg["closes"])
        close_qty = sum(c.contracts for c in lg["closes"])
        ts_open = min(o.ts for o in lg["opens"])
        ts_exit = max((c.ts for c in lg["closes"]), default=None)
        exit_src = lg["closes"][-1].source if lg["closes"] else "OPEN"
        entry_src = lg["opens"][0].source
        fully_closed = bool(lg["closes"]) and close_qty >= open_qty
        pnl = (close_cost - open_cost) if fully_closed else None

        positions.append(Position(
            session=sess, ticker=ticker, side=side, qty=open_qty,
            n_legs=len(lg["opens"]),
            entry_avg=(open_cost / open_qty * 100) if open_qty else 0,
            exit_avg=(close_cost / close_qty * 100) if close_qty else None,
            pnl=pnl, ts_open=ts_open, ts_exit=ts_exit,
            entry_src=entry_src, exit_src=exit_src,
            entry_bucket=entry_bucket(entry_src),
            exit_bucket=exit_bucket(exit_src) if lg["closes"] else "OPEN",
        ))
    return positions


# ── Kalshi market-result cache ───────────────────────────────────────────────
def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2, default=str))


def fetch_results(tickers: list[str], cache: dict) -> dict:
    """Fetch market.result for any ticker missing from cache or marked unsettled.
    Settled (result in {yes,no}) entries are immutable — never re-fetched.
    Returns mutated cache."""
    def is_done(entry):
        return isinstance(entry, dict) and entry.get("result") in ("yes", "no")

    todo = [t for t in tickers if not is_done(cache.get(t))]
    if not todo:
        return cache

    # Deferred import so `--no-fetch` works even when httpx isn't installed
    # (e.g., when running the report from a Python env without project deps).
    import httpx
    client = httpx.Client(timeout=10.0, headers={"accept": "application/json"})
    t0 = time.time()
    try:
        for i, ticker in enumerate(todo, 1):
            try:
                r = client.get(f"{KALSHI_BASE}/markets/{ticker}")
                if r.status_code == 200:
                    mk = r.json().get("market") or {}
                    cache[ticker] = {
                        "result": mk.get("result"),
                        "status": mk.get("status"),
                        "close_time": mk.get("close_time"),
                        "fetched_at": datetime.utcnow().isoformat(),
                    }
                else:
                    cache[ticker] = {"result": None, "http_status": r.status_code}
            except Exception as e:
                cache[ticker] = {"result": None, "error": str(e)[:120]}
            if i % 50 == 0 or i == len(todo):
                save_cache(cache)
                console.print(f"  [dim]Kalshi pre-fetch: {i}/{len(todo)} "
                              f"({i / (time.time() - t0):.1f} req/s)[/dim]")
            time.sleep(REQ_DELAY)
    finally:
        client.close()
        save_cache(cache)
    return cache


def classify_positions(positions: list[Position], cache: dict) -> None:
    """In-place: set result/correct/klass for each position."""
    for p in positions:
        mr = cache.get(p.ticker) or {}
        result = mr.get("result")
        p.result = result
        if p.pnl is None:
            p.klass = "open"
            continue
        if result not in ("yes", "no"):
            p.klass = "unknown"
            continue
        p.correct = (p.side == result)
        if p.correct and p.pnl > 0:        p.klass = "correct_win"
        elif p.correct:                    p.klass = "shaken_out"
        elif p.pnl > 0:                    p.klass = "saved_by_exit"
        else:                              p.klass = "wrong_loss"


# ── bot.log scanning ─────────────────────────────────────────────────────────
@dataclass
class BotLogStats:
    suppressed: int = 0
    suppressed_legit: int = 0      # rec == side AND conf>=40 (post-tweak: all should be legit)
    suppressed_edge_floor_rec_side: int = 0  # rec==side but conf<40 (edge-floor branch)
    suppressed_anomaly: int = 0    # rec=none — should be ZERO post d234614 tweak
    cooloff_armed: int = 0         # LOSS CUT PENDING
    cooloff_fired: int = 0         # LOSS CUT (after Xs cool-off)
    cooloff_saved: int = 0         # armed - fired
    loss_cut_immediate: int = 0    # LOSS CUT: (no PENDING / no "after")
    emergency_stop: int = 0
    reversal_exit: int = 0
    profit_take: int = 0
    pyramid_eligible: int = 0
    entry_suppressed: int = 0
    reconcile_skip_placement: int = 0   # post-6801952
    reconcile_skip_exit_lag: int = 0
    reconcile_skip_settled: int = 0
    reconcile_adopt_gap: int = 0
    ws_reconnects: int = 0
    scan_loop_errors: int = 0
    zerodiv_errors: int = 0
    stale_cache_warnings: int = 0
    cooloff_save_examples: list = field(default_factory=list)


def scan_botlog(since: Optional[datetime]) -> BotLogStats:
    s = BotLogStats()
    if not BOT_LOG.exists():
        return s
    pat_sup_rec = re.compile(
        r"STOP SUPPRESSED:.*\|.*rec=(\S+) conf=(\d+)%"
    )
    armed_keys: set = set()      # (ticker, ts) keys that armed
    # Naive but fine: count pending and after-cooloff lines
    with open(BOT_LOG, errors="ignore") as f:
        for line in f:
            ts_str = line[:23]    # "YYYY-MM-DD HH:MM:SS,mmm"
            if since is not None:
                try:
                    line_ts = parse_botlog_ts(ts_str)
                    if line_ts < since:
                        continue
                except Exception:
                    pass
            if "STOP SUPPRESSED" in line:
                s.suppressed += 1
                m = pat_sup_rec.search(line)
                if m:
                    rec = m.group(1)
                    conf = int(m.group(2))
                    # Figure out which side this is suppressing for: parse "SUPPRESSED: TICKER SIDE"
                    side_m = re.search(r"SUPPRESSED: \S+ ([A-Z]+) \|", line)
                    side = side_m.group(1).lower() if side_m else ""
                    if rec == side and conf >= 40:
                        s.suppressed_legit += 1
                    elif rec == side and conf < 40:
                        s.suppressed_edge_floor_rec_side += 1
                    elif rec == "none":
                        s.suppressed_anomaly += 1
            elif "LOSS CUT PENDING" in line:
                s.cooloff_armed += 1
            elif "LOSS CUT (after" in line:
                s.cooloff_fired += 1
            elif "LOSS CUT:" in line and "PENDING" not in line and "after" not in line:
                s.loss_cut_immediate += 1
            elif "EMERGENCY STOP" in line:
                s.emergency_stop += 1
            elif "REVERSAL EXIT" in line:
                s.reversal_exit += 1
            elif "PROFIT TAKE" in line:
                s.profit_take += 1
            elif "PYRAMID eligible" in line:
                s.pyramid_eligible += 1
            elif "ENTRY SUPPRESSED" in line:
                s.entry_suppressed += 1
            elif "placement race guard" in line:
                s.reconcile_skip_placement += 1
            elif "exit lag guard" in line:
                s.reconcile_skip_exit_lag += 1
            elif "API lag guard" in line:
                s.reconcile_skip_settled += 1
            elif "[RECONCILE] Position gap" in line:
                s.reconcile_adopt_gap += 1
            elif "WebSocket connected" in line and "connection #" in line:
                # only count reconnects (connection #2+)
                m = re.search(r"connection #(\d+)", line)
                if m and int(m.group(1)) > 1:
                    s.ws_reconnects += 1
            elif "Scan loop error" in line:
                s.scan_loop_errors += 1
            elif "ZeroDivisionError" in line:
                s.zerodiv_errors += 1
            elif "[STALE CACHE]" in line:
                s.stale_cache_warnings += 1
    s.cooloff_saved = max(0, s.cooloff_armed - s.cooloff_fired)
    return s


# ── Rendering ────────────────────────────────────────────────────────────────
def _pnl_color(v: float) -> str:
    if v > 0:    return "bright_green"
    if v < 0:    return "bright_red"
    return "white"


def build_mission_status(positions: list[Position], mode_filter: Optional[str],
                         since: Optional[datetime], total_rows: int) -> Panel:
    sessions = sorted({p.session for p in positions}, key=tag_sort_key)
    current_session = sessions[-1] if sessions else "—"
    first_ts = min((p.ts_open for p in positions), default=None)
    last_ts = max((p.ts_exit or p.ts_open for p in positions), default=None)

    lines = []
    lines.append(Text.from_markup(
        f"[bold bright_cyan]15MINBTC — MISSION CONTROL[/bold bright_cyan]   "
        f"[dim]snapshot @ {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}[/dim]"
    ))
    mode_label = mode_filter.upper() if mode_filter else "ALL (paper + live)"
    since_label = since.strftime("%Y-%m-%d") if since else "all time"
    lines.append(Text.from_markup(
        f"  [bold]Mode:[/bold] {mode_label}   "
        f"[bold]Since:[/bold] {since_label}   "
        f"[bold]Sessions:[/bold] {len(sessions)}   "
        f"[bold]Rows loaded:[/bold] {total_rows}"
    ))
    lines.append(Text.from_markup(
        f"  [bold]Range:[/bold] {fmt_dt(first_ts)}  →  {fmt_dt(last_ts)}   "
        f"[bold]Current session:[/bold] {current_session}"
    ))
    return Panel(Group(*lines), border_style="cyan", title="[bold]Status[/bold]",
                 padding=(0, 1))


def build_pnl_panel(positions: list[Position]) -> Panel:
    closed = [p for p in positions if p.pnl is not None]
    wins = [p for p in closed if p.pnl > 0]
    losses = [p for p in closed if p.pnl <= 0]
    open_pos = [p for p in positions if p.pnl is None]
    total_pnl = sum(p.pnl for p in closed)
    gross_cost = sum(p.entry_avg * p.qty / 100 for p in positions)
    win_rate = (len(wins) / len(closed) * 100) if closed else 0
    avg_win = (sum(p.pnl for p in wins) / len(wins)) if wins else 0
    avg_loss = (sum(p.pnl for p in losses) / len(losses)) if losses else 0
    per_trade = (total_pnl / len(closed)) if closed else 0

    # Side accuracy (only positions with known settlement)
    rated = [p for p in positions if p.correct is not None]
    correct = sum(1 for p in rated if p.correct)
    side_pct = (correct / len(rated) * 100) if rated else 0

    pnl_c = _pnl_color(total_pnl)
    edge_c = _pnl_color(per_trade)
    table = Table(box=None, show_header=False, padding=(0, 2))
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_row("Realized P&L",
                  Text.from_markup(f"[bold {pnl_c}]${total_pnl:+,.2f}[/bold {pnl_c}]"))
    table.add_row("Gross deployed", f"${gross_cost:,.2f}")
    roi = (total_pnl / gross_cost * 100) if gross_cost else 0
    roi_c = _pnl_color(roi)
    table.add_row("ROI", Text.from_markup(f"[{roi_c}]{roi:+.2f}%[/{roi_c}]"))
    table.add_row("", "")
    table.add_row("Positions", f"{len(positions)}  [closed: {len(closed)}  open: {len(open_pos)}]")
    table.add_row("Win rate",
                  f"{len(wins)}W / {len(losses)}L  ({win_rate:.1f}%)")
    table.add_row("Side accuracy",
                  f"{correct}/{len(rated)}  ({side_pct:.1f}%)")
    table.add_row("Per-trade edge",
                  Text.from_markup(f"[{edge_c}]${per_trade:+.2f}[/{edge_c}]"))
    table.add_row("Avg win / loss",
                  Text.from_markup(f"[bright_green]${avg_win:+.2f}[/bright_green] / "
                                   f"[bright_red]${avg_loss:+.2f}[/bright_red]"))
    return Panel(table, title="[bold]Headline P&L[/bold]", border_style="green")


def build_outcome_classes(positions: list[Position]) -> Panel:
    classes = ["correct_win", "shaken_out", "saved_by_exit", "wrong_loss"]
    labels = {"correct_win": "CW  correct_win", "shaken_out": "SO  shaken_out",
              "saved_by_exit": "SE  saved_by_exit", "wrong_loss": "WL  wrong_loss"}
    descs = {
        "correct_win":   "right side, exited or settled for profit",
        "shaken_out":    "right side, stopped before settling — leakage",
        "saved_by_exit": "wrong side, exited at profit — active mgmt win",
        "wrong_loss":    "wrong side, stopped or settled as loss",
    }
    colors = {"correct_win": "bright_green", "shaken_out": "bright_red",
              "saved_by_exit": "yellow", "wrong_loss": "bright_red"}

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                  padding=(0, 1))
    table.add_column("Class", style="bold")
    table.add_column("n", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("avg", justify="right")
    table.add_column("meaning", style="dim")

    closed_classified = [p for p in positions
                         if p.pnl is not None and p.klass in classes]
    total_pnl = sum(p.pnl for p in closed_classified)
    for k in classes:
        sub = [p for p in closed_classified if p.klass == k]
        n = len(sub)
        pnl = sum(p.pnl for p in sub)
        avg = pnl / max(n, 1)
        color = colors[k]
        table.add_row(
            Text.from_markup(f"[{color}]{labels[k]}[/{color}]"),
            f"{n}",
            Text.from_markup(f"[{_pnl_color(pnl)}]${pnl:+,.2f}[/{_pnl_color(pnl)}]"),
            Text.from_markup(f"[{_pnl_color(avg)}]${avg:+.2f}[/{_pnl_color(avg)}]"),
            descs[k],
        )
    # Total row
    table.add_section()
    table.add_row(
        "TOTAL (classified)",
        f"{len(closed_classified)}",
        Text.from_markup(f"[{_pnl_color(total_pnl)}]${total_pnl:+,.2f}[/{_pnl_color(total_pnl)}]"),
        "",
        Text.from_markup("[dim]positions with known settlement[/dim]"),
    )
    return Panel(table, title="[bold]Four Outcome Classes[/bold]", border_style="magenta")


def build_sessions_table_with_modes(positions: list[Position],
                                    rows: list[Row]) -> Panel:
    """Per-session table — mode label resolved from the underlying rows."""
    session_mode: dict = {}
    for r in rows:
        session_mode.setdefault(r.session, set()).add(r.mode)

    by_sess: dict = defaultdict(list)
    for p in positions:
        by_sess[p.session].append(p)

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold dim")
    table.add_column("session", style="cyan")
    table.add_column("dur", justify="right")
    table.add_column("mode", style="dim", justify="center")
    table.add_column("n", justify="right")
    table.add_column("W/L", justify="right")
    table.add_column("win%", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("gross", justify="right")
    table.add_column("ROI", justify="right")
    table.add_column("top exit reasons", style="dim")

    sorted_sess = sorted(by_sess.keys(), key=tag_sort_key)
    for tag in sorted_sess:
        ps = by_sess[tag]
        closed = [p for p in ps if p.pnl is not None]
        wins = [p for p in closed if p.pnl > 0]
        losses = [p for p in closed if p.pnl <= 0]
        pnl = sum(p.pnl for p in closed)
        gross = sum(p.entry_avg * p.qty / 100 for p in ps)
        roi = (pnl / gross * 100) if gross else 0
        wr = (len(wins) / len(closed) * 100) if closed else 0
        first_ts = min(p.ts_open for p in ps)
        last_ts = max((p.ts_exit or p.ts_open) for p in ps)
        dur_min = (last_ts - first_ts).total_seconds() / 60
        modes = session_mode.get(tag, set())
        if "live" in modes and "paper" in modes:
            mode_str = "[yellow]mixed[/yellow]"
        elif "live" in modes:
            mode_str = "[bright_red]LIVE[/bright_red]"
        else:
            mode_str = "paper"
        exit_mix = Counter(p.exit_bucket for p in closed)
        mix_str = " ".join(f"{k}:{v}" for k, v in exit_mix.most_common(3))

        pnl_c = _pnl_color(pnl)
        roi_c = _pnl_color(roi)
        table.add_row(
            tag,
            fmt_dur_min(dur_min),
            Text.from_markup(mode_str),
            str(len(ps)),
            f"{len(wins)}/{len(losses)}",
            f"{wr:.0f}%",
            Text.from_markup(f"[{pnl_c}]${pnl:+,.2f}[/{pnl_c}]"),
            f"${gross:.0f}",
            Text.from_markup(f"[{roi_c}]{roi:+.1f}%[/{roi_c}]"),
            mix_str,
        )
    return Panel(table, title="[bold]Per-Session Performance[/bold]",
                 border_style="cyan")


def build_exit_mix(positions: list[Position]) -> Panel:
    closed = [p for p in positions if p.pnl is not None]
    buckets = Counter(p.exit_bucket for p in closed)
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("exit reason", style="bold")
    table.add_column("n", justify="right")
    table.add_column("%", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("avg", justify="right")
    total = max(len(closed), 1)
    order = ["reversal", "profit_take", "loss_cut", "loss_cut_after_cooloff",
             "emergency_stop", "settled", "time_stop", "other"]
    for k in order + [b for b in buckets if b not in order]:
        if buckets[k] == 0:
            continue
        sub = [p for p in closed if p.exit_bucket == k]
        n = len(sub)
        pnl = sum(p.pnl for p in sub)
        avg = pnl / n
        c = _pnl_color(pnl)
        table.add_row(
            k, f"{n}", f"{n/total*100:.0f}%",
            Text.from_markup(f"[{c}]${pnl:+,.2f}[/{c}]"),
            Text.from_markup(f"[{_pnl_color(avg)}]${avg:+.2f}[/{_pnl_color(avg)}]"),
        )
    return Panel(table, title="[bold]Exit Reason Mix[/bold]", border_style="yellow")


def build_entry_mix(positions: list[Position]) -> Panel:
    buckets = Counter(p.entry_bucket for p in positions)
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("entry source", style="bold")
    table.add_column("n", justify="right")
    table.add_column("%", justify="right")
    table.add_column("P&L (closed)", justify="right")
    table.add_column("avg", justify="right")
    total = max(len(positions), 1)
    for k, n in buckets.most_common():
        sub = [p for p in positions if p.entry_bucket == k]
        closed_sub = [p for p in sub if p.pnl is not None]
        pnl = sum(p.pnl for p in closed_sub)
        avg = (pnl / len(closed_sub)) if closed_sub else 0
        c = _pnl_color(pnl)
        table.add_row(
            k, f"{n}", f"{n/total*100:.0f}%",
            Text.from_markup(f"[{c}]${pnl:+,.2f}[/{c}]"),
            Text.from_markup(f"[{_pnl_color(avg)}]${avg:+.2f}[/{_pnl_color(avg)}]"),
        )
    return Panel(table, title="[bold]Entry Source Mix[/bold]", border_style="blue")


def build_pyramid(positions: list[Position]) -> Panel:
    closed = [p for p in positions if p.pnl is not None]
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("legs", justify="right", style="bold")
    table.add_column("n", justify="right")
    table.add_column("W/L", justify="right")
    table.add_column("win%", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("avg", justify="right")
    for legs in sorted({p.n_legs for p in closed}):
        sub = [p for p in closed if p.n_legs == legs]
        wins = sum(1 for p in sub if p.pnl > 0)
        losses = len(sub) - wins
        pnl = sum(p.pnl for p in sub)
        wr = wins / len(sub) * 100 if sub else 0
        avg = pnl / len(sub) if sub else 0
        c = _pnl_color(pnl)
        label = f"{legs}-leg" + (" (base)" if legs == 1 else " (pyramid)")
        table.add_row(
            label, f"{len(sub)}", f"{wins}/{losses}",
            f"{wr:.0f}%",
            Text.from_markup(f"[{c}]${pnl:+,.2f}[/{c}]"),
            Text.from_markup(f"[{_pnl_color(avg)}]${avg:+.2f}[/{_pnl_color(avg)}]"),
        )
    return Panel(table, title="[bold]Pyramid Depth Analysis[/bold]", border_style="magenta")


def build_cooloff(stats: BotLogStats) -> Panel:
    """Cool-off effectiveness — shows Fix #1 + tweak (d234614) in action."""
    armed = stats.cooloff_armed
    fired = stats.cooloff_fired
    saved = stats.cooloff_saved
    save_pct = (saved / armed * 100) if armed else 0
    fire_pct = (fired / armed * 100) if armed else 0

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_column(style="dim")
    table.add_row("STOP SUPPRESSED total", f"{stats.suppressed}",
                  "model still likes our side — held through dip")
    table.add_row("  ↳ legit (rec=side, conf≥40%)",
                  f"{stats.suppressed_legit}",
                  "high-conf model agreement")
    table.add_row("  ↳ edge-floor (rec=side, conf<40%)",
                  f"{stats.suppressed_edge_floor_rec_side}",
                  "low-conf agreement saved by edge floor")
    anomaly_color = "bright_red" if stats.suppressed_anomaly else "dim"
    table.add_row("  ↳ anomaly (rec=none)",
                  Text.from_markup(f"[{anomaly_color}]{stats.suppressed_anomaly}[/{anomaly_color}]"),
                  Text.from_markup("[bright_red]should be 0 after d234614[/bright_red]" if stats.suppressed_anomaly else "should be 0 ✓"))
    table.add_section()
    table.add_row("LOSS CUT PENDING armed", f"{armed}",
                  "cool-off window started (panic-flush guard)")
    save_color = "bright_green" if save_pct >= 60 else ("yellow" if save_pct >= 40 else "bright_red")
    table.add_row("  ↳ saved by cool-off",
                  Text.from_markup(f"[{save_color}]{saved}  ({save_pct:.0f}%)[/{save_color}]"),
                  "model flipped back to agreeing OR pnl recovered")
    table.add_row("  ↳ fired after cool-off",
                  f"{fired}  ({fire_pct:.0f}%)",
                  "conditions persisted — cut fired")
    table.add_section()
    table.add_row("LOSS CUT (immediate, <240s runway)",
                  f"{stats.loss_cut_immediate}",
                  "no cool-off applied — decisive cut")
    emer_color = "bright_red" if stats.emergency_stop else "dim"
    table.add_row("EMERGENCY STOP (-65%)",
                  Text.from_markup(f"[{emer_color}]{stats.emergency_stop}[/{emer_color}]"),
                  "absolute backstop, bypasses cool-off")

    return Panel(table, title="[bold]Cool-off Effectiveness  [dim](Fix #1)[/dim][/bold]",
                 border_style="bright_blue")


def build_reconciler(stats: BotLogStats) -> Panel:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_column(style="dim")
    table.add_row("Skipped — placement race", f"{stats.reconcile_skip_placement}",
                  "guard added in commit 6801952")
    table.add_row("Skipped — exit lag (10-30s)", f"{stats.reconcile_skip_exit_lag}",
                  "guard against API position-clear lag")
    table.add_row("Skipped — settlement lag", f"{stats.reconcile_skip_settled}",
                  "guard against settled-position re-adoption")
    adopt_color = "yellow" if stats.reconcile_adopt_gap > 0 else "dim"
    table.add_row("Gap adoptions",
                  Text.from_markup(f"[{adopt_color}]{stats.reconcile_adopt_gap}[/{adopt_color}]"),
                  "real gap detected — bot adopted missing position")
    return Panel(table, title="[bold]Reconciler Activity[/bold]",
                 border_style="bright_blue")


def build_notable(positions: list[Position], n: int = 5) -> Panel:
    closed = [p for p in positions if p.pnl is not None]
    wins = sorted(closed, key=lambda p: -p.pnl)[:n]
    losses = sorted(closed, key=lambda p: p.pnl)[:n]

    def make_table(title_color: str, items: list[Position]) -> Table:
        t = Table(box=None, show_header=True, header_style="bold dim",
                  padding=(0, 1))
        t.add_column("session", style="dim", no_wrap=True)
        t.add_column("ticker", style="cyan", no_wrap=True)
        t.add_column("side", justify="center")
        t.add_column("qty", justify="right")
        t.add_column("entry → exit", justify="right")
        t.add_column("P&L", justify="right")
        t.add_column("exit reason", style="dim")
        for p in items:
            c = _pnl_color(p.pnl)
            entry_exit = f"{p.entry_avg:.0f}¢ → {p.exit_avg:.0f}¢" if p.exit_avg else f"{p.entry_avg:.0f}¢"
            t.add_row(
                p.session, p.ticker, p.side, str(p.qty), entry_exit,
                Text.from_markup(f"[{c}]${p.pnl:+,.2f}[/{c}]"),
                p.exit_bucket,
            )
        return t

    table_wins = make_table("bright_green", wins)
    table_losses = make_table("bright_red", losses)

    return Panel(
        Group(
            Text.from_markup(f"[bold bright_green]TOP {len(wins)} WINS[/bold bright_green]"),
            table_wins,
            Text.from_markup(""),
            Text.from_markup(f"[bold bright_red]TOP {len(losses)} LOSSES[/bold bright_red]"),
            table_losses,
        ),
        title="[bold]Notable Trades[/bold]", border_style="magenta",
    )


def build_health(stats: BotLogStats) -> Panel:
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(style="bold")
    table.add_column(justify="right")
    table.add_column(style="dim")
    table.add_row("WebSocket reconnects", f"{stats.ws_reconnects}",
                  "Kalshi WS dropped + reconnected")
    err_color = "bright_red" if stats.scan_loop_errors else "dim"
    table.add_row("Scan loop errors",
                  Text.from_markup(f"[{err_color}]{stats.scan_loop_errors}[/{err_color}]"),
                  "exceptions in the main scan loop")
    zdiv_color = "bright_red" if stats.zerodiv_errors else "dim"
    table.add_row("ZeroDivisionErrors",
                  Text.from_markup(f"[{zdiv_color}]{stats.zerodiv_errors}[/{zdiv_color}]"),
                  "should be 0 after commit 8b65781")
    table.add_row("Stale cache warnings", f"{stats.stale_cache_warnings}",
                  "WS data >3s old — triggered REST refresh")
    table.add_section()
    table.add_row("PYRAMID eligible (entries considered)",
                  f"{stats.pyramid_eligible}", "model OK to add to position")
    table.add_row("ENTRY SUPPRESSED",
                  f"{stats.entry_suppressed}", "high-edge/low-conf skip etc.")
    return Panel(table, title="[bold]System Health[/bold]", border_style="white")


# ── Orchestration ────────────────────────────────────────────────────────────
def run_report(mode_filter: Optional[str], since: Optional[datetime],
               no_fetch: bool, trades_path: Path = TRADES_CSV) -> None:
    """Build and render the full report."""
    if not trades_path.exists():
        console.print(f"[bright_red]trades.csv not found at {trades_path}[/bright_red]")
        return

    rows = load_rows(trades_path, mode_filter, since)
    if not rows:
        console.print("[yellow]No trades matched the filters.[/yellow]")
        return

    positions = build_positions(rows)
    cache = load_cache()
    tickers = sorted({p.ticker for p in positions})
    if not no_fetch:
        cache = fetch_results(tickers, cache)
    classify_positions(positions, cache)

    botlog_stats = scan_botlog(since)

    console.print()
    console.print(build_mission_status(positions, mode_filter, since, len(rows)))
    console.print()
    console.print(build_pnl_panel(positions))
    console.print()
    console.print(build_outcome_classes(positions))
    console.print()
    console.print(build_sessions_table_with_modes(positions, rows))
    console.print()
    console.print(build_exit_mix(positions))
    console.print()
    console.print(build_entry_mix(positions))
    console.print()
    console.print(build_pyramid(positions))
    console.print()
    console.print(build_cooloff(botlog_stats))
    console.print()
    console.print(build_reconciler(botlog_stats))
    console.print()
    console.print(build_notable(positions))
    console.print()
    console.print(build_health(botlog_stats))
    console.print()
