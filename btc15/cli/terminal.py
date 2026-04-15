"""
Rich live dashboard for the bot.

Layout:
  ┌─────────────────────────────────────────────────────┐
  │  HEADER: BTC price, feed status, mode               │
  ├──────────────┬──────────────────┬───────────────────┤
  │  SIGNALS     │  OPEN MARKETS    │  RISK / BALANCE   │
  ├──────────────┴──────────────────┴───────────────────┤
  │  POSITIONS (table)                                  │
  ├─────────────────────────────────────────────────────┤
  │  TRADE LOG (scrolling)                              │
  └─────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


console = Console()


def _price_color(price: float, ref: float) -> str:
    if price > ref:
        return "bright_green"
    elif price < ref:
        return "bright_red"
    return "white"


def _pnl_color(pnl: float) -> str:
    return "bright_green" if pnl >= 0 else "bright_red"


def _sig_color(signal: str) -> str:
    s = signal.upper()
    if "STRONG" in s:
        return "bright_green" if "YES" in s else "bright_red"
    if "WEAK" in s:
        return "yellow"
    return "dim white"


def build_header(state: dict) -> Panel:
    price = state.get("current_price", 0.0)
    feed_age = state.get("feed_age_sec", 0.0)
    status = state.get("status", "idle").upper()

    # ── Mode flags (left side) ────────────────────────────────────────────
    mode = []
    if state.get("paper_mode"):
        mode.append("[yellow]PAPER[/yellow]")
    else:
        mode.append("[bright_red]LIVE[/bright_red]")
    if state.get("auto_trade"):
        mode.append("[bright_green]AUTO-TRADE ON[/bright_green]")
    else:
        mode.append("[dim]SIGNAL-ONLY[/dim]")
    mode_str = " | ".join(mode)

    # ── Feed / scan (middle) ──────────────────────────────────────────────
    feed_color = "bright_green" if feed_age < 5 else ("yellow" if feed_age < 30 else "bright_red")
    last_scan = state.get("last_scan", "never")
    if last_scan and last_scan != "never":
        try:
            dt = datetime.fromisoformat(last_scan)
            ago = (datetime.now(timezone.utc) - dt).total_seconds()
            last_scan = f"{ago:.0f}s ago"
        except Exception:
            pass

    # ── Right-side stats ──────────────────────────────────────────────────
    risk = state.get("risk", {})
    wr = risk.get("win_rate")
    if wr is not None:
        wr_color = "bright_green" if wr >= 0.55 else ("yellow" if wr >= 0.45 else "bright_red")
        wr_str = f"[{wr_color}]WR {wr:.0%}[/{wr_color}]"
    else:
        wr_str = "[dim]WR --[/dim]"

    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.strftime("%H:%M:%S") + " UTC"

    session_str = ""
    try:
        session_start = state.get("session_start")
        if session_start:
            elapsed = (now_utc - datetime.fromisoformat(session_start)).total_seconds()
            h, rem = divmod(int(elapsed), 3600)
            m, s = divmod(rem, 60)
            session_str = f"[dim]up {h:02d}:{m:02d}:{s:02d}[/dim]"
    except Exception:
        pass

    content = (
        f"  [dim]15 MIN OF FAME[/dim]"
        f"   [bold white]BTC[/bold white] [bold bright_cyan]${price:,.2f}[/bold bright_cyan]"
        f"   {mode_str}"
        f"   [dim]{status}[/dim]"
        f"   [{feed_color}]feed: {feed_age:.1f}s[/{feed_color}]"
        f"   [dim]scan: {last_scan}[/dim]"
        f"   {wr_str}"
        f"   {session_str}"
        f"   [dim]{now_str}[/dim]"
    )
    return Panel(content, box=box.DOUBLE_EDGE)


def build_signals_panel(state: dict) -> Panel:
    signals = state.get("signals", {})
    if not signals:
        return Panel("[dim]Waiting for markets...[/dim]", title="Signals")

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Strike", justify="right")
    table.add_column("T-Left", justify="right")
    table.add_column("P(YES)", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("Signal", justify="center")

    MIN_CONF = 0.50  # must match config.yaml min_confidence

    for ticker, s in list(signals.items())[:8]:
        sig_str = s.get("signal", "NEUTRAL")
        edge_yes = s.get("edge_yes", 0)
        edge_no = s.get("edge_no", 0)
        best_edge = max(edge_yes, edge_no)
        edge_color = "bright_green" if best_edge > 0.04 else ("yellow" if best_edge > 0.01 else "dim")
        secs = s.get("seconds_left", 0)
        mins = secs // 60
        sec_r = secs % 60
        conf = s.get("confidence", 0.0)
        gap = conf - MIN_CONF
        if conf >= MIN_CONF:
            conf_color = "bright_green"
        elif gap >= -0.05:   # within 5% of threshold
            conf_color = "yellow"
        else:
            conf_color = "dim"
        conf_str = f"[{conf_color}]{conf:.0%}[/{conf_color}]"
        table.add_row(
            ticker[-15:],
            f"${s.get('strike', 0):,.0f}",
            f"{mins}m{sec_r:02d}s",
            f"{s.get('prob_yes', 0.5):.1%}",
            conf_str,
            f"[{edge_color}]{best_edge:+.1%}[/{edge_color}]",
            f"[{_sig_color(sig_str)}]{sig_str}[/{_sig_color(sig_str)}]",
        )
    return Panel(table, title="[bold]Signals[/bold]", border_style="blue")


def build_markets_panel(state: dict) -> Panel:
    markets = state.get("open_markets", [])
    if not markets:
        return Panel("[dim]No open KXBTC markets[/dim]", title="Markets")

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim")
    table.add_column("Strike", justify="right")
    table.add_column("YES ask", justify="right")   # cost to buy YES
    table.add_column("NO ask",  justify="right")   # cost to buy NO = 100 − YES bid
    table.add_column("Vol", justify="right")
    table.add_column("T-Left", justify="right")

    for m in markets[:8]:
        secs = m.get("seconds_left", 0)
        mins = secs // 60
        sec_r = secs % 60
        yes_ask = m.get("yes_ask")   # None = no data; 0 is not a valid Kalshi price
        yes_bid = m.get("yes_bid")
        # NO ask = 100 − YES bid (what you pay to enter a NO position)
        no_ask = round(100 - yes_bid) if yes_bid is not None else None
        table.add_row(
            f"${m.get('strike', 0):,.0f}",
            f"[bright_green]{yes_ask:.0f}¢[/bright_green]" if yes_ask is not None else "[dim]--[/dim]",
            f"[bright_red]{no_ask:.0f}¢[/bright_red]" if no_ask is not None else "[dim]--[/dim]",
            f"{m.get('volume', 0):,}",
            f"{mins}m{sec_r:02d}s",
        )
    return Panel(table, title="[bold]Open Markets[/bold]", border_style="cyan")


def build_risk_panel(state: dict) -> Panel:
    risk = state.get("risk", {})
    balance = state.get("balance", {})

    # Single source of truth: account-level PnL from Kalshi (includes fees + unrealized).
    # Falls back to internal realized sum while the first 30s balance refresh is pending.
    true_pnl = balance.get("true_pnl") if balance else None
    pnl = true_pnl if true_pnl is not None else risk.get("daily_pnl", 0.0)
    pnl_color = _pnl_color(pnl)

    lines = []
    if balance:
        lines.append(f"[bold]Available:[/bold] [bright_cyan]${balance.get('available', 0):,.2f}[/bright_cyan]")
        lines.append(f"[bold]Portfolio:[/bold] ${balance.get('portfolio', 0):,.2f}")
    lines.append(f"[bold]PnL:[/bold] [{pnl_color}][bold]${pnl:+.2f}[/bold][/{pnl_color}]")
    lines.append(f"[bold]Trades:[/bold] {risk.get('daily_trades', 0)}")
    lines.append(f"[bold]Open:[/bold] {risk.get('open_positions', 0)}")

    wr = risk.get("win_rate")
    if wr is not None:
        wr_color = "bright_green" if wr >= 0.55 else ("yellow" if wr >= 0.45 else "bright_red")
        lines.append(f"[bold]Win Rate:[/bold] [{wr_color}]{wr:.1%}[/{wr_color}]")

    if risk.get("halted"):
        lines.append(f"\n[bold bright_red]HALTED: {risk.get('halt_reason', '')}[/bold bright_red]")

    content = "\n".join(lines)
    border = "bright_red" if risk.get("halted") else "green"
    return Panel(content, title="[bold]Risk / Balance[/bold]", border_style=border)


def build_positions_panel(state: dict) -> Panel:
    positions = state.get("open_positions", [])
    if not positions:
        return Panel("[dim]No open positions[/dim]", title="Positions", border_style="dim")

    persona_colors = {"sniper": "bright_magenta", "scalper": "bright_cyan", "arb": "bright_yellow", "auto": "white"}
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim", padding=(0, 1))
    table.add_column("Ticker", style="cyan", no_wrap=True)
    table.add_column("Src", justify="center")
    table.add_column("Side", justify="center")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("PnL", justify="right")

    for p in positions:
        pnl = p.get("pnl", 0)
        pnl_str = f"[{_pnl_color(pnl)}]{pnl:+.2f}[/{_pnl_color(pnl)}]"
        side = p.get("side", "")
        side_color = "bright_green" if side == "yes" else "bright_red"
        src = p.get("source", "")
        src_color = persona_colors.get(src, "white")
        src_label = src[:3].upper()
        table.add_row(
            p.get("ticker", "")[-16:],
            f"[{src_color}]{src_label}[/{src_color}]",
            f"[{side_color}]{side.upper()}[/{side_color}]",
            str(p.get("contracts", 0)),
            f"{p.get('entry_cents', 0)}¢",
            f"${p.get('cost', 0):.2f}",
            pnl_str,
        )
    return Panel(table, title=f"[bold]Positions ({len(positions)})[/bold]", border_style="yellow")


def build_trades_panel(state: dict) -> Panel:
    trades = state.get("recent_trades", [])
    if not trades:
        return Panel("[dim]No trades yet[/dim]", title="Trade Log")

    lines = []
    for t in trades[:30]:
        ts = t.get("entry_time", "")
        try:
            dt = datetime.fromisoformat(ts)
            ts_str = dt.strftime("%H:%M:%S")
        except Exception:
            ts_str = ts[:8]

        side = t.get("side", "")
        is_settled = "settled" in side
        is_exit = "exit" in side
        base_side = side.replace("_settled", "").replace("_exit", "")
        side_color = "bright_green" if base_side == "yes" else "bright_red"
        ticker = t.get("ticker", "")[-15:]
        contracts = t.get("contracts", 0)
        price = t.get("price_cents", 0)
        source = t.get("source", "")
        # Paper trade tag
        paper_tag = "[yellow][P][/yellow] " if "paper" in source else ""
        # Persona tag
        persona_colors = {"sniper": "bright_magenta", "scalper": "bright_cyan", "arb": "bright_yellow"}
        persona = source.split("/")[0] if "/" in source else source
        p_color = persona_colors.get(persona, "dim")
        p_tag = f"[{p_color}][{persona[:3].upper()}][/{p_color}] " if persona in persona_colors else ""
        # Entry / exit / settled action label
        if is_settled:
            action_tag = "[bold bright_white]SETTLED[/bold bright_white]"
        elif is_exit:
            action_tag = f"[yellow]EXIT[/yellow] [{side_color}]{base_side.upper()}[/{side_color}]"
        else:
            action_tag = f"[{side_color}]{side.upper()}[/{side_color}]"
        # Value string: entries show cost, exits show P&L
        if is_exit or is_settled:
            pnl = t.get("pnl")
            if pnl is not None:
                pnl_c = "bright_green" if pnl >= 0 else "bright_red"
                value_str = f"x{contracts} @ {price}¢ [{pnl_c}][bold]${pnl:+.2f}[/bold][/{pnl_c}]"
            else:
                value_str = f"x{contracts} @ [bold]{price}¢[/bold]"
        else:
            cost = contracts * price / 100
            value_str = f"x{contracts} @ {price}¢ [dim](${cost:.2f})[/dim]"
        tid = t.get("trade_id", "")
        tid_str = f"[dim]{tid}[/dim] " if tid else ""
        line = (
            f"{tid_str}"
            f"[dim]{ts_str}[/dim] "
            f"{paper_tag}"
            f"{p_tag}"
            f"{action_tag} "
            f"[cyan]{ticker}[/cyan] "
            f"{value_str}"
        )
        lines.append(line)

    return Panel("\n".join(lines), title="[bold]Trade Log[/bold]", border_style="dim")


# ── Block bar chart renderer ─────────────────────────────────────────────────
# Uses ▁▂▃▄▅▆▇█ (8 sub-levels per text row).
# Each column = one data point. Bars grow up from zero (positive = green,
# negative = red). Works well with 2 points or 200.
_VBLOCKS = " ▁▂▃▄▅▆▇█"   # index 0=empty, 1–8 = 1/8..8/8 fill from bottom
_VBLOCKS_TOP = " ▔▔▔▄▄▄▄█"  # top-anchored for negative bars (fills from top)

def _bar_chart(values: list[float], width: int, height: int) -> list[str]:
    """
    Filled bar chart. Each column = one value. Positive bars grow upward
    (green), negative bars grow downward (red), both anchored at zero.
    Returns `height` Rich-markup text rows, top-first.
    Downsamples when more values than width; upsamples (stretches) when fewer.
    """
    if not values:
        return ["[dim]no data[/dim]"] + [""] * (height - 1)

    n = len(values)
    if n > width:
        # Downsample: pick evenly-spaced samples
        step = n / width
        cols = [values[int(i * step)] for i in range(width)]
    elif n < width:
        # Upsample: stretch each bar proportionally to fill available width
        base = width // n
        extra = width % n
        cols = []
        for i, v in enumerate(values):
            repeat = base + (1 if i < extra else 0)
            cols.extend([v] * repeat)
    else:
        cols = list(values)

    vmin = min(min(cols), 0.0)
    vmax = max(max(cols), 0.0)
    span = vmax - vmin or 1.0

    LEVELS = height * 8  # total sub-levels in the chart

    def to_sub(v: float) -> int:
        """Map a value to a sub-level index (0=bottom, LEVELS=top)."""
        return int((v - vmin) / span * LEVELS)

    zero_sub = to_sub(0.0)
    col_subs  = [to_sub(v) for v in cols]

    rows = []
    for row_i in range(height):
        # row_i=0 → top of chart; row_i=height-1 → bottom
        row_bot = (height - 1 - row_i) * 8  # sub-level at bottom of this text row
        row_top = row_bot + 8                # sub-level at top (exclusive)

        line = ""
        for v, cs in zip(cols, col_subs):
            fill_lo = min(zero_sub, cs)
            fill_hi = max(zero_sub, cs)

            # No fill in this text row at all?
            if fill_hi <= row_bot or fill_lo >= row_top:
                line += " "
                continue

            color = "bright_green" if v >= 0.0 else "bright_red"

            # Is this row fully inside the fill range?
            if fill_lo <= row_bot and fill_hi >= row_top:
                line += f"[{color}]█[/{color}]"
                continue

            # Partial cell — determine which fraction is filled.
            # Block chars grow from the bottom of the cell, so we count
            # sub-levels from row_bot up to where the fill ends.
            filled = min(fill_hi, row_top) - row_bot
            filled = max(1, min(8, round(filled)))
            char = _VBLOCKS[filled]
            line += f"[{color}]{char}[/{color}]"

        rows.append(line)

    return rows


def build_pnl_chart(state: dict) -> Panel:
    """P&L bar chart: one bar per closed trade, cumulative P&L curve."""
    risk = state.get("risk", {})
    balance = state.get("balance", {})

    # Same single source of truth as build_risk_panel:
    # account-level PnL when available, internal realized sum as fallback.
    true_pnl = balance.get("true_pnl") if balance else None
    pnl_ref = true_pnl if true_pnl is not None else risk.get("daily_pnl", 0.0)
    pnl_c = "bright_green" if pnl_ref >= 0 else "bright_red"

    # Build cumulative P&L series from closed trades in the log
    trades = state.get("recent_trades", [])
    closed = [
        t for t in reversed(trades)          # chronological order
        if t.get("pnl") is not None
        and ("exit" in t.get("side", "") or "settled" in t.get("side", ""))
    ]

    if len(closed) < 1:
        content = (
            f"PnL: [{pnl_c}][bold]${pnl_ref:+.2f}[/bold][/{pnl_c}]\n\n"
            f"[dim]Waiting for first closed trade...[/dim]"
        )
        return Panel(
            Align(Text.from_markup(content), align="center", vertical="middle"),
            title="[bold]PnL[/bold]",
            border_style="dim",
        )

    # Anchor cumulative series so the final bar equals pnl_ref.
    # recent_trades is capped at 50 entries so early trades may have rolled off;
    # the offset absorbs any gap so the chart endpoint always matches the panel value.
    visible_sum = sum(t["pnl"] for t in closed)
    running = pnl_ref - visible_sum
    cumulative = []
    for t in closed:
        running += t["pnl"]
        cumulative.append(running)

    # Chart dimensions — fill the available panel width
    # Right panel is ~half terminal width; subtract borders, padding, y-axis prefix
    avail_width = max(20, int((console.width // 2 - 8) * 0.80))
    chart_height = max(4, len(closed) and 5)   # 5 text rows = 40 sub-levels
    chart_width  = avail_width                  # _bar_chart handles up/downsampling

    chart_rows = _bar_chart(cumulative, chart_width, chart_height)

    # Y-axis labels (hi on top, zero line, lo on bottom)
    vmax = max(max(cumulative), 0.0)
    vmin = min(min(cumulative), 0.0)
    hi_lbl  = f"[dim]{vmax:+.2f}[/dim]"
    lo_lbl  = f"[dim]{vmin:+.2f}[/dim]"

    # Zero row indicator — which text row contains zero?
    span = vmax - vmin or 1.0
    zero_frac = (0.0 - vmin) / span          # 0=bottom, 1=top
    zero_row  = chart_height - 1 - int(zero_frac * (chart_height - 1))

    chart_lines = []
    for i, row_str in enumerate(chart_rows):
        if i == zero_row:
            prefix = "[dim]──[/dim]"
        else:
            prefix = "  "
        chart_lines.append(f"{prefix}{row_str}")

    # Footer: trade count, wins, losses
    wins   = sum(1 for t in closed if t["pnl"] > 0)
    losses = sum(1 for t in closed if t["pnl"] <= 0)
    n      = len(closed)
    wr_str = f"[bright_green]{wins}W[/bright_green] [bright_red]{losses}L[/bright_red] [dim]/{n}[/dim]"

    content = (
        f" {hi_lbl}\n"
        + "\n".join(chart_lines)
        + f"\n {lo_lbl}\n"
        + f"  {wr_str}   [{pnl_c}][bold]${pnl_ref:+.2f}[/bold][/{pnl_c}] session"
    )
    trend_color = "bright_green" if pnl_ref >= 0 else "bright_red"
    return Panel(
        Align(Text.from_markup(content), align="center", vertical="middle"),
        title="[bold]PnL[/bold]",
        border_style=trend_color,
    )


def build_personas_panel(state: dict) -> Panel:
    personas = state.get("personas", {})
    if not personas:
        return Panel("[dim]Personas inactive[/dim]", title="Personas")

    lines = []
    icons = {"sniper": ("SNP", "bright_magenta"), "scalper": ("SCA", "bright_cyan"), "arb": ("ARB", "bright_yellow")}
    for name, info in personas.items():
        tag, color = icons.get(name, (name[:3].upper(), "white"))
        pnl = info.get("daily_pnl", 0)
        pnl_c = "bright_green" if pnl >= 0 else "bright_red"
        trades = info.get("daily_trades", 0)
        positions = info.get("positions", 0)
        resting = info.get("resting_orders", 0)
        inv = info.get("inventory", "")
        inv_str = f" inv={inv}" if inv else ""
        lines.append(
            f"[{color}][bold]{tag}[/bold][/{color}] "
            f"PnL:[{pnl_c}]${pnl:+.2f}[/{pnl_c}] "
            f"T:{trades} P:{positions} R:{resting}{inv_str}"
        )
    return Panel("\n".join(lines), title="[bold]Personas[/bold]", border_style="magenta")


def build_event_log_panel(state: dict) -> Panel:
    events = state.get("event_log", [])
    if not events:
        return Panel("[dim]No events yet[/dim]", title="[bold]Engine Log[/bold]", border_style="dim")

    level_colors = {
        "WARNING": "yellow",
        "ERROR":   "bright_red",
        "CRITICAL":"bright_red",
        "INFO":    "dim",
    }
    lines = []
    for e in reversed(events[-22:]):  # most recent first, cap at 22
        lvl = e.get("level", "INFO")
        color = level_colors.get(lvl, "dim")
        ts = e.get("ts", "")
        msg = e.get("msg", "")
        # Truncate long messages to fit panel width
        msg = msg[:120] if len(msg) > 120 else msg
        lines.append(f"[dim]{ts}[/dim] [{color}]{msg}[/{color}]")

    return Panel("\n".join(lines), title="[bold]Engine Log[/bold]", border_style="dim")


def build_layout(state: dict) -> Layout:
    layout = Layout()
    # Three rows: header / top / bottom
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="top", size=12),
        Layout(name="bottom"),
    )
    # Top row unchanged: signals | markets | risk+personas
    layout["top"].split_row(
        Layout(name="signals"),
        Layout(name="markets"),
        Layout(name="right_col", size=30),
    )
    layout["right_col"].split_column(
        Layout(name="risk"),
    )
    # Bottom row: positions+trades stacked on left | chart+event_log stacked on right
    layout["bottom"].split_row(
        Layout(name="left_col"),
        Layout(name="right_bottom"),
    )
    layout["left_col"].split_column(
        Layout(name="positions", size=10),
        Layout(name="trades"),
    )
    layout["right_bottom"].split_column(
        Layout(name="chart"),
        Layout(name="event_log", size=10),
    )
    layout["header"].update(build_header(state))
    layout["signals"].update(build_signals_panel(state))
    layout["markets"].update(build_markets_panel(state))
    layout["risk"].update(build_risk_panel(state))
    layout["positions"].update(build_positions_panel(state))
    layout["trades"].update(build_trades_panel(state))
    layout["chart"].update(build_pnl_chart(state))
    layout["event_log"].update(build_event_log_panel(state))
    return layout


async def run_dashboard(engine, refresh_rate: float = 1.0):
    """Run the live Rich dashboard, refreshing every `refresh_rate` seconds."""
    with Live(
        build_layout(engine.state),
        console=console,
        refresh_per_second=1 / refresh_rate,
        screen=True,
    ) as live:
        while engine.running:
            live.update(build_layout(engine.state))
            await asyncio.sleep(refresh_rate)
