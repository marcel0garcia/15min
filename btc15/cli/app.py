"""
CLI command definitions.

Usage:
  python main.py                       → same as `run` in paper/signal mode
  python main.py run                   → start dashboard + strategy engine
  python main.py run --trade           → enable auto-trading
  python main.py run --live            → disable paper mode (REAL MONEY)
  python main.py markets               → list open KXBTC markets
  python main.py positions             → show current positions
  python main.py balance               → show balance
  python main.py history               → show trade log
  python main.py trade yes TICKER AMT  → manual trade
  python main.py config                → show config
  python main.py train                 → train ML model on collected data
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def setup_logging(level: str, log_file: str):
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stderr)]
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=fmt, handlers=handlers)
    # Quiet noisy libraries
    for lib in ("websockets", "aiohttp", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    # Suppress Python 3.10 SSL cleanup noise on shutdown
    import warnings
    warnings.filterwarnings("ignore", message=".*SSL transport.*")


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """15-Min BTC Prediction Bot for Kalshi."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(run)


@cli.command()
@click.option("--trade", is_flag=True, default=False, help="Enable auto-trading")
@click.option("--live", is_flag=True, default=False, help="Disable paper mode (real money!)")
@click.option("--web", is_flag=True, default=False, help="Also start web dashboard")
@click.option("--web-port", default=8080, help="Web dashboard port (default: 8080)")
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
def run(trade: bool, live: bool, web: bool, web_port: int, config_path: str):
    """Start the bot with live dashboard."""
    from btc15.config import load_config
    from pathlib import Path

    cfg = load_config(Path(config_path) if config_path else None)
    setup_logging(cfg.logging.level, cfg.logging.log_file)

    if trade:
        cfg.strategy.auto_trade = True
        console.print("[bold yellow]Auto-trading ENABLED[/bold yellow]")
    if live:
        cfg.strategy.paper_trade = False
        console.print("[bold bright_red]Paper mode DISABLED — REAL MONEY[/bold bright_red]")
        if not click.confirm("Are you sure you want to trade with real money?"):
            sys.exit(0)

    mode_parts = []
    if cfg.strategy.paper_trade:
        mode_parts.append("PAPER")
    else:
        mode_parts.append("LIVE")
    if cfg.strategy.auto_trade:
        mode_parts.append("AUTO-TRADE")
    else:
        mode_parts.append("SIGNAL-ONLY")

    console.print(f"Starting bot in [bold]{' | '.join(mode_parts)}[/bold] mode...")

    if not cfg.kalshi.api_key and not (cfg.kalshi.email and cfg.kalshi.password):
        console.print(
            "[yellow]WARNING: No Kalshi credentials found in .env — "
            "will run in price-feed-only mode[/yellow]"
        )

    if web:
        console.print(
            f"Web dashboard → [bold cyan]http://localhost:{web_port}[/bold cyan]"
        )

    asyncio.run(_run_engine(cfg, web_port=web_port if web else None))


async def _run_engine(cfg, web_port: int = None):
    from btc15.strategy.engine import StrategyEngine
    from btc15.cli.terminal import run_dashboard

    engine = StrategyEngine(cfg)
    tasks = []

    try:
        await engine.start()

        tasks.append(asyncio.create_task(run_dashboard(engine), name="terminal"))

        if web_port:
            try:
                import uvicorn
                from btc15.web.server import create_app
                app = create_app(engine)
                server = uvicorn.Server(
                    uvicorn.Config(app, host="0.0.0.0", port=web_port, log_level="error")
                )
                tasks.append(asyncio.create_task(server.serve(), name="web"))
            except ImportError:
                console.print("[yellow]fastapi/uvicorn not installed — web dashboard skipped[/yellow]")

        await asyncio.gather(*tasks)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        await engine.stop()
        console.print("\n[dim]Bot stopped.[/dim]")


@cli.command()
@click.option("--config", "config_path", default=None)
def markets(config_path: str):
    """List open KXBTC markets on Kalshi."""
    from btc15.config import load_config
    cfg = load_config(Path(config_path) if config_path else None)

    async def _run():
        from btc15.kalshi.client import KalshiClient
        async with KalshiClient(cfg.kalshi) as client:
            mkts = await client.get_markets(series_ticker=cfg.kalshi.series_ticker, status="open")

        if not mkts:
            console.print("[dim]No open KXBTC markets found.[/dim]")
            return

        table = Table(title="Open KXBTC Markets", box=box.ROUNDED)
        table.add_column("Ticker", style="cyan")
        table.add_column("Strike", justify="right")
        table.add_column("Yes Bid", justify="right")
        table.add_column("Yes Ask", justify="right")
        table.add_column("Volume", justify="right")
        table.add_column("Time Left", justify="right")

        for m in mkts:
            secs = m.seconds_remaining
            mins = int(secs // 60)
            sec_r = int(secs % 60)
            table.add_row(
                m.ticker,
                f"${m.strike_price:,.0f}",
                f"{m.yes_bid:.0f}¢",
                f"{m.yes_ask:.0f}¢",
                f"{m.volume:,}",
                f"{mins}m {sec_r:02d}s",
            )
        console.print(table)

    asyncio.run(_run())


@cli.command()
@click.option("--config", "config_path", default=None)
def positions(config_path: str):
    """Show current portfolio positions."""
    from btc15.config import load_config
    cfg = load_config(Path(config_path) if config_path else None)

    async def _run():
        from btc15.kalshi.client import KalshiClient
        async with KalshiClient(cfg.kalshi) as client:
            pos = await client.get_positions()
            bal = await client.get_balance()

        if not pos:
            console.print("[dim]No open positions.[/dim]")
        else:
            table = Table(title="Open Positions", box=box.ROUNDED)
            table.add_column("Ticker", style="cyan")
            table.add_column("Side")
            table.add_column("Contracts", justify="right")
            table.add_column("Cost", justify="right")
            table.add_column("Current Value", justify="right")
            table.add_column("Unrealized PnL", justify="right")
            for p in pos:
                pnl = p.unrealized_pnl
                color = "bright_green" if pnl >= 0 else "bright_red"
                side_color = "bright_green" if p.side.value == "yes" else "bright_red"
                table.add_row(
                    p.ticker,
                    f"[{side_color}]{p.side.value.upper()}[/{side_color}]",
                    str(p.contracts),
                    f"${p.cost_usd:.2f}",
                    f"${p.current_value_usd:.2f}",
                    f"[{color}]${pnl:+.2f}[/{color}]",
                )
            console.print(table)

        console.print(
            f"\nAvailable: [bold bright_cyan]${bal.available_usd:,.2f}[/bold bright_cyan]  "
            f"Portfolio: ${bal.portfolio_usd:,.2f}"
        )

    asyncio.run(_run())


@cli.command()
@click.option("--config", "config_path", default=None)
def balance(config_path: str):
    """Show Kalshi portfolio balance."""
    from btc15.config import load_config
    cfg = load_config(Path(config_path) if config_path else None)

    async def _run():
        from btc15.kalshi.client import KalshiClient
        async with KalshiClient(cfg.kalshi) as client:
            bal = await client.get_balance()
        console.print(f"Available: [bold bright_cyan]${bal.available_usd:,.2f}[/bold bright_cyan]")
        console.print(f"Portfolio: ${bal.portfolio_usd:,.2f}")

    asyncio.run(_run())


@cli.command()
@click.argument("side", type=click.Choice(["yes", "no"], case_sensitive=False))
@click.argument("ticker")
@click.argument("amount", type=float)
@click.option("--live", is_flag=True, default=False, help="Real money (default: paper)")
@click.option("--config", "config_path", default=None)
def trade(side: str, ticker: str, amount: float, live: bool, config_path: str):
    """Execute a manual trade. Amount is in USD."""
    from btc15.config import load_config
    cfg = load_config(Path(config_path) if config_path else None)
    if live:
        cfg.strategy.paper_trade = False
        if not click.confirm(f"Confirm {side.upper()} ${amount:.2f} on {ticker} with REAL money?"):
            return

    async def _run():
        from btc15.strategy.engine import StrategyEngine
        engine = StrategyEngine(cfg)
        from btc15.kalshi.client import KalshiClient
        async with KalshiClient(cfg.kalshi) as client:
            engine._kalshi = client
            result = await engine.manual_trade(ticker, side, amount)
        console.print(result)

    asyncio.run(_run())


@cli.command()
@click.option("-n", "--num", default=20, help="Number of trades to show")
def history(num: int):
    """Show trade history from log file."""
    from btc15.config import get_config
    cfg = get_config()
    log_path = Path(cfg.logging.trade_log_file)
    if not log_path.exists():
        console.print("[dim]No trade history yet.[/dim]")
        return

    import csv

    def _fmt_ticker(raw: str) -> str:
        """KXBTC15M-26MAR292245-45 → MAR29 22:45"""
        try:
            parts = raw.split('-')
            dt = parts[1]
            return f"{dt[2:5]}{dt[5:7]} {dt[7:9]}:{dt[9:]}"
        except Exception:
            return raw[-15:]

    with open(log_path) as f:
        reader = csv.reader(f)
        raw_rows = list(reader)

    if not raw_rows:
        console.print("[dim]No trade history yet.[/dim]")
        return

    # Detect CSV format by inspecting the header row
    header = raw_rows[0]
    new_format = header[0] == "trade_id"   # new: trade_id, timestamp, ticker, side, contracts, price_cents, cost_usd, source, mode[, session]
    # old format: timestamp, ticker, side, contracts, price_cents, cost_usd, source

    def _extract(row: list) -> dict:
        if new_format:
            return {
                "trade_id": row[0] if len(row) > 0 else "",
                "timestamp": row[1] if len(row) > 1 else "",
                "ticker":    row[2] if len(row) > 2 else "",
                "side":      row[3] if len(row) > 3 else "",
                "contracts": row[4] if len(row) > 4 else "",
                "price_cents": row[5] if len(row) > 5 else "",
                "cost_usd":  row[6] if len(row) > 6 else "",
                "source":    row[7] if len(row) > 7 else "",
                "mode":      row[8] if len(row) > 8 else "",
                "session":   row[9] if len(row) > 9 else "",
            }
        else:
            # Old 7-col format; detect per-row if it's actually a new-format row
            # (new rows written into old-header file have 9 values)
            if len(row) >= 9 and row[0].startswith("T"):
                return {
                    "trade_id": row[0], "timestamp": row[1], "ticker": row[2],
                    "side": row[3], "contracts": row[4], "price_cents": row[5],
                    "cost_usd": row[6], "source": row[7], "mode": row[8],
                    "session": row[9] if len(row) > 9 else "",
                }
            return {
                "trade_id": "", "timestamp": row[0] if len(row) > 0 else "",
                "ticker": row[1] if len(row) > 1 else "",
                "side": row[2] if len(row) > 2 else "",
                "contracts": row[3] if len(row) > 3 else "",
                "price_cents": row[4] if len(row) > 4 else "",
                "cost_usd": row[5] if len(row) > 5 else "",
                "source": row[6] if len(row) > 6 else "",
                "mode": "", "session": "",
            }

    data_rows = raw_rows[1:]  # skip header

    table = Table(title=f"Last {num} Trades", box=box.ROUNDED)
    table.add_column("ID", style="dim")
    table.add_column("Time")
    table.add_column("Ticker", style="cyan")
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Mode", justify="center")
    table.add_column("Session", style="dim")
    table.add_column("Source")

    for row in data_rows[-num:]:
        r = _extract(row)
        side = r["side"]
        base_side = side.replace("_settled", "").replace("_exit", "")
        color = "bright_green" if base_side == "yes" else "bright_red"
        if "exit" in side or "settled" in side:
            side_display = f"[yellow]{side.upper()}[/yellow]"
        else:
            side_display = f"[{color}]{side.upper()}[/{color}]"
        ts = r["timestamp"][:19].replace("T", " ")
        tid = r["trade_id"][-8:] if r["trade_id"] else ""
        table.add_row(
            tid,
            ts,
            _fmt_ticker(r["ticker"]),
            side_display,
            r["contracts"],
            f"{r['price_cents']}¢",
            f"${r['cost_usd']}",
            r["mode"] or "—",
            r["session"] or "—",
            r["source"],
        )
    console.print(table)


@cli.command(name="config")
@click.option("--config", "config_path", default=None)
def show_config(config_path: str):
    """Show current configuration."""
    from btc15.config import load_config
    import dataclasses, json
    cfg = load_config(Path(config_path) if config_path else None)

    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            d = {}
            for f in dataclasses.fields(obj):
                v = getattr(obj, f.name)
                if f.name in ("password", "api_key") and v:
                    v = "***"
                d[f.name] = to_dict(v)
            return d
        return obj

    console.print_json(json.dumps(to_dict(cfg), indent=2))


@cli.command()
@click.option("--paper", "paper_only", is_flag=True, default=False,
              help="Restrict to paper-mode trades")
@click.option("--live", "live_only", is_flag=True, default=False,
              help="Restrict to live-mode trades")
@click.option("--since", "since_str", default=None, metavar="YYYY-MM-DD",
              help="Only include trades on or after this UTC date")
@click.option("--no-fetch", is_flag=True, default=False,
              help="Skip Kalshi pre-fetch; use cached market results only")
def report(paper_only: bool, live_only: bool, since_str: str, no_fetch: bool):
    """Houston — panoramic snapshot of all bot activity (paper + live).

    Examples:
      ./run.sh report                              # all sessions, all modes
      ./run.sh report --paper                      # paper sessions only
      ./run.sh report --live                       # live sessions only
      ./run.sh report --paper --since 2026-05-15   # paper since date
      ./run.sh report --no-fetch                   # skip Kalshi API roundtrip
    """
    from datetime import datetime, timezone
    from btc15.cli.report import run_report

    if paper_only and live_only:
        console.print("[bright_red]--paper and --live are mutually exclusive[/bright_red]")
        return
    mode_filter = "paper" if paper_only else ("live" if live_only else None)
    since = None
    if since_str:
        try:
            since = datetime.strptime(since_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            console.print(f"[bright_red]--since must be YYYY-MM-DD, got '{since_str}'[/bright_red]")
            return
    run_report(mode_filter=mode_filter, since=since, no_fetch=no_fetch)


@cli.command()
@click.option("--min-samples", default=100, help="Minimum samples required to train")
def train(min_samples: int):
    """Train the ML model on collected trade data."""
    from btc15.models.ml_model import train_model, DATA_PATH
    if not DATA_PATH.exists():
        console.print(f"[yellow]No training data at {DATA_PATH}. Run bootstrap first:[/yellow]")
        console.print("  [cyan]./run.sh bootstrap[/cyan]")
        return
    import numpy as np
    data = np.load(DATA_PATH)
    console.print(f"Found {len(data['X']):,} training samples.")
    success = train_model(min_samples=min_samples)
    if success:
        console.print("[bright_green]Model trained successfully![/bright_green]")
    else:
        console.print("[red]Training failed — check logs.[/red]")


@cli.command()
@click.option("--months", default=6, help="Months of historical data to fetch (default: 6)")
def bootstrap(months: int):
    """Bootstrap ML model from years of historical BTC data (no live trading needed)."""
    import asyncio
    from btc15.models.bootstrap import main as bootstrap_main
    console.print(f"[bold]Bootstrapping ML model with {months} months of Kraken data...[/bold]")
    console.print("[dim]This fetches free historical data and trains LightGBM immediately.[/dim]\n")
    asyncio.run(bootstrap_main(months=months))


@cli.command()
@click.option("--port", default=8080, help="Port for web dashboard (default: 8080)")
@click.option("--trade", is_flag=True, default=False, help="Enable auto-trading")
@click.option("--live", is_flag=True, default=False, help="Disable paper mode (real money!)")
@click.option("--config", "config_path", default=None)
def web(port: int, trade: bool, live: bool, config_path: str):
    """Start bot with browser dashboard at http://localhost:PORT"""
    try:
        import fastapi, uvicorn  # noqa
    except ImportError:
        console.print("[red]Missing dependencies. Run:[/red]  pip install fastapi uvicorn")
        return

    from btc15.config import load_config
    cfg = load_config(Path(config_path) if config_path else None)
    setup_logging(cfg.logging.level, cfg.logging.log_file)

    if trade:
        cfg.strategy.auto_trade = True
        console.print("[bold yellow]Auto-trading ENABLED[/bold yellow]")
    if live:
        cfg.strategy.paper_trade = False
        console.print("[bold bright_red]Paper mode DISABLED — REAL MONEY[/bold bright_red]")
        if not click.confirm("Are you sure you want to trade with real money?"):
            sys.exit(0)

    console.print(
        f"Starting web dashboard → [bold cyan]http://localhost:{port}[/bold cyan]"
    )
    asyncio.run(_run_web(cfg, port))


async def _run_web(cfg, port: int):
    import uvicorn
    from btc15.strategy.engine import StrategyEngine
    from btc15.web.server import create_app

    engine = StrategyEngine(cfg)
    app = create_app(engine)

    server = uvicorn.Server(
        uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    )
    try:
        await engine.start()
        await server.serve()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await engine.stop()


# ── v2 Phase 1: replay & analyze recorded sessions ───────────────────────────

@cli.group()
def replay():
    """Replay & analyze a recorded telemetry session."""
    pass


@replay.command("list")
@click.option("--config", "config_path", default=None)
def replay_list(config_path: str):
    """List recorded sessions under data/recordings/."""
    from btc15.config import load_config
    cfg = load_config(Path(config_path) if config_path else None)
    root = Path(cfg.recording.path)
    if not root.exists():
        console.print(f"[yellow]No recordings directory at {root}[/yellow]")
        return
    sessions = sorted(d for d in root.iterdir() if d.is_dir())
    if not sessions:
        console.print("[yellow]No sessions found[/yellow]")
        return
    table = Table(title="Recorded sessions", box=box.SIMPLE)
    table.add_column("session_id")
    table.add_column("mode")
    table.add_column("duration")
    table.add_column("K (kalshi)", justify="right")
    table.add_column("V (venue)", justify="right")
    table.add_column("D (decisions)", justify="right")
    for s in sessions:
        meta_path = s / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = __import__("json").loads(meta_path.read_text())
        except Exception:
            continue
        lines = meta.get("lines") or {}
        dur = meta.get("duration_sec")
        # Fallbacks for sessions that died ungracefully (SIGHUP, kill -9 etc.)
        # — the JSONL files are the truth, meta is just a summary that may
        # never have been finalized.
        finalized = bool(dur)
        if not finalized:
            try:
                end_ts = max(
                    (s / f).stat().st_mtime
                    for f in ("kalshi_frames.jsonl", "venue_ticks.jsonl", "decisions.jsonl")
                    if (s / f).exists()
                )
                dur = end_ts - meta.get("start_ts", end_ts)
            except Exception:
                dur = None
            for name, fname in (
                ("kalshi", "kalshi_frames.jsonl"),
                ("venue", "venue_ticks.jsonl"),
                ("decisions", "decisions.jsonl"),
            ):
                p = s / fname
                if name not in lines and p.exists():
                    try:
                        lines[name] = sum(1 for _ in open(p))
                    except Exception:
                        pass
        dur_str = (
            f"{dur/60:.1f}m" if dur and finalized
            else f"{dur/60:.1f}m*" if dur
            else "running"
        )
        table.add_row(
            s.name,
            meta.get("mode", "?"),
            dur_str,
            str(lines.get("kalshi", "-")),
            str(lines.get("venue", "-")),
            str(lines.get("decisions", "-")),
        )
    # The * suffix on duration marks sessions whose meta.json was never
    # finalized (process killed ungracefully) — stats are derived from file
    # mtimes/line-counts, not the recorder's own bookkeeping.
    console.print(table)


@replay.command("convert")
@click.argument("session_id")
@click.option("--config", "config_path", default=None)
def replay_convert(session_id: str, config_path: str):
    """Reconstruct Kalshi books from raw frames + build index_grid.jsonl."""
    from btc15.config import load_config
    from btc15.recording.replay import cmd_convert
    cfg = load_config(Path(config_path) if config_path else None)
    setup_logging("INFO", cfg.logging.log_file)
    summary = cmd_convert(session_id, Path(cfg.recording.path))
    import json as _json
    console.print(_json.dumps(summary, indent=2))


@replay.command("grid")
@click.argument("session_id")
@click.option("--config", "config_path", default=None)
@click.option("--interval", default=1.0, help="Grid cadence in seconds (default 1.0).")
@click.option("--staleness", default=5.0, help="Max age (sec) before a venue tick is treated as stale.")
@click.option("--n-min", default=2, help="Minimum healthy venues required to emit a mid.")
@click.option("--k-mad", default=3.0, help="MAD outlier rejection threshold.")
def replay_grid(session_id: str, config_path: str, interval: float,
                staleness: float, n_min: int, k_mad: float):
    """Phase 2: Build BRTI reconstruction grid + print stability report."""
    from btc15.config import load_config
    from btc15.recording.replay import build_index_grid
    cfg = load_config(Path(config_path) if config_path else None)
    setup_logging("INFO", cfg.logging.log_file)

    session_dir = Path(cfg.recording.path) / session_id
    if not session_dir.exists():
        console.print(f"[red]Session not found: {session_dir}[/red]")
        return

    grid = build_index_grid(
        session_dir,
        interval_sec=interval,
        staleness_sec=staleness,
        n_min=n_min,
        k_mad=k_mad,
    )
    import json as _json
    report = _json.loads((session_dir / "stability_report.json").read_text())

    table = Table(title=f"BRTI stability — {session_id}", box=box.SIMPLE)
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("grid rows", f"{report['n_rows']:,}")
    table.add_row("healthy rows", f"{report['n_healthy']:,}")
    table.add_row("health %", f"{report['health_pct']:.2f}%")
    table.add_row("inter-venue spread p50", f"${report['spread_p50']:.2f}")
    table.add_row("inter-venue spread p95", f"${report['spread_p95']:.2f}")
    table.add_row("inter-venue spread max", f"${report['spread_max']:.2f}")
    table.add_row("outlier-flag events", f"{report['n_outlier_events']:,}")
    console.print(table)

    if report["venue_uptime_pct"]:
        ut = Table(title="Venue uptime", box=box.SIMPLE)
        ut.add_column("venue")
        ut.add_column("uptime %", justify="right")
        for v, pct in sorted(report["venue_uptime_pct"].items(), key=lambda x: -x[1]):
            ut.add_row(v, f"{pct:.1f}%")
        console.print(ut)

    if report["reason_breakdown"]:
        rb = Table(title="Reason breakdown", box=box.SIMPLE)
        rb.add_column("reason")
        rb.add_column("count", justify="right")
        rb.add_column("%", justify="right")
        total = report["n_rows"]
        for reason, count in sorted(report["reason_breakdown"].items(), key=lambda x: -x[1]):
            rb.add_row(reason, f"{count:,}", f"{count/total*100:.1f}%")
        console.print(rb)

    console.print(f"\n[dim]grid → {session_dir}/index_grid.jsonl  ({len(grid):,} rows)[/dim]")
    console.print(f"[dim]report → {session_dir}/stability_report.json[/dim]")


@replay.command("enrich")
@click.argument("session_id")
@click.option("--config", "config_path", default=None)
@click.option(
    "--cache",
    default="data/market_results_cache.json",
    help="Settlement results cache to update.",
)
def replay_enrich(session_id: str, config_path: str, cache: str):
    """Fetch Kalshi settlement results for every ticker in decisions.jsonl
    that isn't already finalized in the cache (counterfactual prep)."""
    from btc15.config import load_config
    from btc15.recording.replay import cmd_enrich_results
    cfg = load_config(Path(config_path) if config_path else None)
    setup_logging("INFO", cfg.logging.log_file)
    summary = asyncio.run(cmd_enrich_results(
        session_id, Path(cfg.recording.path), Path(cache), cfg,
    ))
    import json as _json
    console.print(_json.dumps(summary, indent=2))


@replay.command("diagnose")
@click.argument("session_id", required=False)
@click.option("--config", "config_path", default=None)
@click.option("--min-edge", default=0.10, help="Min edge gate threshold")
def replay_diagnose(session_id: str, config_path: str, min_edge: float):
    """Phase 3 step 5 diagnostic: which gate is killing each brain's signals?

    Walks the decision_log, replays the entry-gate chain against each
    brain's prob/conf, and reports a kill-table showing where each gate
    catches each brain. Tells you whether the production brain's lack of
    fires is by design (selectivity) or by misalignment (gate calibrated
    to the other brain's distribution).
    """
    from btc15.config import load_config
    from btc15.recording.gate_trace import trace_session, GATE_ORDER
    cfg = load_config(Path(config_path) if config_path else None)
    setup_logging("INFO", cfg.logging.log_file)

    recordings_root = Path(cfg.recording.path)
    if session_id:
        session_dir = recordings_root / session_id
        if not session_dir.exists():
            console.print(f"[red]Session not found: {session_dir}[/red]")
            return
    else:
        sessions = sorted(d for d in recordings_root.iterdir() if d.is_dir())
        if not sessions:
            console.print("[yellow]No sessions found[/yellow]")
            return
        session_dir = sessions[-1]

    # Read the live config so the diagnostic reflects what's ACTUALLY
    # gating in production — not the hardcoded DIR-era defaults. Honors
    # the per-phase confidence floors + entry-price bands as configured.
    cfg_min_conf = {
        k: float(v) for k, v in (cfg.trader.min_confidence_by_phase or {}).items()
    } or None
    cfg_entry_band = {}
    for k, v in (cfg.trader.entry_price_by_phase or {}).items():
        if isinstance(v, dict):
            cfg_entry_band[k] = (v.get("min", 10), v.get("max", 95))
        else:
            cfg_entry_band[k] = v
    cfg_entry_band = cfg_entry_band or None

    dir_trace, fv_trace, action_counts = trace_session(
        session_dir,
        min_edge=cfg.trader.min_edge if cfg.trader.min_edge != 0.10 else min_edge,
        min_conf_by_phase=cfg_min_conf,
        entry_price_by_phase=cfg_entry_band,
    )

    # ── Headline
    console.print(
        f"\n[bold]Session:[/bold] {session_dir.name}  "
        f"[dim]·  production_brain={cfg.strategy.production_brain}  "
        f"·  min_edge={cfg.trader.min_edge}  "
        f"·  min_conf={cfg_min_conf}[/dim]"
    )
    console.print(
        f"[dim]entry_price_by_phase={cfg_entry_band}  "
        f"·  entry_suppression_enabled={cfg.trader.entry_suppression_enabled}[/dim]"
    )

    # ── Actual recorded actions
    act_t = Table(title="Recorded decision actions (what actually happened)", box=box.SIMPLE)
    act_t.add_column("action")
    act_t.add_column("n", justify="right")
    act_t.add_column("%", justify="right")
    total_actions = sum(action_counts.values())
    fires = sum(v for k, v in action_counts.items() if k != "none")
    for action in sorted(action_counts.keys(), key=lambda k: -action_counts[k]):
        v = action_counts[action]
        act_t.add_row(action, f"{v:,}", f"{v/total_actions*100:.1f}%" if total_actions else "—")
    console.print(act_t)
    console.print(f"[dim]Total actual fires (action != none): {fires:,}[/dim]")

    # ── Per-brain gate trace
    gate_t = Table(title="Gate-by-gate filter (where each brain gets killed)", box=box.SIMPLE)
    gate_t.add_column("gate")
    gate_t.add_column("DIR n", justify="right")
    gate_t.add_column("DIR %", justify="right")
    gate_t.add_column("FV n", justify="right")
    gate_t.add_column("FV %", justify="right")
    for gate in GATE_ORDER:
        d_n = dir_trace.by_gate.get(gate, 0)
        f_n = fv_trace.by_gate.get(gate, 0)
        if d_n == 0 and f_n == 0:
            continue
        d_pct = d_n / dir_trace.n_rows_evaluated * 100 if dir_trace.n_rows_evaluated else 0
        f_pct = f_n / fv_trace.n_rows_evaluated * 100 if fv_trace.n_rows_evaluated else 0
        style = "bold bright_green" if gate == "WOULD_FIRE" else ""
        gate_t.add_row(
            f"[{style}]{gate}[/{style}]" if style else gate,
            f"{d_n:,}",
            f"{d_pct:.1f}%",
            f"{f_n:,}",
            f"{f_pct:.1f}%",
        )
    console.print(gate_t)
    console.print(
        f"[dim]Rows evaluated: DIR {dir_trace.n_rows_evaluated:,}  "
        f"FV {fv_trace.n_rows_evaluated:,}  "
        f"(skipped/no-prob: DIR {dir_trace.n_skipped_no_prob:,}  "
        f"FV {fv_trace.n_skipped_no_prob:,})[/dim]"
    )

    # ── Would-fire examples (if any)
    examples_brain = "fv" if cfg.strategy.production_brain == "fair_value" else "dir"
    examples = (fv_trace if examples_brain == "fv" else dir_trace).would_fire_examples
    if examples:
        ex_t = Table(
            title=f"Sample {examples_brain.upper()} would-fire rows (first 10)",
            box=box.SIMPLE,
        )
        ex_t.add_column("ticker")
        ex_t.add_column("secs", justify="right")
        ex_t.add_column("prob_yes", justify="right")
        ex_t.add_column("conf", justify="right")
        ex_t.add_column("mid", justify="right")
        for ex in examples:
            ex_t.add_row(
                str(ex.get("ticker", ""))[-15:],
                f"{ex.get('secs', 0):.0f}",
                f"{ex.get('prob_yes', 0):.3f}",
                f"{ex.get('conf', 0):.3f}",
                f"{ex.get('kalshi_mid', 0):.0f}",
            )
        console.print(ex_t)


@replay.command("pnl")
@click.argument("session_id", required=False)
@click.option("--config", "config_path", default=None)
@click.option(
    "--results-cache",
    default="data/market_results_cache.json",
    help="Path to settled-market results cache",
)
@click.option(
    "--trades-csv",
    default="logs/trades.csv",
    help="Path to the actual trade log (for DIR's realized P&L)",
)
@click.option("--contracts", default=1, help="Contracts per simulated trade")
@click.option("--min-edge", default=0.10, help="Min edge to fire a simulated entry")
def replay_pnl(session_id: str, config_path: str, results_cache: str,
               trades_csv: str, contracts: int, min_edge: float):
    """Phase 3 step 4.5: counterfactual P&L (DIR vs FV, hold-to-settle).

    Brier validates calibration; P&L validates execution outcome. This
    replays the recorded decision log through the production entry gates
    using each brain's prob_yes + confidence, assumes hold-to-settlement
    on every fired entry, and compares dollars made.
    """
    from btc15.config import load_config
    from btc15.recording.shadow_pnl import analyze_pnl
    cfg = load_config(Path(config_path) if config_path else None)
    setup_logging("INFO", cfg.logging.log_file)

    recordings_root = Path(cfg.recording.path)
    if session_id:
        session_dir = recordings_root / session_id
        if not session_dir.exists():
            console.print(f"[red]Session not found: {session_dir}[/red]")
            return
    else:
        sessions = sorted(d for d in recordings_root.iterdir() if d.is_dir())
        if not sessions:
            console.print("[yellow]No sessions found[/yellow]")
            return
        session_dir = sessions[-1]

    result = analyze_pnl(
        session_dir, Path(results_cache), Path(trades_csv),
        contracts=contracts, min_edge=min_edge,
    )

    # ── Headline table
    summary = Table(title=f"P&L — {result.session_id}  [dim](contracts={contracts}, min_edge={min_edge:.2f})[/dim]", box=box.SIMPLE)
    summary.add_column("scenario")
    summary.add_column("n trades", justify="right")
    summary.add_column("win rate", justify="right")
    summary.add_column("total P&L ($)", justify="right")

    def _pnl_color(v):
        return "bright_green" if v > 0 else "bright_red" if v < 0 else "white"

    if result.dir_realized_pnl_dollars is not None and result.dir_realized_n_round_trips:
        c = _pnl_color(result.dir_realized_pnl_dollars)
        summary.add_row(
            "DIR realized (round-trips)",
            f"{result.dir_realized_n_round_trips:,}",
            "[dim]—[/dim]",
            f"[{c}]${result.dir_realized_pnl_dollars:+,.2f}[/{c}]",
        )
    if result.dir_entries_held_pnl_dollars is not None and result.dir_entries_held_pnl_dollars != 0:
        c = _pnl_color(result.dir_entries_held_pnl_dollars)
        summary.add_row(
            "DIR entries held-to-settle",
            f"{result.dir_realized_n_round_trips:,}",
            "[dim]—[/dim]",
            f"[{c}]${result.dir_entries_held_pnl_dollars:+,.2f}[/{c}]",
        )

    for sim, label in [
        (result.dir_simulated, "DIR hold-to-settle (sim)"),
        (result.fv_simulated, "FV hold-to-settle (sim)"),
    ]:
        c = _pnl_color(sim.total_pnl_dollars)
        wr = sim.win_rate
        summary.add_row(
            label,
            f"{sim.n_trades:,}",
            f"{wr:.1%}" if wr is not None else "[dim]—[/dim]",
            f"[{c}]${sim.total_pnl_dollars:+,.2f}[/{c}]",
        )
    console.print(summary)

    console.print(
        f"[dim]Decision rows: {result.n_decision_rows:,}  ·  "
        f"settled rows: {result.n_settled_rows:,}  ·  "
        f"settled tickers: {result.settled_tickers:,}[/dim]"
    )

    # ── Per-phase breakdown
    phase_t = Table(title="Simulated P&L by phase", box=box.SIMPLE)
    phase_t.add_column("phase")
    phase_t.add_column("DIR n", justify="right")
    phase_t.add_column("DIR P&L", justify="right")
    phase_t.add_column("DIR WR", justify="right")
    phase_t.add_column("FV n", justify="right")
    phase_t.add_column("FV P&L", justify="right")
    phase_t.add_column("FV WR", justify="right")
    dir_phase = result.dir_simulated.per_phase()
    fv_phase = result.fv_simulated.per_phase()
    for phase in ("early", "mid", "prime", "late"):
        d = dir_phase.get(phase, {"n": 0, "pnl_cents": 0, "wins": 0})
        f = fv_phase.get(phase, {"n": 0, "pnl_cents": 0, "wins": 0})
        if d["n"] == 0 and f["n"] == 0:
            continue
        d_pnl = d["pnl_cents"] / 100.0
        f_pnl = f["pnl_cents"] / 100.0
        d_c = _pnl_color(d_pnl)
        f_c = _pnl_color(f_pnl)
        phase_t.add_row(
            phase,
            f"{d['n']:,}",
            f"[{d_c}]${d_pnl:+,.2f}[/{d_c}]",
            f"{d['wins']/d['n']:.1%}" if d["n"] else "—",
            f"{f['n']:,}",
            f"[{f_c}]${f_pnl:+,.2f}[/{f_c}]",
            f"{f['wins']/f['n']:.1%}" if f["n"] else "—",
        )
    console.print(phase_t)

    # ── Disagreement P&L
    d = result.disagreement_pnl_cents
    da_t = Table(title="Disagreement-zone P&L (who's making money on signals the other missed?)", box=box.SIMPLE)
    da_t.add_column("category")
    da_t.add_column("n trades", justify="right")
    da_t.add_column("DIR P&L", justify="right")
    da_t.add_column("FV P&L", justify="right")
    if d.get("both_n"):
        both_dir = d["both_dir_pnl_cents"] / 100.0
        both_fv = d["both_fv_pnl_cents"] / 100.0
        da_t.add_row(
            "Both brains entered",
            f"{d['both_n']:,}",
            f"[{_pnl_color(both_dir)}]${both_dir:+,.2f}[/{_pnl_color(both_dir)}]",
            f"[{_pnl_color(both_fv)}]${both_fv:+,.2f}[/{_pnl_color(both_fv)}]",
        )
    if d.get("dir_only_n"):
        do = d["dir_only_pnl_cents"] / 100.0
        da_t.add_row(
            "DIR only (FV skipped)",
            f"{d['dir_only_n']:,}",
            f"[{_pnl_color(do)}]${do:+,.2f}[/{_pnl_color(do)}]",
            "[dim]—[/dim]",
        )
    if d.get("fv_only_n"):
        fo = d["fv_only_pnl_cents"] / 100.0
        da_t.add_row(
            "FV only (DIR skipped)",
            f"{d['fv_only_n']:,}",
            "[dim]—[/dim]",
            f"[{_pnl_color(fo)}]${fo:+,.2f}[/{_pnl_color(fo)}]",
        )
    console.print(da_t)


@replay.command("brier")
@click.argument("session_id", required=False)
@click.option("--all", "all_sessions", is_flag=True, default=False,
              help="Aggregate across every session under data/recordings/")
@click.option("--config", "config_path", default=None)
@click.option(
    "--results-cache",
    default="data/market_results_cache.json",
    help="Path to settled-market results cache",
)
def replay_brier(session_id: str, all_sessions: bool, config_path: str, results_cache: str):
    """Phase 3: DIR vs FV Brier comparison on settled markets.

    Uses the dual-brain decision rows (Phase 3 step 3+) joined to the
    market settlement cache. Lower Brier = better calibrated. Baseline
    is 0.25 (constant-50% predictor); the legacy ensemble's audit
    measured 0.283.
    """
    from btc15.config import load_config
    from btc15.recording.shadow_analysis import (
        analyze_session, analyze_all_sessions, merge_results,
    )
    cfg = load_config(Path(config_path) if config_path else None)
    setup_logging("INFO", cfg.logging.log_file)

    recordings_root = Path(cfg.recording.path)
    cache_path = Path(results_cache)

    if all_sessions:
        per_session = analyze_all_sessions(recordings_root, cache_path)
        result = merge_results(per_session)
        sub_count = len(per_session)
    elif session_id:
        session_dir = recordings_root / session_id
        if not session_dir.exists():
            console.print(f"[red]Session not found: {session_dir}[/red]")
            return
        result = analyze_session(session_dir, cache_path)
        sub_count = None
    else:
        sessions = sorted(d for d in recordings_root.iterdir() if d.is_dir())
        if not sessions:
            console.print("[yellow]No sessions found[/yellow]")
            return
        result = analyze_session(sessions[-1], cache_path)
        sub_count = None

    # ── Summary table
    title = result.session_id
    if sub_count is not None:
        title = f"{title}  [dim]({sub_count} sessions)[/dim]"
    summary = Table(title=f"DIR vs FV — {title}", box=box.SIMPLE)
    summary.add_column("metric")
    summary.add_column("DIR", justify="right")
    summary.add_column("FV", justify="right")
    summary.add_column("baseline", justify="right", style="dim")
    db = result.dir_scores.mean_brier
    fb = result.fv_scores.mean_brier
    da = result.dir_scores.directional_accuracy
    fa = result.fv_scores.directional_accuracy
    summary.add_row(
        "Brier (lower = better)",
        f"{db:.4f}" if db is not None else "—",
        f"{fb:.4f}" if fb is not None else "—",
        "0.2500",
    )
    summary.add_row(
        "directional accuracy",
        f"{da:.1%}" if da is not None else "—",
        f"{fa:.1%}" if fa is not None else "—",
        "50.0%",
    )
    summary.add_row(
        "n predictions",
        f"{result.dir_scores.n_rows:,}",
        f"{result.fv_scores.n_rows:,}",
        "",
    )
    console.print(summary)
    console.print(
        f"[dim]Settled markets: {result.settled_tickers}  ·  "
        f"settled rows: {result.n_settled_rows:,} / {result.n_total_rows:,} total  ·  "
        f"FV-eligible rows: {result.n_with_fv:,}[/dim]"
    )

    # ── Per-phase breakdown
    if result.dir_scores.per_phase:
        phase_t = Table(title="Brier by phase", box=box.SIMPLE)
        phase_t.add_column("phase")
        phase_t.add_column("n", justify="right")
        phase_t.add_column("DIR Brier", justify="right")
        phase_t.add_column("FV Brier", justify="right")
        for phase in ("early", "mid", "prime", "late"):
            d_cell = result.dir_scores.per_phase.get(phase, {"n": 0, "brier": 0.0})
            f_cell = result.fv_scores.per_phase.get(phase, {"n": 0, "brier": 0.0})
            if d_cell["n"] == 0 and f_cell["n"] == 0:
                continue
            phase_t.add_row(
                phase,
                f"{d_cell['n']:,}",
                f"{d_cell['brier']/d_cell['n']:.4f}" if d_cell["n"] else "—",
                f"{f_cell['brier']/f_cell['n']:.4f}" if f_cell["n"] else "—",
            )
        console.print(phase_t)

    # ── Confidence-band breakdown
    if result.dir_scores.per_conf_band:
        cb_t = Table(title="By confidence band (does higher confidence predict better?)", box=box.SIMPLE)
        cb_t.add_column("conf")
        cb_t.add_column("DIR n", justify="right")
        cb_t.add_column("DIR Brier", justify="right")
        cb_t.add_column("DIR WR", justify="right")
        cb_t.add_column("FV n", justify="right")
        cb_t.add_column("FV Brier", justify="right")
        cb_t.add_column("FV WR", justify="right")
        for band in ("0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"):
            d_cell = result.dir_scores.per_conf_band.get(band, {"n": 0, "brier": 0.0, "won": 0})
            f_cell = result.fv_scores.per_conf_band.get(band, {"n": 0, "brier": 0.0, "won": 0})
            if d_cell["n"] == 0 and f_cell["n"] == 0:
                continue
            cb_t.add_row(
                band,
                f"{d_cell['n']:,}",
                f"{d_cell['brier']/d_cell['n']:.4f}" if d_cell["n"] else "—",
                f"{d_cell['won']/d_cell['n']:.1%}" if d_cell["n"] else "—",
                f"{f_cell['n']:,}",
                f"{f_cell['brier']/f_cell['n']:.4f}" if f_cell["n"] else "—",
                f"{f_cell['won']/f_cell['n']:.1%}" if f_cell["n"] else "—",
            )
        console.print(cb_t)

    # ── Agreement matrix
    if result.agreement:
        ag_t = Table(title="Agreement / disagreement (who's right?)", box=box.SIMPLE)
        ag_t.add_column("case")
        ag_t.add_column("n", justify="right")
        ag_t.add_column("DIR correct", justify="right")
        ag_t.add_column("FV correct", justify="right")
        ordered = ["agree_yes", "agree_no", "dir_yes_fv_no", "dir_no_fv_yes"]
        for key in ordered:
            cell = result.agreement.get(key)
            if not cell or cell["n"] == 0:
                continue
            ag_t.add_row(
                key,
                f"{cell['n']:,}",
                f"{cell['dir_correct']/cell['n']:.1%}",
                f"{cell['fv_correct']/cell['n']:.1%}",
            )
        console.print(ag_t)


@replay.command("analyze")
@click.argument("session_id")
@click.option("--config", "config_path", default=None)
@click.option(
    "--results-cache",
    default="data/market_results_cache.json",
    help="Path to settled-market results cache (for counterfactual P&L)",
)
def replay_analyze(session_id: str, config_path: str, results_cache: str):
    """Run the two Phase-1 validation analyses on a recorded session."""
    from btc15.config import load_config
    from btc15.recording.replay import cmd_analyze
    cfg = load_config(Path(config_path) if config_path else None)
    setup_logging("INFO", cfg.logging.log_file)
    summary = cmd_analyze(
        session_id,
        Path(cfg.recording.path),
        Path(results_cache),
    )
    import json as _json
    console.print(_json.dumps(summary, indent=2))
