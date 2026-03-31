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
        console.print("\n[dim]Bot stopped.[/dim]")
