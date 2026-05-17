"""
Main strategy engine — the brain of the bot.

Lifecycle:
  1. On start: connect feeds + Kalshi client, seed historical data
  2. Every SCAN_INTERVAL seconds: fetch open KXBTC markets
  3. For each market: run ensemble model to compute P(YES) and edge
  4. Dispatch to AutoTrader.evaluate() — returns list of Actions
  5. Execute each Action (buy/sell/cancel/amend/batch_buy)
  6. Monitor open positions for settlement

Key architectural change from v1:
  - Single AutoTrader replaces Sniper/Scalper/Arb persona trio
  - Single position tracking (autotrader.positions) — no more dual-tracking
  - Balance cached once per scan cycle (not per market)
  - GTC post_only entries for early window (0% maker fee)
  - IOC escalation for prime window or stale GTC orders
  - Simplified exits: hold to settlement unless reversal or loss-cut
"""
from __future__ import annotations

import asyncio
import csv
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from btc15.config import AppConfig
from btc15.feeds.aggregator import PriceAggregator
from btc15.kalshi.client import KalshiClient, KalshiAPIError
from btc15.kalshi.models import (
    Market, Order, OrderType, OrderStatus, Side, MarketStatus,
    TimeInForce, SelfTradePrevention,
)
from btc15.kalshi.ws_client import KalshiWebSocket, MarketDataCache
from btc15.models.ensemble import EnsembleModel
from btc15.models.ml_model import collect_sample
from btc15.risk.manager import RiskManager
from btc15.strategy.personas import Action, AutoTrader


log = logging.getLogger(__name__)

SCAN_INTERVAL = 3          # seconds between market scans
OB_REFRESH_INTERVAL = 4    # full orderbook REST snapshot refresh (fills WS gaps)
POSITION_CHECK = 10        # seconds between position status checks


# ── In-memory log handler ─────────────────────────────────────────────────────

class _DashboardLogHandler(logging.Handler):
    """Captures WARNING+ and key INFO lines into state["event_log"]."""

    _KEYWORDS = (
        "RISK BLOCK", "SIGNAL", "EXECUTING", "EXIT", "SETTLED",
        "HALTED", "ERROR", "WARNING", "disconnected", "PURE ARB",
        "MM QUOTE", "BATCH", "conf=", "edge=",
        "SIZER", "EV=", "Cost=", "exceeds", "below min",
        "exposure", "win rate", "HALT", "BUY:", "SELL:", "flip",
        "[SKIP]", "GTC", "IOC", "ESCALATE", "REVERSAL", "LOSS CUT",
        "RECONCILE", "ORPHAN", "FILL",
    )
    _DEDUP_FRAGMENTS = ("[SKIP]",)

    def __init__(self, state: dict):
        super().__init__()
        self._state = state
        self._seen: set[str] = set()

    def emit(self, record: logging.LogRecord):
        try:
            raw = record.getMessage()
            lvl = record.levelno
            if lvl < logging.WARNING:
                if not any(kw in raw for kw in self._KEYWORDS):
                    return
            for frag in self._DEDUP_FRAGMENTS:
                if frag in raw:
                    key = frag + raw[:60]
                    if key in self._seen:
                        return
                    self._seen.add(key)
                    if len(self._seen) > 200:
                        self._seen.clear()
                    break
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            entry = {"ts": ts, "level": record.levelname, "msg": raw}
            log_list = self._state.setdefault("event_log", [])
            log_list.append(entry)
            self._state["event_log"] = log_list[-100:]
        except Exception:
            pass


class StrategyEngine:
    """
    Async strategy engine. Run with `await engine.run()`.
    Exposes state dict for the CLI dashboard.
    """

    def __init__(self, config: AppConfig):
        self.cfg = config
        self.risk = RiskManager(config.risk)
        self.price_feed = PriceAggregator(
            bar_interval_sec=config.feeds.bar_interval_sec,
            lookback_bars=config.feeds.lookback_bars,
            coinbase_rest_url=config.feeds.coinbase_rest_url,
        )
        self.ensemble = EnsembleModel(
            weights=config.models.ensemble_weights,
            config=config.models,
        )
        self._kalshi: Optional[KalshiClient] = None
        self._market_cache = MarketDataCache()
        self._ws: Optional[KalshiWebSocket] = None

        # Unified auto-trader (replaces Sniper/Scalper/Arb)
        self.autotrader = AutoTrader(config.trader)

        # Shared state readable by CLI dashboard
        self.running = False
        self.state: dict = {
            "status": "idle",
            "current_price": 0.0,
            "feed_age_sec": 0.0,
            "open_markets": [],
            "signals": {},
            "open_positions": [],
            "recent_trades": [],
            "balance": None,
            "session_start_balance": None,
            "risk": {},
            "trader": {},
            "last_scan": None,
            "paper_mode": config.strategy.paper_trade,
            "auto_trade": config.strategy.auto_trade,
            "session_start": datetime.now(timezone.utc).isoformat(),
            "pnl_history": [],
            "event_log": [],
        }

        _now = datetime.now(timezone.utc)
        self._session_label = _now.strftime("%d%b%H:%M").upper()

        self._dash_handler = _DashboardLogHandler(self.state)
        self._dash_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self._dash_handler)

        self._open_orders: dict[str, Order] = {}
        self._watched_markets: dict[str, Market] = {}
        self._position_features: dict[str, list] = {}  # ticker → ML features at entry
        self._tasks: list[asyncio.Task] = []
        self._log_file = Path(config.logging.trade_log_file)
        self._ensure_trade_log()

        # Balance cache — refreshed once per scan cycle, not per market
        self._cached_balance_usd: float = config.risk.max_trade_usd * 20
        self._balance_cache_ts: float = 0.0

        # Reconcile guards
        # _reconcile_lock: prevents two concurrent reconcile tasks from both adopting the same gap
        # _settled_tickers: suppresses reconcile adoption for 60s after settlement to avoid
        #   re-adopting positions still visible in the Kalshi API due to settlement lag
        self._reconcile_lock = asyncio.Lock()
        self._settled_tickers: dict[str, float] = {}   # ticker → epoch when settled
        self._recently_exited: dict[str, float] = {}   # ticker → epoch when we last exited (exit lag guard)
        # _recently_placed: ticker → epoch when we last sent place_order. Suppresses
        # reconciler for ~15s after each entry to close the race where:
        #   1. Bot awaits place_order (IOC) — yields to event loop
        #   2. Order fills on Kalshi within ms
        #   3. Reconciler runs in this window — sees gap, adopts as ghost
        #   4. place_order returns, bot calls record_fill — same fill recorded twice
        # GTC orders are already covered via resting_orders; this guards IOC and
        # the GTC→IOC escalation path that produced the duplicate "reconciled_gap"
        # rows in the live trade log.
        self._recently_placed: dict[str, float] = {}
        # Tracks how many _check_settlements cycles a ticker has been past close_time
        # without Kalshi publishing the official result yet (API lag is normal 5–30s).
        self._settlement_pending_count: dict[str, int] = {}

    # ── Public interface ─────────────────────────────────────────────────────

    async def start(self):
        log.info("Strategy engine starting...")
        self.state["status"] = "starting"
        self.running = True

        await self.price_feed.start()

        self._kalshi = KalshiClient(self.cfg.kalshi)
        await self._kalshi.connect()

        self._ws = KalshiWebSocket(
            config=self.cfg.kalshi,
            auth_header_factory=self._kalshi.ws_auth_headers if self._kalshi._using_rsa else None,
        )
        self._ws.on("ticker", self._market_cache.handle_ticker)
        self._ws.on("orderbook_delta", self._market_cache.handle_orderbook_delta)
        self._ws.on("orderbook_snapshot", self._market_cache.handle_orderbook_snapshot)
        self._ws.on("fill", self._handle_fill)
        self._ws.on_reconnect = self._on_ws_reconnect
        # Lets the cache trigger a REST refresh when data goes stale between ticks.
        self._market_cache.rest_refresh = self._refresh_ticker_orderbook
        self._tasks.append(asyncio.create_task(self._ws.run(), name="kalshi-ws"))

        try:
            _start_bal = await self._kalshi.get_balance()
            self.state["session_start_balance"] = round(_start_bal.available_usd, 2)
            self._cached_balance_usd = _start_bal.available_usd
            self._balance_cache_ts = time.time()
            log.info(f"Session start balance: ${self.state['session_start_balance']:.2f}")
        except Exception:
            pass

        # Wait for WS handshake to complete before starting loops.
        # 1s is enough — RSA auth is synchronous during HTTP upgrade.
        await asyncio.sleep(1)

        self._tasks.append(asyncio.create_task(self._scan_loop(), name="scan-loop"))
        self._tasks.append(asyncio.create_task(self._position_loop(), name="position-loop"))
        self._tasks.append(asyncio.create_task(self._state_updater(), name="state-updater"))
        self._tasks.append(asyncio.create_task(self._orderbook_refresh_loop(), name="ob-refresh"))

        self.state["status"] = "running"
        log.info("Strategy engine running")

    async def stop(self):
        log.info("Strategy engine stopping...")
        self.running = False
        self.state["status"] = "stopping"

        if not self.cfg.strategy.paper_trade and self._kalshi:
            try:
                n = await self._kalshi.cancel_all_orders()
                if n:
                    log.info(f"Cancelled {n} open orders on shutdown")
            except Exception as e:
                log.warning(f"Error cancelling orders on shutdown: {e}")

        for t in self._tasks:
            t.cancel()

        await self.price_feed.stop()
        if self._kalshi:
            await self._kalshi.close()

        self.state["status"] = "stopped"
        log.info("Strategy engine stopped")

    # ── Manual trading interface ──────────────────────────────────────────────

    async def manual_trade(self, ticker: str, side: str, amount_usd: float) -> str:
        """Execute a manual trade from the CLI. Returns result message."""
        if not self._kalshi:
            return "Not connected to Kalshi"
        try:
            market = await self._kalshi.get_market(ticker)
            if market.status not in (MarketStatus.OPEN, MarketStatus.ACTIVE):
                return f"Market {ticker} is not open"

            ob = await self._kalshi.get_orderbook(ticker)
            side_enum = Side(side.lower())

            if side_enum == Side.YES:
                price_cents = int(ob.best_yes_ask or market.yes_ask)
            else:
                price_cents = int(100 - (ob.best_yes_bid or market.yes_bid))

            price_cents = max(1, min(99, price_cents))
            contracts = int(amount_usd / (price_cents / 100))
            if contracts <= 0:
                return f"Amount ${amount_usd:.2f} too small for price {price_cents}¢"

            # Risk check
            allowed, reason = self.risk.check_trade(
                ticker=ticker, side=side_enum.value,
                contracts=contracts, price_cents=price_cents,
                bankroll_usd=self._cached_balance_usd,
            )
            if not allowed:
                return f"Risk block: {reason}"

            trade_id = f"T{uuid.uuid4().hex[:8].upper()}"
            paper = self.cfg.strategy.paper_trade
            label = "[PAPER]" if paper else "[LIVE]"
            log.info(
                f"{label} MANUAL: {ticker} {side_enum.value.upper()} "
                f"×{contracts} @ {price_cents}¢"
            )

            if not paper:
                self._recently_placed[ticker] = time.time()
                order = await self._kalshi.place_order(
                    ticker=ticker, side=side_enum, contracts=contracts,
                    price_cents=price_cents, order_type=OrderType.LIMIT,
                    time_in_force=TimeInForce.IOC,
                    client_order_id=f"btc15-manual-{uuid.uuid4().hex[:6]}",
                )
                filled = order.filled_count if order.filled_count > 0 else (
                    contracts if order.status == OrderStatus.EXECUTED else 0
                )
                if filled == 0:
                    return f"Manual order unfilled at {price_cents}¢ — market moved"
                contracts = filled

            self.autotrader.record_fill(ticker, side_enum.value, contracts, price_cents, trade_id)
            self.risk.record_open(ticker, side_enum.value, contracts, price_cents)
            self._log_trade(ticker, side_enum.value, contracts, price_cents, "manual", trade_id)
            self.state["recent_trades"].insert(0, {
                "ticker": ticker, "side": side_enum.value, "contracts": contracts,
                "price_cents": price_cents,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "source": "manual", "status": "open", "trade_id": trade_id,
                "session": self._session_label,
            })
            self.state["recent_trades"] = self.state["recent_trades"][:50]
            return f"{'Paper' if paper else 'Live'} trade: {ticker} {side.upper()} ×{contracts} @ {price_cents}¢"
        except Exception as e:
            return f"Error: {e}"

    # ── Background loops ──────────────────────────────────────────────────────

    async def _scan_loop(self):
        while self.running:
            try:
                await self._scan_markets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Scan loop error: {e}", exc_info=True)
            await asyncio.sleep(SCAN_INTERVAL)

    async def _on_ws_reconnect(self):
        """
        Called after every WS reconnect (connection #2+).
        Re-seeds the orderbook cache from REST immediately — the WS delta stream
        was interrupted so the cache may have drifted or be completely stale.
        Then runs the normal resting-order reconcile.
        """
        log.info("[WS RECONNECT] Re-seeding orderbook cache from REST...")
        if self._watched_markets and self._kalshi:
            tickers = list(self._watched_markets.keys())
            results = await asyncio.gather(
                *[self._kalshi.get_orderbook(t) for t in tickers],
                return_exceptions=True,
            )
            for ticker, result in zip(tickers, results):
                if not isinstance(result, Exception):
                    await self._market_cache.apply_snapshot(ticker, result)
            log.info(f"[WS RECONNECT] Orderbook cache re-seeded for {len(tickers)} markets")
        await self._reconcile_positions()

    async def _refresh_ticker_orderbook(self, ticker: str) -> None:
        """Single-ticker REST refresh, invoked by the cache when data is stale."""
        if not self._kalshi:
            return
        try:
            ob = await self._kalshi.get_orderbook(ticker)
            await self._market_cache.apply_snapshot(ticker, ob)
        except Exception as e:
            log.debug(f"[STALE CACHE] REST refresh failed for {ticker}: {e}")

    async def _orderbook_refresh_loop(self):
        """
        Periodically fetch full REST orderbook snapshots for all watched markets.
        Runs immediately on start (no initial sleep) so there's never a window
        where the cache is empty. Subsequent refreshes repair any WS delta drift.
        """
        while self.running:
            if not self._watched_markets or not self._kalshi:
                await asyncio.sleep(1)
                continue
            tickers = list(self._watched_markets.keys())
            try:
                results = await asyncio.gather(
                    *[self._kalshi.get_orderbook(t) for t in tickers],
                    return_exceptions=True,
                )
                refreshed = 0
                for ticker, result in zip(tickers, results):
                    if isinstance(result, Exception):
                        log.debug(f"[OB REFRESH] {ticker}: {result}")
                        continue
                    await self._market_cache.apply_snapshot(ticker, result)
                    refreshed += 1
                log.debug(f"[OB REFRESH] {refreshed}/{len(tickers)} markets updated")
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"[OB REFRESH] Loop error: {e}")
            await asyncio.sleep(OB_REFRESH_INTERVAL)

    async def _get_cached_balance(self) -> float:
        """
        Return cached balance. Refresh from API if cache is older than 5 seconds.
        Prevents N balance API calls per scan when watching N markets.
        """
        if time.time() - self._balance_cache_ts > 5.0 and self._kalshi:
            try:
                bal = await self._kalshi.get_balance()
                self._cached_balance_usd = bal.available_usd
                self._balance_cache_ts = time.time()
            except Exception:
                pass
        return self._cached_balance_usd

    async def _scan_markets(self):
        markets = await self._kalshi.get_markets(
            series_ticker=self.cfg.kalshi.series_ticker,
            status="open",
            limit=20,
        )

        new_tickers = [m.ticker for m in markets if m.ticker not in self._watched_markets]
        if new_tickers:
            await self._ws.subscribe(new_tickers, ["orderbook_delta", "ticker", "fill"])
            # Immediately seed the cache with REST snapshots for new tickers.
            # The WS will also deliver a snapshot on subscription but may take
            # a few seconds — without this the terminal shows "--" until the
            # first orderbook_delta or the 10s refresh loop fires.
            if self._kalshi:
                seed_results = await asyncio.gather(
                    *[self._kalshi.get_orderbook(t) for t in new_tickers],
                    return_exceptions=True,
                )
                for ticker, result in zip(new_tickers, seed_results):
                    if not isinstance(result, Exception):
                        await self._market_cache.apply_snapshot(ticker, result)
                        log.debug(f"[OB SEED] {ticker}: seeded from REST")

        self._watched_markets = {m.ticker: m for m in markets}
        self.state["last_scan"] = datetime.now(timezone.utc).isoformat()

        if not markets:
            return

        current_price = self.price_feed.current_price
        if not current_price:
            log.debug("Skipping scan — no price data yet")
            return
        annual_vol = self.price_feed.realized_vol()
        bars = self.price_feed.bars
        now_utc = datetime.now(timezone.utc)

        # Fetch balance once for the whole scan cycle
        bankroll = await self._get_cached_balance()

        # Parallel fresh orderbook fetch for markets with open positions
        position_tickers = [
            m.ticker for m in markets
            if m.ticker in self.autotrader.positions
        ]
        fresh_obs: dict[str, object] = {}
        if position_tickers:
            results = await asyncio.gather(
                *[self._kalshi.get_orderbook(t) for t in position_tickers],
                return_exceptions=True,
            )
            for ticker, result in zip(position_tickers, results):
                if not isinstance(result, Exception):
                    fresh_obs[ticker] = result

        signals_snapshot = {}
        markets_snapshot = []
        # Accumulate live WS prices keyed by ticker so we can refresh position
        # PnL display at the end of each scan — avoids waiting for the 10s
        # _position_loop and stops using stale REST market prices for mark-to-market.
        live_prices: dict[str, tuple[float, float]] = {}  # ticker → (yes_bid, yes_ask)

        for market in markets:
            # Compute seconds_remaining from local clock (not stale REST value)
            close_time = market.close_time
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            secs = max(0.0, (close_time - now_utc).total_seconds())

            tradeable = (
                self.cfg.strategy.min_seconds_remaining
                <= secs <=
                self.cfg.strategy.max_seconds_remaining
            )

            # Best prices: WS cache first, fall back to REST market snapshot
            ws_bid, ws_ask = await self._market_cache.get_best_prices(market.ticker)
            # Display prices: WS cache first, fall back to REST market snapshot.
            # These are shown in the terminal and fed to the model. Never zeroed out
            # so the terminal always shows the last known price rather than --.
            yes_bid = ws_bid if ws_bid is not None else market.yes_bid
            yes_ask = ws_ask if ws_ask is not None else market.yes_ask
            if market.ticker in fresh_obs:
                fresh_ob = fresh_obs[market.ticker]
                if fresh_ob.best_yes_bid is not None:
                    yes_bid = fresh_ob.best_yes_bid
                if fresh_ob.best_yes_ask is not None:
                    yes_ask = fresh_ob.best_yes_ask

            # Trading prices: used by AutoTrader exit/entry logic.
            # Mirror the display-path policy — prefer the fresh REST level when
            # it exists, otherwise fall back to the WS / market-snapshot price.
            # The old behavior (force 0 when REST side is empty) misread transient
            # thin-book snapshots as "contract is worthless" and fed phantom
            # -100% pnls into the exit logic, triggering emergency stops and
            # trapping positions in `settling=True` for the rest of the market.
            trade_bid = yes_bid
            trade_ask = yes_ask
            if market.ticker in fresh_obs:
                fresh_ob = fresh_obs[market.ticker]
                if fresh_ob.best_yes_bid is not None:
                    trade_bid = fresh_ob.best_yes_bid
                if fresh_ob.best_yes_ask is not None:
                    trade_ask = fresh_ob.best_yes_ask

            # Stash live prices for position PnL refresh below.
            if yes_bid is not None or yes_ask is not None:
                live_prices[market.ticker] = (yes_bid or 0.0, yes_ask or 0.0)

            # Open Markets snapshot — terminal display uses display prices.
            markets_snapshot.append({
                "ticker": market.ticker,
                "strike": market.strike_price,
                "yes_bid": yes_bid if yes_bid else None,
                "yes_ask": yes_ask if yes_ask else None,
                "volume": market.volume,
                "seconds_left": round(secs),
                "status": market.status.value,
            })

            # Determine whether this market is in the entry price window.
            # Deep ITM/OTM contracts are excluded from new entries — the model
            # can't add edge there — but we still run the model and show signals
            # so the UI never goes blank mid-session (e.g. after a profit-take
            # pushes the contract above 85¢).
            has_position = market.ticker in self.autotrader.positions
            mid_price = None
            if yes_bid and yes_ask:
                mid_price = (yes_bid + yes_ask) / 2
            elif yes_bid:
                mid_price = yes_bid
            elif yes_ask:
                mid_price = yes_ask
            max_entry = getattr(self.cfg.trader, "max_entry_price_cents", 85)
            in_entry_window = (
                mid_price is None
                or has_position
                or self.cfg.trader.min_entry_price_cents <= mid_price <= max_entry
            )

            # Orderbook depth for imbalance signal
            bid_depth, ask_depth = await self._market_cache.get_orderbook_depth(market.ticker)

            # Run ensemble model for all tradeable markets — display uses this
            # regardless of the entry window check above.
            output = self.ensemble.predict(
                ticker=market.ticker,
                strike=market.strike_price,
                current_price=current_price,
                seconds_remaining=secs,
                annual_vol=annual_vol,
                bars=bars,
                kalshi_yes_bid=yes_bid,
                kalshi_yes_ask=yes_ask,
                orderbook_bid_depth=bid_depth,
                orderbook_ask_depth=ask_depth,
                min_edge=self.cfg.trader.min_edge,
                min_confidence=self.cfg.trader.min_confidence,
            )

            signals_snapshot[market.ticker] = {
                "strike": market.strike_price,
                "seconds_left": round(secs),
                "prob_yes": round(output.prob_yes, 3),
                "prob_no": round(output.prob_no, 3),
                "confidence": round(output.confidence, 3),
                "edge_yes": round(output.edge_yes or 0, 3),
                "edge_no": round(output.edge_no or 0, 3),
                "signal": output.signal_str,
                "kalshi_price": round(((yes_bid or 0) + (yes_ask or 0)) / 2, 1),
                "tradeable": tradeable and in_entry_window,
                "ob_bid_depth": round(bid_depth, 1),
                "ob_ask_depth": round(ask_depth, 1),
            }

            # Dispatch to AutoTrader only when in the entry price window
            # (or holding a position that needs exit evaluation).
            if self.cfg.strategy.auto_trade and in_entry_window:
                ob_info = {"yes_bid": trade_bid, "yes_ask": trade_ask}
                mkt_info = {
                    "seconds_left": round(secs),
                    "volume": market.volume,
                    "ticker": market.ticker,
                    "annual_vol": annual_vol,
                }
                try:
                    actions = self.autotrader.evaluate(
                        ticker=market.ticker,
                        market_info=mkt_info,
                        orderbook=ob_info,
                        output=output,
                        bankroll_usd=bankroll,
                    )
                    for action in actions:
                        await self._execute_action(action)
                except Exception as e:
                    log.error(f"[AUTO] Error on {market.ticker}: {e}", exc_info=True)

            # Store ML features at entry time for training later
            if market.ticker in self.autotrader.positions and len(bars) >= 5:
                if market.ticker not in self._position_features:
                    self._position_features[market.ticker] = self.ensemble._build_ml_features(
                        market.strike_price, current_price, secs, annual_vol, bars
                    )

        # Always overwrite signals — even when empty (all markets filtered by price
        # range). The old guard `if signals_snapshot:` left the pane frozen on stale
        # T-left values whenever BTC moved all contracts deep ITM/OTM.
        self.state["signals"] = signals_snapshot
        if markets_snapshot:
            self.state["open_markets"] = markets_snapshot

        # Refresh position PnL using the live WS prices computed this scan.
        # _check_positions runs every 10s and uses stale REST prices — this
        # gives the Positions panel the same 3s refresh rate as everything else.
        if self.autotrader.positions and live_prices:
            refreshed = []
            for ticker, entries in self.autotrader.positions.items():
                wb, wa = live_prices.get(ticker, (0.0, 0.0))
                for entry in entries:
                    side = entry["side"]
                    if side == "yes":
                        bid = wb
                    else:
                        bid = max(0.0, 100.0 - wa) if wa else 0.0
                    cost = round(entry["contracts"] * entry["entry_cents"] / 100, 2)
                    value = round(entry["contracts"] * bid / 100, 2)
                    refreshed.append({
                        "ticker": ticker,
                        "side": side,
                        "contracts": entry["contracts"],
                        "entry_cents": entry["entry_cents"],
                        "cost": cost,
                        "value": value,
                        "pnl": round(value - cost, 2),
                        "mode": entry.get("mode", "directional"),
                        "source": "auto",
                    })
            if refreshed:
                self.state["open_positions"] = refreshed

    # ── Action execution ──────────────────────────────────────────────────────

    async def _execute_action(self, action: Action):
        """Route an AutoTrader action to the appropriate Kalshi API call."""
        paper = self.cfg.strategy.paper_trade
        label = "[PAPER]" if paper else "[LIVE]"

        if action.action_type == "buy":
            # Guard: don't enter if we hold a position on the OPPOSITE side
            # (reversal sell may have failed). Same-side buys are allowed for
            # pyramiding (adding to a winner).
            if action.ticker in self.autotrader.positions:
                existing = self.autotrader.positions[action.ticker]
                opposite_side = any(p["side"] != action.side for p in existing)
                if opposite_side:
                    log.info(
                        f"[AUTO] Skipping buy {action.ticker} {action.side} — "
                        f"existing position not yet cleared"
                    )
                    return

            side_enum = Side(action.side)

            allowed, reason = self.risk.check_trade(
                ticker=action.ticker,
                side=action.side,
                contracts=action.contracts,
                price_cents=action.price_cents,
                bankroll_usd=self._cached_balance_usd,
            )
            if not allowed:
                log.info(f"[AUTO] [RISK BLOCK] {action.ticker}: {reason}")
                return

            use_gtc = (action.time_in_force == "gtc")
            order_tag = "GTC" if use_gtc else "IOC"
            log.info(
                f"{label} [AUTO] BUY [{order_tag}]: {action.ticker} {action.side.upper()} "
                f"×{action.contracts} @ {action.price_cents}¢ | {action.reason}"
            )

            trade_id = f"T{uuid.uuid4().hex[:8].upper()}"

            if not paper:
                try:
                    tif = TimeInForce.GTC if use_gtc else TimeInForce.IOC
                    stp = (
                        SelfTradePrevention(action.self_trade_prevention)
                        if action.self_trade_prevention else None
                    )
                    # For GTC entries, set server-side expiration matching our
                    # local escalation timeout. Kalshi auto-cancels at expiration,
                    # giving us a deterministic upper bound on resting time and
                    # eliminating the orphan-resting-order class of bugs after
                    # crashes. Manual escalation still runs as a belt-and-suspenders.
                    exp_ts = None
                    if use_gtc:
                        exp_ts = int(time.time() + self.cfg.trader.gtc_escalate_seconds)
                    self._recently_placed[action.ticker] = time.time()
                    order = await self._kalshi.place_order(
                        ticker=action.ticker,
                        side=side_enum,
                        contracts=action.contracts,
                        price_cents=action.price_cents,
                        order_type=OrderType.LIMIT,
                        post_only=action.post_only,
                        time_in_force=tif,
                        client_order_id=f"btc15-auto-{uuid.uuid4().hex[:6]}",
                        self_trade_prevention=stp,
                        expiration_time=exp_ts,
                    )
                    self._open_orders[order.order_id] = order

                    if use_gtc:
                        # GTC: order rests on book — fill arrives via WS fill event.
                        # Record as resting; _handle_fill will call record_fill + record_open.
                        if order.status != OrderStatus.CANCELED:
                            purpose = "mm" if "mm_quote" in action.reason else "entry"
                            mode = "mm" if purpose == "mm" else "directional"
                            self.autotrader.record_order(
                                order.order_id, action.ticker, action.side,
                                action.price_cents, action.contracts,
                                purpose=purpose, mode=mode,
                                signal_mid_cents=action.signal_mid_cents,
                            )
                            log.debug(
                                f"[AUTO] GTC resting: {action.ticker} {action.side.upper()} "
                                f"×{action.contracts} @ {action.price_cents}¢ "
                                f"order={order.order_id[:8]}"
                            )
                            # NOTE: do NOT insert a "resting" entry into
                            # recent_trades. The Personas panel already shows
                            # resting-order count via `R:N`, so the user has
                            # visibility into pending orders without polluting
                            # Recent Trades with pseudo-fills. When the GTC
                            # actually fills via WS, _handle_fill inserts the
                            # real entry (status="open") with the correct
                            # post-fill trade_id. If the GTC is cancelled
                            # (e.g., server-side expiration_time hits), no
                            # cleanup is needed since nothing was inserted.
                        return
                    else:
                        # IOC: either filled now or cancelled
                        filled = order.filled_count if order.filled_count > 0 else (
                            action.contracts if order.status == OrderStatus.EXECUTED else 0
                        )
                        if filled == 0:
                            # Single retry with fresh price
                            retry_price = await self._fresh_entry_price(
                                action.ticker, side_enum, action.price_cents
                            )
                            if retry_price is not None:
                                log.info(
                                    f"[AUTO] IOC retry: {action.ticker} {action.side.upper()} "
                                    f"×{action.contracts} @ {retry_price}¢ (was {action.price_cents}¢)"
                                )
                                self._recently_placed[action.ticker] = time.time()
                                retry_order = await self._kalshi.place_order(
                                    ticker=action.ticker, side=side_enum,
                                    contracts=action.contracts, price_cents=retry_price,
                                    order_type=OrderType.LIMIT, time_in_force=TimeInForce.IOC,
                                    client_order_id=f"btc15-auto-retry-{uuid.uuid4().hex[:6]}",
                                )
                                filled = retry_order.filled_count if retry_order.filled_count > 0 else (
                                    action.contracts if retry_order.status == OrderStatus.EXECUTED else 0
                                )
                                if filled > 0:
                                    action.price_cents = retry_price
                            if filled == 0:
                                # If this was a GTC escalation, check how far the price
                                # moved from the original GTC price to decide next action.
                                if "gtc_escalated" in action.reason:
                                    orig = action.original_price_cents
                                    drift = (
                                        abs(action.price_cents - orig)
                                        if orig is not None else 99
                                    )
                                    if drift >= 10:
                                        # Market moved significantly — don't chase.
                                        self.autotrader._entry_retry_cooldown[action.ticker] = (
                                            time.time() + 90
                                        )
                                        log.warning(
                                            f"[AUTO] IOC unfilled + drift {drift}¢ — "
                                            f"90s entry cooldown: {action.ticker}"
                                        )
                                        return
                                    else:
                                        # Small drift: market is stable but book is thin.
                                        # Force one final IOC at retry_price + extra slippage
                                        # rather than cycling through another 12s GTC.
                                        if retry_price is not None:
                                            force_price = min(
                                                99,
                                                retry_price + self.cfg.kalshi.limit_slippage_cents
                                            )
                                            log.info(
                                                f"[AUTO] IOC drift only {drift}¢ — forcing "
                                                f"final IOC @ {force_price}¢: {action.ticker}"
                                            )
                                            self._recently_placed[action.ticker] = time.time()
                                            force_ord = await self._kalshi.place_order(
                                                ticker=action.ticker, side=side_enum,
                                                contracts=action.contracts,
                                                price_cents=force_price,
                                                order_type=OrderType.LIMIT,
                                                time_in_force=TimeInForce.IOC,
                                                client_order_id=f"btc15-force-{uuid.uuid4().hex[:6]}",
                                            )
                                            filled = (
                                                force_ord.filled_count
                                                if force_ord.filled_count > 0
                                                else (
                                                    action.contracts
                                                    if force_ord.status == OrderStatus.EXECUTED
                                                    else 0
                                                )
                                            )
                                            if filled > 0:
                                                action.price_cents = force_price
                                                # fall through to record_fill below
                                            else:
                                                # All 3 IOC attempts failed — book is too thin even
                                                # at small drift. Set a short cooldown to prevent
                                                # immediately posting another GTC that will just
                                                # escalate at an even worse price next cycle.
                                                self.autotrader._entry_retry_cooldown[action.ticker] = (
                                                    time.time() + 30
                                                )
                                                log.info(
                                                    f"[AUTO] All 3 IOC attempts failed: "
                                                    f"{action.ticker} — 30s cooldown"
                                                )
                                                return
                                        else:
                                            return
                                else:
                                    log.warning(
                                        f"[AUTO] IOC unfilled: {action.ticker} {action.side.upper()} "
                                        f"×{action.contracts} @ {action.price_cents}¢ — market moved"
                                    )
                                    return

                        self.autotrader.record_fill(
                            action.ticker, action.side, filled, action.price_cents, trade_id
                        )
                        self.risk.record_open(
                            action.ticker, action.side, filled, action.price_cents
                        )
                        self._log_trade(
                            action.ticker, action.side, filled,
                            action.price_cents, f"auto/{action.reason}", trade_id
                        )

                except KalshiAPIError as e:
                    if "post only cross" in str(e).lower():
                        log.debug(
                            f"[AUTO] Post-only skipped (market moved): "
                            f"{action.ticker} {action.side} @ {action.price_cents}¢"
                        )
                    else:
                        log.error(f"[AUTO] Order failed: {e}")
                    return

            else:
                # Paper trade — assume immediate fill
                self.autotrader.record_fill(
                    action.ticker, action.side,
                    action.contracts, action.price_cents, trade_id
                )
                self.risk.record_open(
                    action.ticker, action.side, action.contracts, action.price_cents
                )
                self._log_trade(
                    action.ticker, action.side, action.contracts,
                    action.price_cents, f"auto/{action.reason}", trade_id
                )

            self.state["recent_trades"].insert(0, {
                "ticker": action.ticker, "side": action.side,
                "contracts": action.contracts, "price_cents": action.price_cents,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "source": "auto", "status": "open",
                "trade_id": trade_id, "session": self._session_label,
            })
            self.state["recent_trades"] = self.state["recent_trades"][:50]

        elif action.action_type == "sell":
            log.info(
                f"{label} [AUTO] SELL: {action.ticker} {action.side.upper()} "
                f"×{action.contracts} (sweep, target ~{action.price_cents}¢) | {action.reason}"
            )

            if not paper and self._kalshi:
                # SWEEP exits: submit the IOC sell at the 1¢ floor so Kalshi
                # takes whatever bid exists, highest first. The old exact-bid
                # limit approach filled=0 whenever the top-of-book was a phantom
                # — positions rode stop-losses to 0¢ even with real liquidity
                # sitting one tick below. Sweep semantics fix that: we always
                # get out on the first cycle, at whatever the book pays.
                try:
                    order = await self._kalshi.sell_position_sweep(
                        action.ticker, Side(action.side), action.contracts,
                    )
                    filled = order.filled_count if order.filled_count > 0 else (
                        action.contracts if order.status == OrderStatus.EXECUTED else 0
                    )

                    if filled == 0:
                        # Book is genuinely empty (no bids at all). This is rare
                        # — usually means the market is already at 0¢ / 100¢ and
                        # awaiting settlement. Don't latch `settling=True`;
                        # personas will keep firing exits and if any bid appears
                        # before expiry the next scan will take it.
                        log.warning(
                            f"[AUTO] EXIT sweep filled=0: {action.ticker} "
                            f"{action.side.upper()} ×{action.contracts} — "
                            f"book empty, retry next scan"
                        )
                        for pos in self.autotrader.positions.get(action.ticker, []):
                            if pos["side"] == action.side:
                                pos["failed_exit_count"] = pos.get("failed_exit_count", 0) + 1
                        return

                    # Determine the effective fill price for P&L logging.
                    #
                    # Kalshi's order response echoes a price field, but its
                    # semantics on a POST that filled immediately via IOC are
                    # not ironclad — it may return the submitted limit (1¢)
                    # or the average fill price. The reconcile path trusts
                    # it for EXECUTED orders (see _reconcile_positions), so
                    # we do too — BUT if the echoed value equals our 1¢ floor
                    # it almost certainly means Kalshi returned the limit and
                    # not the fill, in which case we fall back to the pre-submit
                    # estimate from personas (set from the observed top-of-book
                    # bid at scan time). The ledger reconcile path will correct
                    # any residual drift from the true Kalshi fill.
                    echoed_price = (
                        order.yes_price if Side(action.side) == Side.YES
                        else order.no_price
                    )
                    if echoed_price > 1:
                        actual_price = echoed_price
                    elif action.price_cents > 0:
                        actual_price = action.price_cents
                    else:
                        actual_price = 1
                    action.price_cents = actual_price

                    if filled < action.contracts:
                        # Partial sweep — book ran out mid-fill. Record what
                        # filled, leave the remainder for the next scan.
                        for pos in self.autotrader.positions.get(action.ticker, []):
                            if pos["side"] == action.side:
                                partial_pnl = (
                                    (actual_price - pos["entry_cents"]) * filled / 100
                                )
                                pos["contracts"] -= filled
                                self.risk.record_close(
                                    action.ticker, won=(partial_pnl > 0), pnl=partial_pnl
                                )
                                self._log_trade(
                                    action.ticker, f"{action.side}_partial_exit",
                                    filled, actual_price,
                                    f"auto/{action.reason}", pos.get("trade_id", "")
                                )
                                log.info(
                                    f"[AUTO] PARTIAL EXIT: {action.ticker} {action.side.upper()} "
                                    f"filled={filled}/{action.contracts} @ {actual_price}¢ "
                                    f"pnl=${partial_pnl:+.2f} | {pos['contracts']} remaining"
                                )
                                return
                except Exception as e:
                    log.error(f"[AUTO] Exit sweep failed: {e}")
                    # Dedicated counter for genuine API exceptions — keeps the
                    # thin-book retry counter (`failed_exit_count`) clean.
                    # Latches `settling=True` only after 3 consecutive real
                    # failures, which is a terminal API-down condition.
                    for pos in self.autotrader.positions.get(action.ticker, []):
                        if pos["side"] == action.side:
                            pos["exit_exception_count"] = pos.get("exit_exception_count", 0) + 1
                            if pos["exit_exception_count"] >= 3:
                                pos["settling"] = True
                    return

            # Compute P&L and record close
            pnl = 0.0
            trade_id = ""
            for pos in self.autotrader.positions.get(action.ticker, []):
                if pos["side"] == action.side:
                    pnl = (action.price_cents - pos["entry_cents"]) * action.contracts / 100
                    trade_id = pos.get("trade_id", "")
                    break

            self.autotrader.record_exit(action.ticker, pnl, side=action.side)
            self._recently_exited[action.ticker] = time.time()
            self.risk.record_close(action.ticker, won=(pnl > 0), pnl=pnl)
            self._log_trade(
                action.ticker, f"{action.side}_exit",
                action.contracts, action.price_cents,
                f"auto/{action.reason}", trade_id
            )

            self.state["recent_trades"].insert(0, {
                "ticker": action.ticker,
                "side": f"{action.side}_exit",
                "contracts": action.contracts,
                "price_cents": action.price_cents,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "source": "auto",
                "pnl": round(pnl, 3),
                "trade_id": trade_id,
                "session": self._session_label,
            })
            self.state["recent_trades"] = self.state["recent_trades"][:50]

        elif action.action_type == "amend":
            if not paper and self._kalshi and action.order_id:
                try:
                    log.debug(f"[AUTO] AMEND: {action.order_id[:8]} → {action.price_cents}¢")
                    await self._kalshi.amend_order(
                        action.order_id,
                        ticker=action.ticker,
                        side=Side(action.side),
                        price_cents=action.price_cents,
                    )
                    if action.order_id in self.autotrader.resting_orders:
                        self.autotrader.resting_orders[action.order_id]["price"] = action.price_cents
                except KalshiAPIError as e:
                    if e.status in (404, 409):
                        log.info(
                            f"[AUTO] AMEND {action.order_id[:8]}: "
                            f"order gone (HTTP {e.status}) — removing"
                        )
                        self.autotrader.remove_order(action.order_id)
                    else:
                        log.debug(f"[AUTO] Amend failed: {e}")
                except Exception as e:
                    log.debug(f"[AUTO] Amend failed: {e}")
            elif paper and action.order_id in self.autotrader.resting_orders:
                self.autotrader.resting_orders[action.order_id]["price"] = action.price_cents

        elif action.action_type == "cancel":
            # Strip the resting placeholder from the terminal display before the order is gone.
            # Without this, cancelled GTC entries linger as "resting" in recent_trades forever.
            _cinfo = self.autotrader.resting_orders.get(action.order_id or "", {})
            if _cinfo:
                _ct, _cs = _cinfo.get("ticker"), _cinfo.get("side")
                self.state["recent_trades"] = [
                    t for t in self.state["recent_trades"]
                    if not (t.get("ticker") == _ct
                            and t.get("side") == _cs
                            and t.get("status") == "resting")
                ]

            if not paper and self._kalshi and action.order_id:
                try:
                    log.debug(f"[AUTO] CANCEL: {action.order_id[:8]}")
                    await self._kalshi.cancel_order(action.order_id)
                    self.autotrader.remove_order(action.order_id)
                except KalshiAPIError as e:
                    if e.status == 404:
                        log.info(
                            f"[AUTO] CANCEL {action.order_id[:8]}: already gone — removing"
                        )
                        self.autotrader.remove_order(action.order_id)
                    else:
                        log.debug(f"[AUTO] Cancel failed: {e}")
                except Exception as e:
                    log.debug(f"[AUTO] Cancel failed: {e}")
            elif paper and action.order_id:
                self.autotrader.remove_order(action.order_id)

        elif action.action_type == "batch_buy":
            # Risk gate each leg
            for bo in action.batch_orders:
                _pd = bo.get("yes_price_dollars") or bo.get("no_price_dollars")
                price = int(round(float(_pd) * 100)) if _pd else (bo.get("yes_price") or bo.get("no_price") or 50)
                leg_cost = bo.get("count", 0) * price / 100
                if leg_cost > self.cfg.risk.max_trade_usd:
                    log.info(
                        f"[AUTO] [RISK BLOCK] {action.ticker}: batch leg "
                        f"${leg_cost:.2f} exceeds max ${self.cfg.risk.max_trade_usd:.2f}"
                    )
                    return

            log.info(
                f"{label} [AUTO] BATCH BUY: {action.ticker} | "
                f"{len(action.batch_orders)} orders | {action.reason}"
            )

            if not paper and self._kalshi:
                try:
                    # Arm reconcile-skip for every ticker in the batch (arb pairs
                    # touch two tickers at once).
                    now_ts = time.time()
                    for bo in action.batch_orders:
                        self._recently_placed[bo.get("ticker", action.ticker)] = now_ts
                    orders = await self._kalshi.batch_place_orders(action.batch_orders)
                    for o in orders:
                        self._open_orders[o.order_id] = o
                    # Register each leg as resting — fill arrives via WS fill event
                    for o in orders:
                        price = o.yes_price if o.side == Side.YES else o.no_price
                        self.autotrader.record_order(
                            o.order_id, action.ticker, o.side.value,
                            price, o.count, purpose="arb", mode="arb"
                        )
                    for o in orders:
                        price = o.yes_price if o.side == Side.YES else o.no_price
                        self._log_trade(
                            action.ticker, o.side.value, o.count,
                            price, f"auto/{action.reason}"
                        )
                        self.state["recent_trades"].insert(0, {
                            "ticker": action.ticker, "side": o.side.value,
                            "contracts": o.count, "price_cents": price,
                            "entry_time": datetime.now(timezone.utc).isoformat(),
                            "source": "auto/arb", "status": "resting",
                            "session": self._session_label,
                        })
                except KalshiAPIError as e:
                    log.error(f"[AUTO] Batch order failed: {e}")
                    return
            else:
                batch_id = f"T{uuid.uuid4().hex[:8].upper()}"
                for bo in action.batch_orders:
                    side_str = bo["side"]
                    _pd = bo.get("yes_price_dollars") or bo.get("no_price_dollars")
                    price = int(round(float(_pd) * 100)) if _pd else (bo.get("yes_price") or bo.get("no_price") or 50)
                    trade_id = f"{batch_id}-{side_str[:1].upper()}"
                    self.autotrader.record_fill(
                        action.ticker, side_str, bo["count"], price, trade_id, mode="arb"
                    )
                    self.risk.record_open(action.ticker, side_str, bo["count"], price)
                    self._log_trade(
                        action.ticker, side_str, bo["count"],
                        price, f"auto/{action.reason}", trade_id
                    )
                    self.state["recent_trades"].insert(0, {
                        "ticker": action.ticker, "side": side_str,
                        "contracts": bo["count"], "price_cents": price,
                        "entry_time": datetime.now(timezone.utc).isoformat(),
                        "source": "auto/arb", "status": "open",
                        "trade_id": trade_id, "session": self._session_label,
                    })

            self.state["recent_trades"] = self.state["recent_trades"][:50]

    # ── Reconcile resting orders after WS reconnect (or proactively) ─────────

    async def _reconcile_positions(self):
        """
        Full reconcile: checks resting orders AND compares actual Kalshi
        positions against internal tracking. Runs every 5s.
        Catches: missed WS fills, stacked-order phantom contracts, any gap
        between what Kalshi holds and what autotrader.positions tracks.

        Guards:
          - asyncio.Lock: only one reconcile runs at a time (prevents two concurrent
            tasks from both seeing the same gap and double-adopting)
          - _settled_tickers: skip tickers settled in the last 60s to avoid re-adopting
            positions still visible in Kalshi's API due to settlement lag
        """
        if not self._kalshi or self.cfg.strategy.paper_trade:
            return
        if self._reconcile_lock.locked():
            return  # another reconcile is already running — skip this cycle
        async with self._reconcile_lock:
            try:
                # Clean up stale guard entries
                now_t = time.time()
                self._settled_tickers = {
                    t: ts for t, ts in self._settled_tickers.items()
                    if now_t - ts < 120
                }
                self._recently_exited = {
                    t: ts for t, ts in self._recently_exited.items()
                    if now_t - ts < 60
                }
                self._recently_placed = {
                    t: ts for t, ts in self._recently_placed.items()
                    if now_t - ts < 30
                }

                # Step 1: reconcile resting orders (fills missed during WS gaps)
                await self._reconcile_resting_orders()

                # Step 2: compare actual Kalshi positions against internal tracking
                api_positions = await self._kalshi.get_positions()
                for api_pos in api_positions:
                    ticker = api_pos.ticker
                    side = api_pos.side.value       # "yes" | "no"
                    api_contracts = api_pos.contracts

                    # Only reconcile our own series — ignore any other Kalshi positions
                    # (e.g. manually-traded Trump or election markets) to avoid adopting
                    # positions that have nothing to do with BTC prediction.
                    if not ticker.startswith(self.cfg.kalshi.series_ticker):
                        log.debug(f"[RECONCILE] Skipping {ticker} — not {self.cfg.kalshi.series_ticker}")
                        continue

                    # Skip tickers recently settled — Kalshi API lag can return stale positions
                    # for up to ~30s after settlement, causing ghost re-adoptions
                    if ticker in self._settled_tickers:
                        log.debug(
                            f"[RECONCILE] Skipping {ticker} — settled "
                            f"{now_t - self._settled_tickers[ticker]:.0f}s ago (API lag guard)"
                        )
                        continue

                    # Skip tickers we just exited — Kalshi API lags ~10-30s before removing
                    # a position from get_positions() after a sell. Without this guard the
                    # reconciler re-adopts the just-cleared contracts as a phantom gap.
                    if ticker in self._recently_exited:
                        age = now_t - self._recently_exited[ticker]
                        log.debug(
                            f"[RECONCILE] Skipping {ticker} — exited {age:.0f}s ago (exit lag guard)"
                        )
                        continue

                    # Skip tickers with a place_order in flight or just-completed.
                    # Closes the race where an IOC fills on Kalshi within ms but our
                    # local record_fill hasn't run yet — without this guard, the
                    # reconciler adopts the same fill again as a "reconciled_gap"
                    # ghost, doubling our internal exposure.
                    placed_ts = self._recently_placed.get(ticker)
                    if placed_ts is not None and (now_t - placed_ts) < 15:
                        log.debug(
                            f"[RECONCILE] Skipping {ticker} — placed "
                            f"{now_t - placed_ts:.1f}s ago (placement race guard)"
                        )
                        continue

                    # Initial gap estimate (pre-await snapshot)
                    tracked_pre = sum(
                        p["contracts"]
                        for p in self.autotrader.positions.get(ticker, [])
                        if p["side"] == side
                    )
                    resting_pre = sum(
                        info["contracts"]
                        for info in self.autotrader.resting_orders.values()
                        if info.get("ticker") == ticker and info.get("side") == side
                    )
                    if api_contracts - (tracked_pre + resting_pre) <= 0:
                        continue  # no gap, fast path

                    # Re-check tracked immediately before adoption to close the TOCTOU
                    # window: a WS _handle_fill event can update autotrader.positions
                    # between the get_positions() await and here, making the pre-check stale.
                    # Since asyncio is cooperative, this re-check runs without any await,
                    # so it's guaranteed to see the latest state before we commit.
                    tracked = sum(
                        p["contracts"]
                        for p in self.autotrader.positions.get(ticker, [])
                        if p["side"] == side
                    )
                    resting = sum(
                        info["contracts"]
                        for info in self.autotrader.resting_orders.values()
                        if info.get("ticker") == ticker and info.get("side") == side
                    )
                    gap = api_contracts - (tracked + resting)
                    if gap <= 0:
                        log.debug(
                            f"[RECONCILE] {ticker} {side.upper()}: gap closed by concurrent fill "
                            f"(api={api_contracts} tracked={tracked} resting={resting})"
                        )
                        continue

                    # Kalshi has more contracts than we track — adopt the difference
                    price = int(api_pos.avg_price_cents) if api_pos.avg_price_cents else 50
                    trade_id = f"T{uuid.uuid4().hex[:8].upper()}"
                    log.warning(
                        f"[RECONCILE] Position gap: {ticker} {side.upper()} "
                        f"Kalshi={api_contracts} tracked={tracked} resting={resting} "
                        f"→ adopting {gap} missing contracts @ {price}¢"
                    )
                    self.autotrader.record_fill(ticker, side, gap, price, trade_id, mode="directional")
                    self.risk.record_open(ticker, side, gap, price)
                    self._log_trade(ticker, side, gap, price, "auto/reconciled_gap", trade_id)
                    self.state["recent_trades"].insert(0, {
                        "ticker": ticker, "side": side, "contracts": gap,
                        "price_cents": price,
                        "entry_time": datetime.now(timezone.utc).isoformat(),
                        "source": "auto/reconciled_gap", "status": "open",
                        "trade_id": trade_id, "session": self._session_label,
                    })
                self.state["recent_trades"] = self.state["recent_trades"][:50]
            except Exception as e:
                log.debug(f"[RECONCILE] Error: {e}")

    async def _reconcile_resting_orders(self):
        """
        Check all resting orders against the Kalshi REST API.
        Synthesizes fill events for any that filled during a WS disconnect gap.
        Called: (a) after every WS reconnect, (b) via _reconcile_positions every 5s.
        """
        if not self._kalshi or not self.autotrader.resting_orders:
            return

        resting = dict(self.autotrader.resting_orders)
        log.info(f"[RECONCILE] Checking {len(resting)} resting order(s)")

        tickers = {info["ticker"] for info in resting.values()}
        api_orders: dict[str, Order] = {}
        fetched_tickers: set[str] = set()
        for ticker in tickers:
            try:
                orders = await self._kalshi.get_orders(ticker=ticker)
                for o in orders:
                    api_orders[o.order_id] = o
                fetched_tickers.add(ticker)
            except Exception as e:
                log.warning(f"[RECONCILE] Failed to fetch orders for {ticker}: {e}")

        for oid, info in resting.items():
            # Bug 3: skip null/empty order IDs from edge-case arb pair registration
            if not oid:
                continue
            # Bug 1: if this order was removed from resting_orders during the get_orders() await
            # (e.g. by GTC→IOC escalation that also called record_fill), skip it here to
            # prevent recording a second fill for the same order → phantom position inflation.
            if oid not in self.autotrader.resting_orders:
                log.debug(f"[RECONCILE] Order {oid[:8]} removed during reconcile await — skipping")
                continue

            ticker = info["ticker"]
            # Defensive: if the get_orders fetch failed for this ticker, do NOT
            # synthesize a phantom fill. The old logic treated "api_order is None"
            # as "must have filled and been removed", but that's only true when
            # the fetch SUCCEEDED and returned an empty/missing entry. If the
            # fetch itself failed (404, network error, etc.), we just don't know
            # — wait for the next reconcile cycle or the WS fill event. This is
            # the bug that produced paired duplicate rows in the trade log:
            # phantom + real fill on the same Kalshi position.
            if ticker not in fetched_tickers:
                log.debug(
                    f"[RECONCILE] Skipping {oid[:8]} on {ticker} — verification "
                    f"fetch failed; will retry next cycle"
                )
                continue

            side_str = info["side"]
            price = info["price"]
            contracts = info["contracts"]
            mode = info.get("mode", "directional")
            api_order = api_orders.get(oid)

            if api_order is None or api_order.status == OrderStatus.EXECUTED:
                filled = contracts
                if api_order and api_order.filled_count > 0:
                    filled = api_order.filled_count
                    if api_order.side == Side.YES and api_order.yes_price:
                        price = api_order.yes_price
                    elif api_order.side == Side.NO and api_order.no_price:
                        price = api_order.no_price

                trade_id = f"T{uuid.uuid4().hex[:8].upper()}"
                log.warning(
                    f"[RECONCILE] Missed fill: {ticker} {side_str.upper()} "
                    f"×{filled} @ {price}¢ | order={oid[:8]}"
                )
                self.autotrader.record_fill(ticker, side_str, filled, price, trade_id, mode=mode)
                self.risk.record_open(ticker, side_str, filled, price)
                self._log_trade(ticker, side_str, filled, price, "auto/reconciled", trade_id)
                self.autotrader.remove_order(oid)

                self.state["recent_trades"].insert(0, {
                    "ticker": ticker, "side": side_str, "contracts": filled,
                    "price_cents": price,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "source": "auto/reconciled", "status": "open",
                    "trade_id": trade_id, "session": self._session_label,
                })

            elif api_order.status == OrderStatus.CANCELED:
                log.info(f"[RECONCILE] Order {oid[:8]} cancelled externally — removing")
                self.autotrader.remove_order(oid)

        self.state["recent_trades"] = self.state["recent_trades"][:50]

    async def _handle_fill(self, msg: dict):
        """Route Kalshi WS fill events to the AutoTrader."""
        data = msg.get("msg", {})
        order_id = data.get("order_id", "")
        ticker = data.get("market_ticker", "")
        side_str = data.get("side", "")
        count = int(data.get("count") or 0)
        yes_price = int(data.get("yes_price") or 0)
        no_price = int(data.get("no_price") or 0)
        price = yes_price if side_str == "yes" else no_price

        if not order_id or not count or not ticker:
            return

        if order_id not in self.autotrader.resting_orders:
            # Fill arrived for an order we don't have in resting_orders.
            # This happens when a stacked duplicate order filled after the first
            # order's fill already cleared it from resting_orders.
            # Don't silently drop it — adopt the fill immediately.
            log.warning(
                f"[FILL] Untracked fill: {ticker} {side_str.upper()} "
                f"×{count} @ {price}¢ order={order_id[:8]} — adopting"
            )
            trade_id = f"T{uuid.uuid4().hex[:8].upper()}"
            self.autotrader.record_fill(ticker, side_str, count, price, trade_id, mode="directional")
            self.risk.record_open(ticker, side_str, count, price)
            self._log_trade(ticker, side_str, count, price, "auto/untracked_fill", trade_id)
            self.state["recent_trades"].insert(0, {
                "ticker": ticker, "side": side_str, "contracts": count,
                "price_cents": price,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "source": "auto/untracked_fill", "status": "open",
                "trade_id": trade_id, "session": self._session_label,
            })
            self.state["recent_trades"] = self.state["recent_trades"][:50]
            return

        info = self.autotrader.resting_orders[order_id]
        mode = info.get("mode", "directional")
        trade_id = f"T{uuid.uuid4().hex[:8].upper()}"

        log.info(
            f"[FILL] [AUTO] {ticker} {side_str.upper()} "
            f"×{count} @ {price}¢ | order={order_id[:8]} mode={mode}"
        )

        self.autotrader.record_fill(ticker, side_str, count, price, trade_id, mode=mode)
        self.risk.record_open(ticker, side_str, count, price)
        self._log_trade(ticker, side_str, count, price, "auto/fill", trade_id)

        remaining = info.get("contracts", 0) - count
        if remaining <= 0:
            self.autotrader.remove_order(order_id)
        else:
            self.autotrader.resting_orders[order_id]["contracts"] = remaining

        # Remove the "resting" placeholder for this ticker+side so the terminal
        # doesn't show both a resting entry and a filled entry for the same order.
        self.state["recent_trades"] = [
            t for t in self.state["recent_trades"]
            if not (t.get("ticker") == ticker
                    and t.get("side") == side_str
                    and t.get("status") == "resting")
        ]
        self.state["recent_trades"].insert(0, {
            "ticker": ticker, "side": side_str, "contracts": count,
            "price_cents": price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "source": "auto/fill", "status": "open",
            "trade_id": trade_id, "session": self._session_label,
        })
        self.state["recent_trades"] = self.state["recent_trades"][:50]

    async def _fresh_entry_price(
        self,
        ticker: str,
        side: Side,
        original_price: int,
    ) -> Optional[int]:
        """Re-fetch orderbook for IOC retry. Returns None if market moved against us."""
        try:
            ob = await self._kalshi.get_orderbook(ticker)
        except Exception:
            return None
        slip = self.cfg.kalshi.limit_slippage_cents
        if side == Side.YES:
            if ob.best_yes_ask is None:
                return None
            fresh_price = int(ob.best_yes_ask) + slip
        else:
            if ob.best_yes_bid is None:
                return None
            fresh_price = int(100 - ob.best_yes_bid) + slip
        fresh_price = max(1, min(99, fresh_price))
        if fresh_price > original_price:
            return None  # market moved against us
        return fresh_price

    # ── Position monitoring ───────────────────────────────────────────────────

    async def _position_loop(self):
        while self.running:
            try:
                await self._check_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Position loop error: {e}", exc_info=True)
            await asyncio.sleep(POSITION_CHECK)

    async def _check_positions(self):
        if not self._kalshi:
            return

        try:
            api_positions = await self._kalshi.get_positions()
            if self.cfg.strategy.paper_trade:
                self.risk.state.open_positions = sum(
                    len(v) for v in self.autotrader.positions.values()
                )
            else:
                self.risk.state.open_positions = len(api_positions)
        except Exception:
            api_positions = []

        # Build display from autotrader positions (reliable cost basis)
        display = []

        def _mtm(ticker: str, side: str, entry_cents: int) -> tuple[float, float]:
            mkt = self._watched_markets.get(ticker)
            if mkt is None:
                return entry_cents, 0.0
            bid = mkt.yes_bid if side == "yes" else max(0.0, 100.0 - (mkt.yes_ask or 100.0))
            return bid, bid - entry_cents

        for ticker, entries in self.autotrader.positions.items():
            for entry in entries:
                bid, _ = _mtm(ticker, entry["side"], entry["entry_cents"])
                cost = round(entry["contracts"] * entry["entry_cents"] / 100, 2)
                value = round(entry["contracts"] * bid / 100, 2)
                display.append({
                    "ticker": ticker,
                    "side": entry["side"],
                    "contracts": entry["contracts"],
                    "entry_cents": entry["entry_cents"],
                    "cost": cost,
                    "value": value,
                    "pnl": round(value - cost, 2),
                    "mode": entry.get("mode", "directional"),
                    "source": "auto",
                })

        # Orphan detection (live mode only)
        if not self.cfg.strategy.paper_trade and api_positions:
            tracked = set(self.autotrader.positions.keys())
            # Tickers with a pending resting order are excluded — fill still in-flight
            resting_tickers = {
                info["ticker"] for info in self.autotrader.resting_orders.values()
            }
            tracked.update(resting_tickers)

            for api_pos in api_positions:
                if api_pos.ticker not in tracked and api_pos.ticker not in self._settled_tickers:
                    avg_cents = max(1, min(99, round(api_pos.avg_price_cents)))
                    log.warning(
                        f"[ORPHAN] Adopting untracked position: {api_pos.ticker} "
                        f"{api_pos.side.value.upper()} ×{api_pos.contracts} avg={avg_cents}¢"
                    )
                    self.autotrader.record_fill(
                        api_pos.ticker, api_pos.side.value,
                        api_pos.contracts, avg_cents,
                        trade_id=f"ORPHAN-{api_pos.ticker[-8:]}",
                    )
                    self.risk.record_open(
                        api_pos.ticker, api_pos.side.value,
                        api_pos.contracts, avg_cents
                    )
                    bid, _ = _mtm(api_pos.ticker, api_pos.side.value, avg_cents)
                    cost = round(api_pos.contracts * avg_cents / 100, 2)
                    value = round(api_pos.contracts * bid / 100, 2)
                    display.append({
                        "ticker": api_pos.ticker,
                        "side": api_pos.side.value,
                        "contracts": api_pos.contracts,
                        "entry_cents": avg_cents,
                        "cost": cost, "value": value,
                        "pnl": round(value - cost, 2),
                        "mode": "adopted", "source": "adopted",
                    })

        self.state["open_positions"] = display
        await self._check_settlements()

    async def _check_settlements(self):
        """Detect settled/expired markets and resolve held positions."""
        now = datetime.now(timezone.utc)
        current_price = self.price_feed.current_price
        if not current_price:
            return

        all_tickers = set(self.autotrader.positions.keys())
        if not all_tickers:
            return

        for ticker in list(all_tickers):
            cached_market = self._watched_markets.get(ticker)
            result = None

            # Method 1: Kalshi official result (always preferred)
            try:
                market_data = await self._kalshi.get_market(ticker)
                if market_data.status in (MarketStatus.SETTLED, MarketStatus.FINALIZED) and market_data.result:
                    result = market_data.result
                    self._settlement_pending_count.pop(ticker, None)
                elif market_data.seconds_remaining > 0:
                    continue  # still open
                # else: closed but result not published yet — fall through below
            except Exception:
                pass

            # Method 2: Local BTC price fallback — paper mode ONLY.
            # In live mode Kalshi's own price feed determines settlement. Our Binance
            # feed differs by seconds and can flip the result (e.g. T2F47F66C settled
            # YES on Kalshi but our local Binance price 1s later showed BTC below strike).
            # In live mode: wait up to 18 checks (~3 min) for the official API result.
            if result is None and cached_market:
                close_time = cached_market.close_time
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=timezone.utc)
                if now > close_time:
                    if self.cfg.strategy.paper_trade and current_price > 0 and cached_market.strike_price > 0:
                        result = "yes" if current_price > cached_market.strike_price else "no"
                        log.info(
                            f"[SETTLEMENT] {ticker}: BTC ${current_price:,.2f} vs "
                            f"strike ${cached_market.strike_price:,.2f} → {result.upper()} (paper)"
                        )
                    else:
                        # Live: keep waiting for Kalshi — typically publishes within 30s
                        count = self._settlement_pending_count.get(ticker, 0) + 1
                        self._settlement_pending_count[ticker] = count
                        if count == 1:
                            log.info(f"[SETTLEMENT] {ticker}: market closed — awaiting Kalshi result")
                        elif count % 6 == 0:
                            log.warning(f"[SETTLEMENT] {ticker}: no result after {count * POSITION_CHECK}s")
                        continue
                else:
                    continue

            if result is None:
                continue

            for pos in list(self.autotrader.positions.get(ticker, [])):
                settlement_cents = 100 if result == pos["side"] else 0
                pnl = (settlement_cents - pos["entry_cents"]) * pos["contracts"] / 100
                won = pnl > 0

                label = "[PAPER]" if self.cfg.strategy.paper_trade else "[LIVE]"
                log.info(
                    f"{label} [AUTO] SETTLED: {ticker} {pos['side'].upper()} "
                    f"×{pos['contracts']} | entry={pos['entry_cents']}¢ → {settlement_cents}¢ | "
                    f"result={result.upper()} | pnl=${pnl:+.2f}"
                )

                trade_id = pos.get("trade_id", "")
                self.autotrader.record_exit(ticker, pnl, side=pos["side"])
                self.risk.record_close(ticker, won=won, pnl=pnl)
                self._log_trade(
                    ticker, f"{pos['side']}_settled",
                    pos["contracts"], settlement_cents,
                    f"settled/{result}", trade_id
                )

                self.state["recent_trades"].insert(0, {
                    "ticker": ticker, "side": f"{pos['side']}_settled",
                    "contracts": pos["contracts"], "price_cents": settlement_cents,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "source": f"settled/{result}",
                    "pnl": round(pnl, 3), "trade_id": trade_id,
                })

            # Cancel any remaining resting orders for this ticker
            for oid in [
                oid for oid, info in list(self.autotrader.resting_orders.items())
                if info.get("ticker") == ticker
            ]:
                self.autotrader.remove_order(oid)

            # Collect ML training data
            if ticker in self._position_features:
                outcome = 1 if result == "yes" else 0
                collect_sample(self._position_features.pop(ticker), outcome)

            # Suppress reconcile adoption for 60s after settlement.
            # Kalshi's REST API can lag and still return the settled position,
            # causing _reconcile_positions to create a ghost duplicate.
            self._settled_tickers[ticker] = time.time()

            self._watched_markets.pop(ticker, None)
            self.state["recent_trades"] = self.state["recent_trades"][:50]

    # ── State updater ─────────────────────────────────────────────────────────

    async def _state_updater(self):
        """Update dashboard state variables every second."""
        _reconcile_ts: float = 0.0  # track last proactive reconcile

        while self.running:
            try:
                price = self.price_feed.current_price
                if price:
                    self.state["current_price"] = price
                self.state["feed_age_sec"] = round(self.price_feed.feed_age_sec(), 1)
                risk_summary = self.risk.summary()
                self.state["risk"] = risk_summary
                self.state["trader"] = self.autotrader.summary()

                # Sample P&L history every 10s
                if int(time.time()) % 10 == 0:
                    pnl_now = round(risk_summary.get("session_pnl", 0.0), 3)
                    ts_now = datetime.now(timezone.utc).strftime("%H:%M")
                    history = self.state["pnl_history"]
                    if not history or history[-1][1] != pnl_now:
                        history.append((ts_now, pnl_now))
                        self.state["pnl_history"] = history[-500:]

                # Refresh balance display every 30s
                if self._kalshi and int(time.time()) % 30 == 0:
                    try:
                        bal = await self._kalshi.get_balance()
                        available = round(bal.available_usd, 2)
                        portfolio = round(bal.portfolio_usd, 2)
                        start_bal = self.state.get("session_start_balance")
                        # In paper mode, Kalshi balance never changes (trades are simulated
                        # locally), so true_pnl from the API would always read ~$0 and
                        # override the accurate internal risk.session_pnl. Leave it None so
                        # the dashboard falls back to risk.session_pnl instead.
                        if not self.cfg.strategy.paper_trade and start_bal is not None:
                            true_pnl = round((available + portfolio) - start_bal, 2)
                        else:
                            true_pnl = None
                        self.state["balance"] = {
                            "available": available,
                            "portfolio": portfolio,
                            "true_pnl": true_pnl,
                        }
                        self._cached_balance_usd = available
                        self._balance_cache_ts = time.time()
                        # Account-level halt check (catches unrealized drawdown)
                        if (
                            true_pnl is not None
                            and true_pnl <= -self.cfg.risk.daily_loss_limit_usd
                            and not self.risk.state.halted
                        ):
                            self.risk._halt(
                                f"Daily loss limit hit "
                                f"(account P&L ${true_pnl:+.2f}, "
                                f"limit -${self.cfg.risk.daily_loss_limit_usd:.0f})"
                            )
                    except Exception:
                        pass

                # Proactive reconciliation every 5s — catches missed fills fast
                now_ts = time.time()
                if now_ts - _reconcile_ts > 5.0:
                    _reconcile_ts = now_ts
                    asyncio.create_task(self._reconcile_positions())

            except Exception as e:
                log.debug(f"State updater error: {e}")
            await asyncio.sleep(1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ensure_trade_log(self):
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_file.exists():
            with open(self._log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trade_id", "timestamp", "ticker", "side", "contracts",
                    "price_cents", "cost_usd", "source", "mode", "session",
                ])

    def _log_trade(
        self,
        ticker: str,
        side: str,
        contracts: int,
        price_cents: int,
        source: str,
        trade_id: str = "",
    ):
        cost = contracts * price_cents / 100
        mode = "paper" if self.cfg.strategy.paper_trade else "live"
        with open(self._log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trade_id,
                datetime.now(timezone.utc).isoformat(),
                ticker, side, contracts, price_cents,
                round(cost, 4), source, mode, self._session_label,
            ])
