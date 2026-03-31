"""
Main strategy engine — the brain of the bot.

Lifecycle:
  1. On start: connect feeds + Kalshi client, seed historical data
  2. Every SCAN_INTERVAL seconds: fetch open KXBTC markets
  3. For each market: run ensemble model to compute P(YES) and edge
  4. If signal qualifies: compute position size, check risk, execute
  5. Monitor open positions for settlement
  6. Log all activity + collect ML training data

Entry timing strategy:
  - Prefer windows where 5–10 min remain (high conviction, less noise)
  - Only enter when confidence AND edge exceed thresholds
  - We don't need to hold to expiry — but for binary options we typically do
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
    Market, Order, OrderType, Side, MarketStatus,
    TimeInForce, SelfTradePrevention,
)
from btc15.kalshi.ws_client import KalshiWebSocket, MarketDataCache
from btc15.models.ensemble import EnsembleModel, ModelOutput
from btc15.models.ml_model import collect_sample
from btc15.risk.manager import RiskManager
from btc15.strategy.personas import (
    Action, SniperPersona, ScalperPersona, ArbPersona,
)
from btc15.strategy.sizer import size_position, log_bet_info

log = logging.getLogger(__name__)

SCAN_INTERVAL = 3       # seconds between market scans
OB_REFRESH_INTERVAL = 30  # full orderbook snapshot refresh for all watched markets

# ── In-memory log handler — pipes key log lines into dashboard state ──────────

class _DashboardLogHandler(logging.Handler):
    """Captures WARNING+ and key INFO lines into state["event_log"]."""

    _KEYWORDS = ("RISK BLOCK", "SIGNAL", "EXECUTING", "EXIT", "SETTLED",
                 "HALTED", "ERROR", "WARNING", "disconnected", "STAT ARB",
                 "QUOTING", "BATCH", "conf=", "edge=",
                 "SIZER", "EV=", "Cost=", "exceeds", "below min",
                 "exposure", "win rate", "HALT", "BUY:", "SELL:", "flip",
                 "[SKIP]")

    # Deduplicate keys: messages matching these fragments only appear once
    # per unique (market_ticker, fragment) combo — avoids [SKIP] spam
    _DEDUP_FRAGMENTS = ("[SKIP]",)

    def __init__(self, state: dict):
        super().__init__()
        self._state = state
        self._seen: set[str] = set()   # dedup keys already shown

    def emit(self, record: logging.LogRecord):
        try:
            raw = record.getMessage()
            lvl = record.levelno
            # Always capture warnings/errors; for INFO filter by keyword
            if lvl < logging.WARNING:
                if not any(kw in raw for kw in self._KEYWORDS):
                    return
            # Deduplicate noisy repeated messages (e.g. [SKIP] same market)
            for frag in self._DEDUP_FRAGMENTS:
                if frag in raw:
                    # Key = fragment + first 60 chars (captures market + reason)
                    key = frag + raw[:60]
                    if key in self._seen:
                        return
                    self._seen.add(key)
                    # Clear seen set occasionally so updated skips can resurface
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
POSITION_CHECK = 10     # seconds between position status checks
ORDERBOOK_REFRESH = 3   # seconds between orderbook polls


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

        # Live state (read by CLI dashboard)
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
            "risk": {},
            "last_scan": None,
            "paper_mode": config.strategy.paper_trade,
            "auto_trade": config.strategy.auto_trade,
            "personas": {},
            "session_start": datetime.now(timezone.utc).isoformat(),
            "pnl_history": [],   # list of (iso_timestamp, cumulative_pnl) sampled over session
            "event_log": [],
        }

        # Session label for trade tagging: e.g. "30MAR08:02" UTC
        _now = datetime.now(timezone.utc)
        self._session_label = _now.strftime("%d%b%H:%M").upper()  # "30MAR08:02"

        # Attach dashboard log handler to root logger so all modules feed into it
        self._dash_handler = _DashboardLogHandler(self.state)
        self._dash_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self._dash_handler)

        # Personas
        self.sniper = SniperPersona(config.personas.sniper)
        self.scalper = ScalperPersona(config.personas.scalper)
        self.arb = ArbPersona(config.personas.arb)
        self._personas = [self.sniper, self.scalper, self.arb]

        self._open_orders: dict[str, Order] = {}      # order_id → Order
        self._watched_markets: dict[str, Market] = {} # ticker → Market
        self._position_features: dict[str, list] = {} # ticker → ML features at entry
        self._bot_positions: dict[str, dict] = {}     # ticker → {side, entry_cents, contracts}
        self._stop_cooldown: dict[str, float] = {}    # ticker → timestamp of last stop-loss exit
        self._ticker_session_pnl: dict[str, float] = {}  # ticker → cumulative realized PnL this session
        self._last_ob_refresh: float = 0.0            # timestamp of last full orderbook refresh
        self._tasks: list[asyncio.Task] = []
        self._log_file = Path(config.logging.trade_log_file)
        self._ensure_trade_log()

    # ── Public interface ─────────────────────────────────────────────────────

    async def start(self):
        log.info("Strategy engine starting...")
        self.state["status"] = "starting"
        self.running = True

        # Start price feed
        await self.price_feed.start()

        # Connect Kalshi
        self._kalshi = KalshiClient(self.cfg.kalshi)
        await self._kalshi.connect()

        # Start Kalshi WS — RSA users auth via upgrade headers, token users via login cmd
        self._ws = KalshiWebSocket(
            config=self.cfg.kalshi,
            token=self._kalshi._token,
            auth_header_factory=self._kalshi.ws_auth_headers if self._kalshi._using_rsa else None,
        )
        self._ws.on("ticker", self._market_cache.handle_ticker)
        self._ws.on("orderbook_delta", self._market_cache.handle_orderbook_delta)
        self._ws.on("fill", self._handle_fill)
        self._tasks.append(asyncio.create_task(self._ws.run(), name="kalshi-ws"))

        # Wait a moment for price feed to warm up
        await asyncio.sleep(3)

        # Start background loops
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

        # Cancel all open orders (paper mode: no-op)
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

    # ── Manual trading interface (called from CLI) ────────────────────────────

    async def manual_trade(self, ticker: str, side: str, amount_usd: float) -> str:
        """Execute a manual trade. Returns result message."""
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

            contracts = int(amount_usd / (price_cents / 100))
            if contracts <= 0:
                return f"Amount ${amount_usd:.2f} too small for price {price_cents}¢"

            return await self._execute_trade(market, side_enum, contracts, price_cents, "manual")
        except Exception as e:
            return f"Error: {e}"

    # ── Background loops ─────────────────────────────────────────────────────

    async def _scan_loop(self):
        """Scan open KXBTC markets and evaluate signals."""
        while self.running:
            try:
                await self._scan_markets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Scan loop error: {e}", exc_info=True)
            await asyncio.sleep(SCAN_INTERVAL)

    async def _orderbook_refresh_loop(self):
        """
        Periodically fetch full orderbook snapshots for all watched markets.
        Repairs any WS delta drift/corruption — runs independently of the scan loop.
        """
        while self.running:
            await asyncio.sleep(OB_REFRESH_INTERVAL)
            if not self._watched_markets or not self._kalshi:
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
                    await self._market_cache.apply_snapshot(
                        ticker, result.best_yes_bid, result.best_yes_ask
                    )
                    refreshed += 1
                log.debug(f"[OB REFRESH] Full snapshot: {refreshed}/{len(tickers)} markets updated")
                self._last_ob_refresh = time.time()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"[OB REFRESH] Loop error: {e}")

    async def _scan_markets(self):
        markets = await self._kalshi.get_markets(
            series_ticker=self.cfg.kalshi.series_ticker,
            status="open",
            limit=20,
        )

        # Subscribe new markets to WS
        new_tickers = [m.ticker for m in markets if m.ticker not in self._watched_markets]
        if new_tickers:
            await self._ws.subscribe(new_tickers, ["orderbook_delta", "ticker"])

        self._watched_markets = {m.ticker: m for m in markets}
        self.state["open_markets"] = [self._market_info(m) for m in markets]
        self.state["last_scan"] = datetime.now(timezone.utc).isoformat()

        if not markets:
            return

        # Evaluate each market
        current_price = self.price_feed.current_price
        annual_vol = self.price_feed.realized_vol()
        bars = self.price_feed.bars
        now_utc = datetime.now(timezone.utc)

        # Fix B: parallel fresh orderbook fetch for all markets with open positions
        position_tickers = [
            m.ticker for m in markets
            if m.ticker in self._bot_positions or
            any(m.ticker in p.positions for p in self._personas)
        ]
        if position_tickers:
            fresh_obs: dict[str, object] = {}
            results = await asyncio.gather(
                *[self._kalshi.get_orderbook(t) for t in position_tickers],
                return_exceptions=True,
            )
            for ticker, result in zip(position_tickers, results):
                if not isinstance(result, Exception):
                    fresh_obs[ticker] = result
        else:
            fresh_obs = {}

        signals_snapshot = {}
        for market in markets:
            # Fix E: compute seconds_remaining from local clock against cached close_time
            # rather than relying on the REST snapshot value (which is 0–3s old by now)
            close_time = market.close_time
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            secs = max(0.0, (close_time - now_utc).total_seconds())

            tradeable = (
                self.cfg.strategy.min_seconds_remaining
                <= secs <=
                self.cfg.strategy.max_seconds_remaining
            )

            # Get live orderbook prices (WS cache)
            yes_bid, yes_ask = await self._market_cache.get_best_prices(market.ticker)
            if yes_bid is None:
                yes_bid = market.yes_bid
            if yes_ask is None:
                yes_ask = market.yes_ask

            # Apply parallel-fetched fresh orderbook if available (Fix B)
            if market.ticker in fresh_obs:
                fresh_ob = fresh_obs[market.ticker]
                if fresh_ob.best_yes_bid is not None:
                    yes_bid = fresh_ob.best_yes_bid
                if fresh_ob.best_yes_ask is not None:
                    yes_ask = fresh_ob.best_yes_ask

            # Run ensemble — always compute for all markets so signals never go blank
            output = self.ensemble.predict(
                ticker=market.ticker,
                strike=market.strike_price,
                current_price=current_price,
                seconds_remaining=secs,
                annual_vol=annual_vol,
                bars=bars,
                kalshi_yes_bid=yes_bid,
                kalshi_yes_ask=yes_ask,
                min_edge=self.cfg.models.min_edge,
                min_confidence=self.cfg.models.min_confidence,
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
                "kalshi_price": round((yes_bid or 0 + yes_ask or 0) / 2, 1),
                "tradeable": tradeable,
            }

            # Evaluate exit/flip for any existing position in this market
            should_flip = False
            if market.ticker in self._bot_positions:
                should_flip = await self._evaluate_exits(market, yes_bid, yes_ask, output)

            # New entry — only if no existing position (or just flipped) and within window
            if (tradeable
                    and self.cfg.strategy.auto_trade
                    and output.recommended_side
                    and (market.ticker not in self._bot_positions or should_flip)):
                await self._consider_trade(market, output, yes_bid, yes_ask, annual_vol, bars)

            # ── Persona dispatch ──────────────────────────────────────────
            if self.cfg.strategy.auto_trade:
                ob_info = {
                    "yes_bid": yes_bid,
                    "yes_ask": yes_ask,
                }
                mkt_info = {
                    "seconds_left": round(secs),
                    "volume": market.volume,
                    "ticker": market.ticker,
                }
                try:
                    balance = await self._kalshi.get_balance()
                    bankroll = balance.available_usd
                except Exception:
                    bankroll = self.cfg.risk.max_trade_usd * 10

                for persona in self._personas:
                    try:
                        actions = persona.evaluate(
                            ticker=market.ticker,
                            market=mkt_info,
                            orderbook=ob_info,
                            output=output,
                            bankroll_usd=bankroll,
                            engine_state=self.state,
                        )
                        for action in actions:
                            await self._execute_persona_action(action)
                    except Exception as e:
                        log.error(f"Persona {persona.name} error on {market.ticker}: {e}", exc_info=True)

        # Only overwrite signals if we got fresh data — avoids blank panel on brief API gaps
        if signals_snapshot:
            self.state["signals"] = signals_snapshot

    async def _consider_trade(
        self,
        market: Market,
        output: ModelOutput,
        yes_bid: Optional[float],
        yes_ask: Optional[float],
        annual_vol: float,
        bars: list,
    ):
        """Evaluate whether to execute a trade based on model output."""
        ticker = market.ticker
        side = Side(output.recommended_side)

        # Always fetch a fresh orderbook just before entry to avoid stale WS prices
        try:
            ob = await self._kalshi.get_orderbook(ticker)
            fresh_bid = ob.best_yes_bid or yes_bid
            fresh_ask = ob.best_yes_ask or yes_ask
        except Exception:
            fresh_bid, fresh_ask = yes_bid, yes_ask

        # Determine entry price from fresh data
        # Use raw market price for Kelly sizing (slippage would kill thin edges)
        # Add slippage only for the actual order placement
        slip = self.cfg.kalshi.limit_slippage_cents
        if side == Side.YES:
            if fresh_ask is None:
                log.info(f"[SKIP] {ticker}: no YES ask available")
                return
            kelly_price = int(fresh_ask)
            order_price = min(99, kelly_price + slip)
            prob_win = output.prob_yes
        else:
            if fresh_bid is None:
                log.info(f"[SKIP] {ticker}: no YES bid available")
                return
            kelly_price = int(100 - fresh_bid)
            order_price = min(99, kelly_price + slip)
            prob_win = output.prob_no

        kelly_price = max(1, min(99, kelly_price))
        order_price = max(1, min(99, order_price))

        # Skip contracts already priced at pennies — market has made up its mind
        min_price = getattr(self.cfg.strategy, "min_entry_price_cents", 10)
        if kelly_price < min_price:
            log.debug(f"[SKIP] {ticker}: price={kelly_price}¢ below min_entry={min_price}¢")
            return

        # Cooldown after stop-loss — prevents re-entering a losing market every 5s
        cooldown = getattr(self.cfg.strategy, "stop_loss_cooldown_seconds", 120)
        last_stop = self._stop_cooldown.get(ticker, 0)
        if time.time() - last_stop < cooldown:
            remaining = int(cooldown - (time.time() - last_stop))
            log.debug(f"[SKIP] {ticker}: stop-loss cooldown {remaining}s remaining")
            return

        # Per-candle loss brake — skip auto entries if this ticker has lost too much this session
        per_candle_brake = getattr(self.cfg.strategy, "per_candle_max_loss_usd", 8.00)
        ticker_loss = self._ticker_session_pnl.get(ticker, 0.0)
        if ticker_loss <= -per_candle_brake:
            log.debug(f"[SKIP] {ticker}: candle loss brake (${ticker_loss:.2f} cumulative loss)")
            return

        # Get bankroll
        try:
            balance = await self._kalshi.get_balance()
            bankroll = balance.available_usd
        except Exception:
            bankroll = self.cfg.risk.max_trade_usd * 10

        # Size position using raw price (not slippage-inflated)
        contracts = size_position(
            prob_win=prob_win,
            price_cents=kelly_price,
            bankroll_usd=bankroll,
            max_trade_usd=self.cfg.risk.max_trade_usd,
            min_trade_usd=self.cfg.risk.min_trade_usd,
            kelly_fraction=self.cfg.risk.kelly_fraction,
        )
        if contracts <= 0:
            log.info(
                f"[SKIP] {ticker}: Kelly sizing returned 0 contracts "
                f"(p={prob_win:.1%} kelly_price={kelly_price}¢ bankroll=${bankroll:.2f})"
            )
            return

        # Cost cap for cheap entries — Kelly at penny prices explodes contract counts;
        # clamp the dollar exposure regardless of what Kelly returns
        cheap_threshold = getattr(self.cfg.strategy, "cheap_entry_threshold_cents", 25)
        max_cost_cheap = getattr(self.cfg.strategy, "max_cost_cheap_entry_usd", 4.00)
        if kelly_price < cheap_threshold:
            max_contracts_cheap = int(max_cost_cheap / (kelly_price / 100))
            if contracts > max_contracts_cheap:
                log.debug(
                    f"[CAP] {ticker}: cheap entry {kelly_price}¢ — "
                    f"contracts {contracts} → {max_contracts_cheap} (cap ${max_cost_cheap:.2f})"
                )
                contracts = max_contracts_cheap
        if contracts <= 0:
            return

        price_cents = order_price  # use slippage price for actual order

        # Risk check
        allowed, reason = self.risk.check_trade(
            ticker=ticker,
            side=side.value,
            contracts=contracts,
            price_cents=price_cents,
            bankroll_usd=bankroll,
        )
        if not allowed:
            log.info(f"[RISK BLOCK] {ticker}: {reason}")
            return

        # Log bet parameters
        log_bet_info(
            market.ticker, side.value, prob_win, price_cents,
            contracts, bankroll, self.cfg.risk.kelly_fraction,
        )

        # Store features for ML training later
        if len(bars) >= 5:
            features = self.ensemble._build_ml_features(
                market.strike_price,
                self.price_feed.current_price,
                market.seconds_remaining,
                annual_vol,
                bars,
            )
            self._position_features[market.ticker] = features

        await self._execute_trade(market, side, contracts, price_cents, "auto")

    async def _execute_persona_action(self, action: Action):
        """Route a persona action to the appropriate Kalshi API call."""
        paper = self.cfg.strategy.paper_trade
        label = "[PAPER]" if paper else "[LIVE]"
        persona_tag = action.persona.upper()[:3]

        if action.action_type == "buy":
            side_enum = Side(action.side)

            # Risk check (persona-aware)
            allowed, reason = self.risk.check_trade(
                ticker=action.ticker,
                side=action.side,
                contracts=action.contracts,
                price_cents=action.price_cents,
                bankroll_usd=self.cfg.risk.max_trade_usd * 10,
                persona=action.persona,
            )
            if not allowed:
                log.info(f"[{persona_tag}] [RISK BLOCK] {action.ticker}: {reason}")
                return

            log.info(
                f"{label} [{persona_tag}] BUY: {action.ticker} {action.side.upper()} "
                f"x{action.contracts} @ {action.price_cents}¢ | {action.reason}"
            )

            trade_id = f"T{uuid.uuid4().hex[:8].upper()}"

            if not paper:
                try:
                    order = await self._kalshi.place_order(
                        ticker=action.ticker,
                        side=side_enum,
                        contracts=action.contracts,
                        price_cents=action.price_cents,
                        order_type=OrderType.LIMIT,
                        client_order_id=f"btc15-{action.persona}-{uuid.uuid4().hex[:6]}",
                        post_only=action.post_only,
                        self_trade_prevention=(
                            SelfTradePrevention(action.self_trade_prevention)
                            if action.self_trade_prevention else None
                        ),
                    )
                    self._open_orders[order.order_id] = order
                    persona_obj = self._get_persona(action.persona)
                    if persona_obj:
                        if action.post_only:
                            # Resting order — track open order ID; fill recorded via WS fill event
                            persona_obj.record_order(order.order_id, action.ticker, action.side, action.price_cents, action.contracts)
                        else:
                            # Aggressive order — assume immediate fill
                            persona_obj.record_fill(action.ticker, action.side, action.contracts, action.price_cents, trade_id)
                except KalshiAPIError as e:
                    log.error(f"[{persona_tag}] Order failed: {e}")
                    return
            else:
                # Paper trade
                persona_obj = self._get_persona(action.persona)
                if persona_obj:
                    persona_obj.record_fill(action.ticker, action.side, action.contracts, action.price_cents, trade_id)
                    if action.post_only:
                        # Track resting order in paper mode so amend/cancel logic works
                        # (avoids stacking new orders every scan cycle)
                        persona_obj.record_order(trade_id, action.ticker, action.side, action.price_cents, action.contracts)

            self.risk.record_open(action.ticker, action.side, action.contracts, action.price_cents, persona=action.persona)
            self._log_trade(action.ticker, action.side, action.contracts, action.price_cents, f"{action.persona}/{action.reason}", trade_id)

            trade_info = {
                "ticker": action.ticker,
                "side": action.side,
                "contracts": action.contracts,
                "price_cents": action.price_cents,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "source": f"{action.persona}",
                "status": "open",
                "trade_id": trade_id,
                "session": self._session_label,
            }
            self.state["recent_trades"].insert(0, trade_info)
            self.state["recent_trades"] = self.state["recent_trades"][:50]

        elif action.action_type == "sell":
            log.info(
                f"{label} [{persona_tag}] SELL: {action.ticker} {action.side.upper()} "
                f"x{action.contracts} @ {action.price_cents}¢ | {action.reason}"
            )
            if not paper and self._kalshi:
                try:
                    await self._kalshi.sell_position(
                        action.ticker, Side(action.side), action.contracts, action.price_cents,
                    )
                except Exception as e:
                    log.error(f"[{persona_tag}] Exit order failed: {e}")
                    return

            persona_obj = self._get_persona(action.persona)
            pnl = 0.0
            trade_id = ""
            if persona_obj and action.ticker in persona_obj.positions:
                # Find the matching position by side
                for pos in persona_obj.positions[action.ticker]:
                    if pos["side"] == action.side:
                        pnl = (action.price_cents - pos["entry_cents"]) * action.contracts / 100
                        trade_id = pos.get("trade_id", "")
                        break
                persona_obj.record_exit(action.ticker, pnl, side=action.side)
                self.risk.record_close(action.ticker, won=(pnl > 0), pnl=pnl, persona=action.persona)

            self._log_trade(action.ticker, f"{action.side}_exit", action.contracts, action.price_cents, f"{action.persona}/{action.reason}", trade_id)
            exit_info = {
                "ticker": action.ticker,
                "side": f"{action.side}_exit",
                "contracts": action.contracts,
                "price_cents": action.price_cents,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "source": f"{action.persona}",
                "pnl": round(pnl, 3),
                "trade_id": trade_id,
                "session": self._session_label,
            }
            self.state["recent_trades"].insert(0, exit_info)
            self.state["recent_trades"] = self.state["recent_trades"][:50]

        elif action.action_type == "amend":
            persona_obj = self._get_persona(action.persona)
            if not paper and self._kalshi and action.order_id:
                try:
                    log.debug(f"[{persona_tag}] AMEND: {action.order_id} → {action.price_cents}¢")
                    await self._kalshi.amend_order(
                        action.order_id,
                        price_cents=action.price_cents,
                        side=Side(action.side) if action.side else None,
                    )
                    if persona_obj and action.order_id in persona_obj.resting_orders:
                        persona_obj.resting_orders[action.order_id]["price"] = action.price_cents
                except Exception as e:
                    log.debug(f"[{persona_tag}] Amend failed: {e}")
            elif paper and persona_obj and action.order_id in persona_obj.resting_orders:
                # Keep resting order price in sync in paper mode so reprice detection works
                persona_obj.resting_orders[action.order_id]["price"] = action.price_cents

        elif action.action_type == "cancel":
            persona_obj = self._get_persona(action.persona)
            if not paper and self._kalshi and action.order_id:
                try:
                    log.debug(f"[{persona_tag}] CANCEL: {action.order_id}")
                    await self._kalshi.cancel_order(action.order_id)
                    if persona_obj:
                        persona_obj.remove_order(action.order_id)
                except Exception as e:
                    log.debug(f"[{persona_tag}] Cancel failed: {e}")
            elif paper and persona_obj and action.order_id:
                persona_obj.remove_order(action.order_id)

        elif action.action_type == "batch_buy":
            log.info(
                f"{label} [{persona_tag}] BATCH BUY: {action.ticker} | "
                f"{len(action.batch_orders)} orders | {action.reason}"
            )
            if not paper and self._kalshi:
                try:
                    orders = await self._kalshi.batch_place_orders(action.batch_orders)
                    for o in orders:
                        self._open_orders[o.order_id] = o
                except KalshiAPIError as e:
                    log.error(f"[{persona_tag}] Batch order failed: {e}")
                    return

            persona_obj = self._get_persona(action.persona)
            batch_id = f"T{uuid.uuid4().hex[:8].upper()}"  # shared ID for both legs of arb
            for bo in action.batch_orders:
                side_str = bo["side"]
                price = bo.get("yes_price") or bo.get("no_price", 50)
                trade_id = f"{batch_id}-{side_str[:1].upper()}"
                self.risk.record_open(action.ticker, side_str, bo["count"], price, persona=action.persona)
                self._log_trade(action.ticker, side_str, bo["count"], price, f"{action.persona}/{action.reason}", trade_id)
                if persona_obj:
                    persona_obj.record_fill(action.ticker, side_str, bo["count"], price, trade_id)

                trade_info = {
                    "ticker": action.ticker,
                    "side": side_str,
                    "contracts": bo["count"],
                    "price_cents": price,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "source": f"{action.persona}",
                    "status": "open",
                    "trade_id": trade_id,
                    "session": self._session_label,
                }
                self.state["recent_trades"].insert(0, trade_info)
            self.state["recent_trades"] = self.state["recent_trades"][:50]

    async def _handle_fill(self, msg: dict):
        """Route Kalshi WS fill events to the owning persona's inventory."""
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

        for persona in self._personas:
            if order_id not in persona.resting_orders:
                continue
            trade_id = f"T{uuid.uuid4().hex[:8].upper()}"
            log.info(
                f"[FILL] {persona.tag} {ticker} {side_str.upper()} "
                f"x{count} @ {price}¢ | order={order_id[:8]}"
            )
            persona.record_fill(ticker, side_str, count, price, trade_id)
            self.risk.record_open(ticker, side_str, count, price, persona=persona.name)
            self._log_trade(ticker, side_str, count, price, f"{persona.name}/fill", trade_id)
            # Remove from resting if fully filled
            resting = persona.resting_orders.get(order_id, {})
            remaining = resting.get("contracts", 0) - count
            if remaining <= 0:
                persona.remove_order(order_id)
            else:
                persona.resting_orders[order_id]["contracts"] = remaining
            break

    def _get_persona(self, name: str):
        for p in self._personas:
            if p.name == name:
                return p
        return None

    async def _execute_trade(
        self,
        market: Market,
        side: Side,
        contracts: int,
        price_cents: int,
        source: str,
    ) -> str:
        paper = self.cfg.strategy.paper_trade
        label = "[PAPER]" if paper else "[LIVE]"

        log.info(
            f"{label} EXECUTING: {market.ticker} {side.value.upper()} "
            f"x{contracts} @ {price_cents}¢ (${contracts * price_cents / 100:.2f}) | src={source}"
        )

        trade_id = f"T{uuid.uuid4().hex[:8].upper()}"

        if not paper:
            try:
                order_type = OrderType(self.cfg.kalshi.order_type)
                order = await self._kalshi.place_order(
                    ticker=market.ticker,
                    side=side,
                    contracts=contracts,
                    price_cents=price_cents,
                    order_type=order_type,
                    client_order_id=f"btc15-{uuid.uuid4().hex[:8]}",
                )
                self._open_orders[order.order_id] = order
                self.risk.record_open(market.ticker, side.value, contracts, price_cents)
                self._log_trade(market.ticker, side.value, contracts, price_cents, source, trade_id)
                self._bot_positions[market.ticker] = {
                    "side": side.value, "entry_cents": price_cents,
                    "contracts": contracts, "trade_id": trade_id,
                }
                return f"Order placed: {order.order_id} ({order.status.value})"
            except KalshiAPIError as e:
                log.error(f"Order failed: {e}")
                return f"Order failed: {e}"
        else:
            # Paper trade
            self.risk.record_open(market.ticker, side.value, contracts, price_cents)
            self._log_trade(market.ticker, side.value, contracts, price_cents, f"{source}/paper", trade_id)
            self._bot_positions[market.ticker] = {
                "side": side.value, "entry_cents": price_cents,
                "contracts": contracts, "trade_id": trade_id,
            }
            trade_info = {
                "ticker": market.ticker,
                "side": side.value,
                "contracts": contracts,
                "price_cents": price_cents,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "source": source,
                "status": "open",
                "trade_id": trade_id,
                "session": self._session_label,
            }
            self.state["recent_trades"].insert(0, trade_info)
            self.state["recent_trades"] = self.state["recent_trades"][:50]
            return f"Paper trade opened: {market.ticker} {side.value.upper()} x{contracts}"

    async def _evaluate_exits(
        self,
        market: Market,
        yes_bid: Optional[float],
        yes_ask: Optional[float],
        output,
    ) -> bool:
        """
        Check if an open position should be exited early.
        Returns True if the position was exited AND the signal flipped
        (caller should then enter the opposite side).
        """
        ticker = market.ticker
        pos = self._bot_positions.get(ticker)
        if not pos:
            return False

        side        = pos["side"]
        entry_cents = pos["entry_cents"]
        secs        = market.seconds_remaining

        # Current bid for our side (what we'd receive if we sold now)
        if side == "yes":
            current_bid = float(yes_bid or 0)
        else:
            current_bid = max(0.0, 100.0 - float(yes_ask or 100))

        if entry_cents <= 0 or current_bid <= 0:
            return False

        pnl_pct = (current_bid - entry_cents) / entry_cents

        # Update trailing peak
        peak = max(pnl_pct, pos.get("peak_pnl_pct", 0.0))
        pos["peak_pnl_pct"] = peak

        # ── Exit conditions ────────────────────────────────────────────
        exit_reason = None

        if pnl_pct >= self.cfg.strategy.take_profit_pct:
            exit_reason = f"take_profit ({pnl_pct:+.1%})"

        elif pnl_pct <= -self.cfg.strategy.stop_loss_pct:
            exit_reason = f"stop_loss ({pnl_pct:+.1%})"

        elif secs < self.cfg.strategy.lock_profit_seconds and pnl_pct > 0.10:
            exit_reason = f"time_lock ({pnl_pct:+.1%}, {secs:.0f}s left)"

        else:
            trail_activate = getattr(self.cfg.strategy, "trail_activate_pct", 0.20)
            trail_retrace = getattr(self.cfg.strategy, "trail_retracement_pct", 0.50)
            if peak >= trail_activate:
                trail_floor = peak * (1.0 - trail_retrace)
                if pnl_pct < trail_floor:
                    exit_reason = f"trail_stop (peak={peak:+.1%} → floor={trail_floor:+.1%} now={pnl_pct:+.1%})"

        # ── Flip condition ─────────────────────────────────────────────
        should_flip = False
        if output.recommended_side and output.recommended_side != side:
            opp_edge = float(
                output.edge_no if side == "yes" else output.edge_yes or 0
            )
            if opp_edge >= self.cfg.strategy.flip_min_edge:
                exit_reason = f"flip→{output.recommended_side} (edge {opp_edge:+.1%})"
                should_flip = True

        if exit_reason:
            await self._exit_position(ticker, pos, int(current_bid), exit_reason)

        return should_flip and exit_reason is not None

    async def _exit_position(
        self,
        ticker: str,
        pos: dict,
        exit_cents: int,
        reason: str,
    ):
        """Close a bot-opened position (paper or real)."""
        side      = pos["side"]
        contracts = pos["contracts"]
        entry_c   = pos["entry_cents"]
        pnl_usd   = (exit_cents - entry_c) * contracts / 100
        won       = pnl_usd > 0

        label = "[PAPER]" if self.cfg.strategy.paper_trade else "[LIVE]"
        log.info(
            f"{label} EXIT: {ticker} {side.upper()} x{contracts} | "
            f"{entry_c}¢ → {exit_cents}¢ | pnl=${pnl_usd:+.2f} | {reason}"
        )

        if not self.cfg.strategy.paper_trade and self._kalshi:
            try:
                await self._kalshi.sell_position(
                    ticker, Side(side), contracts, exit_cents
                )
            except Exception as e:
                log.error(f"Exit order failed for {ticker}: {e}")
                return  # keep position in tracker; will retry next scan

        trade_id = pos.get("trade_id", "")
        self.risk.record_close(ticker, won=won, pnl=pnl_usd)
        self._log_trade(ticker, f"{side}_exit", contracts, exit_cents, f"exit/{reason}", trade_id)
        self._bot_positions.pop(ticker, None)

        # Track cumulative realized PnL per ticker this session (for per-candle loss brake)
        self._ticker_session_pnl[ticker] = self._ticker_session_pnl.get(ticker, 0.0) + pnl_usd

        # Cooldown: block re-entry after a stop-loss so the engine doesn't loop
        if "stop_loss" in reason:
            self._stop_cooldown[ticker] = time.time()

        exit_info = {
            "ticker": ticker,
            "side": f"{side}_exit",
            "contracts": contracts,
            "price_cents": exit_cents,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "source": f"exit/{reason}",
            "pnl": round(pnl_usd, 3),
            "trade_id": trade_id,
            "session": self._session_label,
        }
        self.state["recent_trades"].insert(0, exit_info)
        self.state["recent_trades"] = self.state["recent_trades"][:50]

    async def _position_loop(self):
        """Monitor open positions and check for settlement."""
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
            self.risk.state.open_positions = len(api_positions)
        except Exception as e:
            api_positions = []

        # Build display positions from our internal records (reliable cost basis).
        # The Kalshi API avg_price / market_exposure fields are unreliable for display.
        display = []
        def _mtm(ticker: str, side: str, entry_cents: int) -> tuple[float, float]:
            """Mark-to-market using cached market prices. Returns (current_bid, pnl_per_contract)."""
            mkt = self._watched_markets.get(ticker)
            if mkt is None:
                return entry_cents, 0.0
            if side == "yes":
                bid = mkt.yes_bid or entry_cents
            else:
                bid = max(0.0, 100.0 - (mkt.yes_ask or 100.0))
            return bid, bid - entry_cents

        for ticker, pos in self._bot_positions.items():
            bid, _ = _mtm(ticker, pos["side"], pos["entry_cents"])
            cost = round(pos["contracts"] * pos["entry_cents"] / 100, 2)
            value = round(pos["contracts"] * bid / 100, 2)
            display.append({
                "ticker": ticker,
                "side": pos["side"],
                "contracts": pos["contracts"],
                "entry_cents": pos["entry_cents"],
                "cost": cost,
                "value": value,
                "pnl": round(value - cost, 2),
                "source": "auto",
            })

        for persona in self._personas:
            for ticker, entries in persona.positions.items():
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
                        "source": persona.name[:3],
                    })

        self.state["open_positions"] = display

        # ── Settlement detection ──────────────────────────────────────────
        # Check for expired markets — both from API and time-based (paper mode)
        await self._check_settlements()

    async def _check_settlements(self):
        """Detect settled/expired markets and resolve all held positions."""
        now = datetime.now(timezone.utc)
        current_price = self.price_feed.current_price

        # Collect all tickers that have positions (engine + personas)
        all_position_tickers = set(self._bot_positions.keys())
        for persona in self._personas:
            all_position_tickers.update(persona.positions.keys())

        if not all_position_tickers:
            return

        for ticker in list(all_position_tickers):
            cached_market = self._watched_markets.get(ticker)
            result = None

            # Method 1: Try fetching from API (works for live mode, gets actual result)
            try:
                market_data = await self._kalshi.get_market(ticker)
                if market_data.status in (MarketStatus.SETTLED, MarketStatus.FINALIZED) and market_data.result:
                    result = market_data.result
                elif market_data.seconds_remaining > 0:
                    continue  # not expired yet
            except Exception:
                pass

            # Method 2: Time-based settlement for paper mode
            # If close_time has passed and we didn't get a result from API,
            # determine outcome from actual BTC price vs strike
            if result is None and cached_market:
                close_time = cached_market.close_time
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=timezone.utc)
                if now > close_time and current_price > 0 and cached_market.strike_price > 0:
                    result = "yes" if current_price > cached_market.strike_price else "no"
                    log.info(
                        f"[SETTLEMENT] {ticker}: BTC ${current_price:,.2f} vs strike "
                        f"${cached_market.strike_price:,.2f} → {result.upper()}"
                    )
                else:
                    continue  # not expired yet

            if result is None:
                continue

            # ── Settle engine positions ────────────────────────────────────
            if ticker in self._bot_positions:
                pos = self._bot_positions[ticker]
                settlement_cents = 100 if result == pos["side"] else 0
                pnl = (settlement_cents - pos["entry_cents"]) * pos["contracts"] / 100
                won = pnl > 0

                label = "[PAPER]" if self.cfg.strategy.paper_trade else "[LIVE]"
                log.info(
                    f"{label} SETTLED: {ticker} {pos['side'].upper()} x{pos['contracts']} | "
                    f"entry={pos['entry_cents']}¢ → {settlement_cents}¢ | "
                    f"result={result.upper()} | pnl=${pnl:+.2f}"
                )

                trade_id = pos.get("trade_id", "")
                self.risk.record_close(ticker, won=won, pnl=pnl)
                self._log_trade(ticker, f"{pos['side']}_settled", pos["contracts"], settlement_cents, f"settled/{result}", trade_id)
                self._bot_positions.pop(ticker, None)
                self._ticker_session_pnl[ticker] = self._ticker_session_pnl.get(ticker, 0.0) + pnl

                # Add to trade log in dashboard
                settle_info = {
                    "ticker": ticker,
                    "side": f"{pos['side']}_settled",
                    "contracts": pos["contracts"],
                    "price_cents": settlement_cents,
                    "entry_time": datetime.now(timezone.utc).isoformat(),
                    "source": f"settled/{result}",
                    "pnl": round(pnl, 3),
                    "trade_id": trade_id,
                }
                self.state["recent_trades"].insert(0, settle_info)

            # ── Settle persona positions ───────────────────────────────────
            for persona in self._personas:
                if ticker in persona.positions:
                    # Settle each side (YES/NO) independently
                    for pos in list(persona.positions[ticker]):
                        settlement_cents = 100 if result == pos["side"] else 0
                        pnl = (settlement_cents - pos["entry_cents"]) * pos["contracts"] / 100
                        won = pnl > 0

                        label = "[PAPER]" if self.cfg.strategy.paper_trade else "[LIVE]"
                        log.info(
                            f"{label} [{persona.tag}] SETTLED: {ticker} {pos['side'].upper()} "
                            f"x{pos['contracts']} | entry={pos['entry_cents']}¢ → {settlement_cents}¢ | "
                            f"result={result.upper()} | pnl=${pnl:+.2f}"
                        )

                        trade_id = pos.get("trade_id", "")
                        persona.record_exit(ticker, pnl, side=pos["side"])
                        self.risk.record_close(ticker, won=won, pnl=pnl, persona=persona.name)
                        self._log_trade(
                            ticker, f"{pos['side']}_settled", pos["contracts"],
                            settlement_cents, f"{persona.name}/settled/{result}", trade_id,
                        )

                        settle_info = {
                            "ticker": ticker,
                            "side": f"{pos['side']}_settled",
                            "contracts": pos["contracts"],
                            "price_cents": settlement_cents,
                            "entry_time": datetime.now(timezone.utc).isoformat(),
                            "source": f"{persona.name}/settled",
                            "pnl": round(pnl, 3),
                            "trade_id": trade_id,
                        }
                        self.state["recent_trades"].insert(0, settle_info)

            # Clean up also any resting scalper orders for this ticker
            for persona in self._personas:
                for oid in list(persona.resting_orders.keys()):
                    if persona.resting_orders[oid].get("ticker") == ticker:
                        persona.remove_order(oid)

            # Collect ML training data
            if ticker in self._position_features:
                outcome = 1 if result == "yes" else 0
                collect_sample(self._position_features.pop(ticker), outcome)

            # Remove from watched markets (it's settled)
            self._watched_markets.pop(ticker, None)

            self.state["recent_trades"] = self.state["recent_trades"][:50]

    async def _state_updater(self):
        """Update dashboard state variables."""
        while self.running:
            try:
                self.state["current_price"] = self.price_feed.current_price
                self.state["feed_age_sec"] = round(self.price_feed.feed_age_sec(), 1)
                risk_summary = self.risk.summary()
                self.state["risk"] = risk_summary
                self.state["personas"] = {p.name: p.summary() for p in self._personas}
                # Sample P&L history every 10s (keep last 500 points ~ 80 min)
                if int(time.time()) % 10 == 0:
                    pnl_now = round(risk_summary.get("daily_pnl", 0.0), 3)
                    ts_now = datetime.now(timezone.utc).strftime("%H:%M")
                    history = self.state["pnl_history"]
                    if not history or history[-1][1] != pnl_now:
                        history.append((ts_now, pnl_now))
                        self.state["pnl_history"] = history[-500:]
                # Refresh balance every 30s
                if self._kalshi and int(time.time()) % 30 == 0:
                    try:
                        bal = await self._kalshi.get_balance()
                        self.state["balance"] = {
                            "available": round(bal.available_usd, 2),
                            "portfolio": round(bal.portfolio_usd, 2),
                        }
                    except Exception:
                        pass
            except Exception as e:
                log.debug(f"State updater error: {e}")
            await asyncio.sleep(1)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _market_info(self, m: Market) -> dict:
        return {
            "ticker": m.ticker,
            "strike": m.strike_price,
            "yes_bid": m.yes_bid,
            "yes_ask": m.yes_ask,
            "volume": m.volume,
            "seconds_left": round(m.seconds_remaining),
            "status": m.status.value,
        }

    def _ensure_trade_log(self):
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_file.exists():
            with open(self._log_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trade_id", "timestamp", "ticker", "side", "contracts",
                    "price_cents", "cost_usd", "source", "mode", "session",
                ])

    def _log_trade(self, ticker, side, contracts, price_cents, source, trade_id: str = ""):
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
