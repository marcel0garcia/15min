"""Core engine (v3) — the async loop.

Responsibilities, in order:

  1. Ingest    Kalshi market list + WS orderbook/trade stream; venue WS
               feeds -> BRTI reconstruction -> tick buffer (the settlement
               instrument's own price history).
  2. Price     One MarketQuote per open market per scan (core/pricer).
  3. Decide    Policy evaluates exits first, then entries (core/policy).
  4. Execute   PaperBroker in paper mode; KalshiClient in live mode.
  5. Settle    Kalshi's official result is authoritative. Paper falls back
               to our own final-minute BRTI TWAP — the real instrument —
               and NEVER to a drained orderbook.
  6. Record    Every market-scan observation to decisions.jsonl with BOTH
               p_model and p_market, so core/score.py can measure us
               against the market offline.
  7. Surface   Populate the state dict the existing Rich dashboard reads.

What is deliberately NOT here: market making, arbitrage, pyramiding,
GTC laddering, escalation, cool-offs. Those are experiments to be
re-introduced one at a time, each with its own measured justification.
"""
from __future__ import annotations

import asyncio
import csv
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from btc15.config import AppConfig
from btc15.core.fees import FeeCalibrator, taker_fee_usd
from btc15.core.paper import PaperBroker, PaperBrokerConfig
from btc15.core.policy import Policy, PolicyConfig, Position, slice_key
from btc15.core.pricer import MarketQuote, SliceGrid, price_band, quote_market
from btc15.core.sigma import SigmaConfig, blended_sigma
from btc15.feeds.brti_feed import BRTIPriceFeed
from btc15.kalshi.client import KalshiClient
from btc15.kalshi.models import MarketStatus, Order, OrderType, Side, TimeInForce
from btc15.kalshi.ws_client import KalshiWebSocket, MarketDataCache
from btc15.recording import SessionRecorder
from btc15.recording.brti import reconstruct as brti_reconstruct
from btc15.recording.kalshi_tap import KalshiRawTap
from btc15.recording.venues import build_venue_tasks

log = logging.getLogger(__name__)

# Fallbacks only. The live values come from CoreConfig (config.yaml) so the
# sweep and the agent can vary cadence without editing source.
SCAN_INTERVAL = 1.0
OB_REFRESH_INTERVAL = 4.0
SETTLEMENT_CHECK_INTERVAL = 5.0
BRTI_HZ = 4.0
VENUE_STALENESS_SEC = 5.0

# Shared with btc15/research/replay.py — see btc15/core/engine_constants.py.
# Replay MUST use the same history window and the same book depth, or it is
# quietly measuring a different bot.
from btc15.core.engine_constants import (  # noqa: E402
    DECISION_BOOK_LEVELS, TICK_HISTORY_SEC,
)


class _DashboardLogHandler(logging.Handler):
    """Feeds the dashboard's event-log panel."""

    _KEYWORDS = ("ENTER", "EXIT", "SETTLED", "HALT", "FILL", "REJECT", "ERROR", "WARN")

    def __init__(self, state: dict):
        super().__init__()
        self._state = state

    def emit(self, record: logging.LogRecord):
        try:
            msg = record.getMessage()
            if record.levelno < logging.WARNING and not any(k in msg for k in self._KEYWORDS):
                return
            entry = {
                "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                "level": record.levelname,
                "msg": msg,
            }
            log_list = self._state.setdefault("event_log", [])
            log_list.append(entry)
            self._state["event_log"] = log_list[-100:]
        except Exception:
            pass


class CoreEngine:
    """Async engine. `await engine.start()`, then `await engine.stop()`.
    Exposes `.state` for the Rich dashboard and `.running` for its loop."""

    def __init__(self, config: AppConfig):
        self.cfg = config
        core = config.core

        self.price_feed = BRTIPriceFeed(
            bar_interval_sec=config.feeds.bar_interval_sec,
            lookback_bars=config.feeds.lookback_bars,
        )
        self.grid = SliceGrid(
            phase_early_sec=core.phase_early_sec,
            phase_mid_sec=core.phase_mid_sec,
            phase_prime_sec=core.phase_prime_sec,
            band_extreme_cents=core.band_extreme_cents,
            band_outer_cents=core.band_outer_cents,
        )
        self.sigma_cfg = SigmaConfig(
            fast_sec=core.sigma_fast_sec,
            slow_sec=core.sigma_slow_sec,
            fast_weight=core.sigma_fast_weight,
            floor=core.sigma_floor,
            ceiling=core.sigma_ceiling,
            min_samples=core.sigma_min_samples,
            scale=core.sigma_scale,
            sample_sec=core.sigma_sample_sec,
        )
        self.scan_interval = core.scan_interval_sec
        self.ob_refresh_interval = core.ob_refresh_interval_sec
        self.settlement_check_interval = core.settlement_check_interval_sec
        self.brti_hz = core.brti_hz
        self.venue_staleness_sec = core.venue_staleness_sec

        self.policy = Policy(PolicyConfig(
            min_seconds=core.min_seconds,
            max_seconds=core.max_seconds,
            enabled_slices=set(core.enabled_slices or []),
            ev_margin_cents=core.ev_margin_cents,
            min_confidence=core.min_confidence,
            max_spread_cents=core.max_spread_cents,
            kelly_fraction=core.kelly_fraction,
            max_single_trade_usd=core.max_single_trade_usd,
            min_single_trade_usd=core.min_single_trade_usd,
            max_open_positions=core.max_open_positions,
            max_per_market_usd=core.max_per_market_usd,
            daily_loss_limit_usd=core.daily_loss_limit_usd,
            exit_flip_min_ev_cents=core.exit_flip_min_ev_cents,
            exit_min_seconds=core.exit_min_seconds,
            slippage_cents=core.slippage_cents,
            fee_rate=core.fee_rate,
            fee_multiplier=core.fee_multiplier,
            max_entries_per_market=core.max_entries_per_market,
            entry_cooldown_sec=core.entry_cooldown_sec,
            warmup_sec=core.warmup_sec,
            reject_clamped_sigma=core.reject_clamped_sigma,
            grid=self.grid,
        ))
        # Verifies our fee model against what Kalshi actually charges on
        # live fills. A silent fee-model error is the quietest way for a
        # bot with real edge to lose money.
        self.fee_calibrator = FeeCalibrator(
            rate=core.fee_rate, multiplier=core.fee_multiplier,
        )
        self.broker = PaperBroker(PaperBrokerConfig(
            starting_cash_usd=core.paper_starting_cash_usd,
            adverse_cents=core.paper_adverse_cents,
            fee_rate=core.fee_rate,
            fee_multiplier=core.fee_multiplier,
        ))

        self._kalshi: Optional[KalshiClient] = None
        self._ws: Optional[KalshiWebSocket] = None
        self._market_cache = MarketDataCache()
        self._watched: dict[str, object] = {}
        self._tasks: list[asyncio.Task] = []
        self._venue_ws_list: list = []
        self._settled: dict[str, float] = {}
        self._pending_settlement: dict[str, int] = {}
        self.running = False

        now = datetime.now(timezone.utc)
        self._session_label = now.strftime("%d%b%H:%M").upper()
        self._recorder = SessionRecorder(config, self._session_label)
        self._kalshi_tap = KalshiRawTap(self._recorder)

        self._log_file = Path(config.logging.trade_log_file)
        self._ensure_trade_log()

        self.state: dict = {
            "status": "idle",
            "current_price": 0.0,
            "feed_age_sec": 0.0,
            "open_markets": [],
            "signals": {},
            "open_positions": [],
            "recent_trades": [],
            "balance": None,
            "session_start_balance": round(core.paper_starting_cash_usd, 2),
            "risk": {},
            "last_scan": None,
            "paper_mode": config.strategy.paper_trade,
            "auto_trade": config.strategy.auto_trade,
            "production_brain": "fair_value",
            "session_start": now.isoformat(),
            "event_log": [],
            "btc_tape": deque(maxlen=80),
            "recon_brti": None,
            "venue_status": {},
            "fair_value": None,
            "z_score": None,
            "sigma_nowcast": None,
            "unrealized_pnl": 0.0,
        }

        self._dash_handler = _DashboardLogHandler(self.state)
        self._dash_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(self._dash_handler)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self):
        log.info("Core engine starting (v3)")
        self.state["status"] = "starting"
        self.running = True

        await self.price_feed.start()
        self.price_feed.feed.on_tick(self._on_btc_tick)

        self._kalshi = KalshiClient(self.cfg.kalshi)
        await self._kalshi.connect()

        self._ws = KalshiWebSocket(
            config=self.cfg.kalshi,
            auth_header_factory=(
                self._kalshi.ws_auth_headers if self._kalshi._using_rsa else None
            ),
        )
        self._ws.on("ticker", self._market_cache.handle_ticker)
        self._ws.on("orderbook_delta", self._market_cache.handle_orderbook_delta)
        self._ws.on("orderbook_snapshot", self._market_cache.handle_orderbook_snapshot)
        self._ws.on("trade", self._market_cache.handle_trade)
        self._market_cache.rest_refresh = self._refresh_orderbook
        self._kalshi_tap.attach(self._ws)
        self._tasks.append(asyncio.create_task(self._ws.run(), name="kalshi-ws"))

        self._venue_ws_list = build_venue_tasks(self.cfg, self._recorder)
        for name, ws in self._venue_ws_list:
            self._tasks.append(asyncio.create_task(ws.run(), name=name))

        if not self.cfg.strategy.paper_trade:
            try:
                bal = await self._kalshi.get_balance()
                self.state["session_start_balance"] = round(bal.available_usd, 2)
                self.broker.cash_usd = bal.available_usd
            except Exception as e:
                log.warning(f"Live balance fetch failed: {e}")

        await asyncio.sleep(1)  # let the WS handshake settle

        self._tasks += [
            asyncio.create_task(self._brti_loop(), name="brti"),
            asyncio.create_task(self._scan_loop(), name="scan"),
            asyncio.create_task(self._settlement_loop(), name="settle"),
            asyncio.create_task(self._orderbook_refresh_loop(), name="ob-refresh"),
            asyncio.create_task(self._state_loop(), name="state"),
        ]
        self.state["status"] = "running"
        log.info(
            f"Core engine running | paper={self.cfg.strategy.paper_trade} "
            f"auto_trade={self.cfg.strategy.auto_trade} "
            f"cash=${self.broker.cash_usd:.2f}"
        )

    async def stop(self):
        log.info("Core engine stopping")
        self.running = False
        self.state["status"] = "stopping"
        try:
            self._recorder.close()
        except Exception:
            pass
        if not self.cfg.strategy.paper_trade and self._kalshi:
            try:
                await self._kalshi.cancel_all_orders()
            except Exception:
                pass
        for _name, ws in self._venue_ws_list:
            try:
                await ws.stop()
            except Exception:
                pass
        for t in self._tasks:
            t.cancel()
        await self.price_feed.stop()
        if self._kalshi:
            await self._kalshi.close()
        self.state["status"] = "stopped"
        log.info(f"Session summary: {self.broker.summary()}")

    # ── Ingest ───────────────────────────────────────────────────────────────

    async def _on_btc_tick(self, price: float, qty: float, ts_ms: int) -> None:
        self.state["btc_tape"].append((ts_ms / 1000.0, price, qty))
        self.state["current_price"] = price

    async def _brti_loop(self) -> None:
        """Reconstruct the consolidated mid from venue top-of-book and push
        it into the price feed. This is the settlement instrument."""
        while self.running:
            try:
                await asyncio.sleep(1.0 / self.brti_hz)
                now = time.time()
                venue_mids: dict[str, float] = {}
                venue_status: dict = {}
                for _name, ws in self._venue_ws_list:
                    vname = ws.name
                    if ws.last_bid is None or ws.last_ask is None:
                        venue_status[vname] = {"connected": False, "age_sec": None}
                        continue
                    age = now - ws.last_ts if ws.last_ts > 0 else None
                    fresh = age is not None and age <= self.venue_staleness_sec
                    mid = (ws.last_bid + ws.last_ask) / 2.0
                    venue_status[vname] = {
                        "connected": True, "bid": ws.last_bid, "ask": ws.last_ask,
                        "mid": round(mid, 2),
                        "age_sec": round(age, 2) if age is not None else None,
                        "fresh": fresh,
                    }
                    if fresh:
                        venue_mids[vname] = mid

                if not venue_mids:
                    self.state["venue_status"] = venue_status
                    continue

                mid, outliers, healthy, reason = brti_reconstruct(venue_mids)
                spread = (
                    round(max(venue_mids.values()) - min(venue_mids.values()), 2)
                    if len(venue_mids) > 1 else 0.0
                )
                self.state["recon_brti"] = {
                    "mid": round(mid, 2) if mid is not None else None,
                    "healthy": healthy, "n_venues": len(venue_mids),
                    "outliers": outliers, "spread": spread, "reason": reason,
                }
                self.state["venue_status"] = venue_status
                if mid is not None and healthy:
                    await self.price_feed.push_brti(mid, now)
                    # Record the series the model consumes, at the rate it
                    # consumes it — see SessionRecorder.write_brti.
                    self._recorder.write_brti(now, mid)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug(f"BRTI loop error: {e}")

    async def _refresh_orderbook(self, ticker: str) -> None:
        try:
            ob = await self._kalshi.get_orderbook(ticker)
            await self._market_cache.apply_snapshot(ticker, ob)
        except Exception as e:
            log.debug(f"OB refresh failed for {ticker}: {e}")

    async def _orderbook_refresh_loop(self) -> None:
        while self.running:
            try:
                await asyncio.sleep(self.ob_refresh_interval)
                for ticker in list(self._watched.keys()):
                    await self._refresh_orderbook(ticker)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug(f"OB refresh loop error: {e}")

    # ── Scan → price → decide → execute ──────────────────────────────────────

    async def _scan_loop(self) -> None:
        while self.running:
            try:
                await self._scan()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Scan error: {e}", exc_info=True)
            await asyncio.sleep(self.scan_interval)

    async def _scan(self) -> None:
        markets = await self._kalshi.get_markets(
            series_ticker=self.cfg.kalshi.series_ticker, status="open", limit=20,
        )
        if not markets:
            return

        new = [m.ticker for m in markets if m.ticker not in self._watched]
        if new:
            await self._ws.subscribe(new, ["orderbook_delta", "ticker", "trade"])
            seeds = await asyncio.gather(
                *[self._kalshi.get_orderbook(t) for t in new], return_exceptions=True
            )
            for ticker, ob in zip(new, seeds):
                if not isinstance(ob, Exception):
                    await self._market_cache.apply_snapshot(ticker, ob)
        self._watched = {m.ticker: m for m in markets}
        self.state["last_scan"] = datetime.now(timezone.utc).isoformat()

        spot = self.price_feed.current_price
        if not spot:
            return

        now_ts = time.time()
        now_utc = datetime.now(timezone.utc)
        ticks = [
            (t.ts, t.price)
            for t in self.price_feed.recent_ticks(seconds=TICK_HISTORY_SEC)
            if t.price > 0
        ]
        nowcast = blended_sigma(ticks, now_ts=now_ts, cfg=self.sigma_cfg)
        self.state["sigma_nowcast"] = round(nowcast.sigma, 4)
        self.state["sigma_raw"] = round(nowcast.sigma_raw, 4)
        self.state["sigma_clamped"] = nowcast.clamped
        if nowcast.clamped:
            # Silent floor-binding is how the model manufactures false
            # certainty at extreme strikes. Surface it.
            log.debug(
                f"sigma clamped: raw={nowcast.sigma_raw:.4f} -> {nowcast.sigma:.4f} "
                f"(floor={self.sigma_cfg.floor} ceiling={self.sigma_cfg.ceiling})"
            )

        # How much BRTI history stands behind sigma. Guards the warm-up case
        # where a floored sigma manufactures false conviction.
        tick_span = (now_ts - ticks[0][0]) if ticks else 0.0

        bankroll = self._bankroll_usd()
        signals: dict = {}
        markets_snapshot: list = []
        first = True

        for market in markets:
            close_time = market.close_time
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            secs = max(0.0, (close_time - now_utc).total_seconds())

            yes_bid, yes_ask = await self._market_cache.get_best_prices(market.ticker)
            if yes_bid is None:
                yes_bid = market.yes_bid
            if yes_ask is None:
                yes_ask = market.yes_ask

            q = quote_market(
                ticker=market.ticker, strike=market.strike_price, spot=spot,
                secs=secs, sigma=nowcast.sigma, ticks=ticks, now_ts=now_ts,
                yes_bid=yes_bid, yes_ask=yes_ask, grid=self.grid,
                sigma_clamped=nowcast.clamped, tick_span_sec=tick_span,
            )

            markets_snapshot.append({
                "ticker": market.ticker, "strike": market.strike_price,
                "yes_bid": yes_bid or None, "yes_ask": yes_ask or None,
                "volume": market.volume, "seconds_left": round(secs),
                "status": market.status.value,
            })

            mid = q.mid_cents
            band = price_band(mid, self.grid) if mid is not None else "?"
            best_edge = 0.0
            if q.market_prob_yes is not None and q.recommended_side:
                p = q.prob_win(q.recommended_side)
                mp = (q.market_prob_yes if q.recommended_side == "yes"
                      else 1.0 - q.market_prob_yes)
                best_edge = p - mp
            signals[market.ticker] = {
                "strike": market.strike_price,
                "seconds_left": round(secs),
                "prob_yes": q.prob_yes,
                "fv_prob_yes": q.prob_yes,
                "fv_confidence": q.confidence,
                "fv_degenerate": q.degenerate,
                "confidence": q.confidence,
                "fv_edge_yes": best_edge if q.recommended_side == "yes" else 0.0,
                "fv_edge_no": best_edge if q.recommended_side == "no" else 0.0,
                "edge_yes": best_edge if q.recommended_side == "yes" else 0.0,
                "edge_no": best_edge if q.recommended_side == "no" else 0.0,
                "fv_signal": self._signal_label(q, best_edge),
                "signal": self._signal_label(q, best_edge),
            }

            if first:
                self.state["fair_value"] = None if q.degenerate else round(q.prob_yes, 4)
                self.state["z_score"] = None if q.degenerate else round(q.z_score, 3)
                first = False

            # Exits before entries — never both on one market in one scan.
            acted = False
            if market.ticker in self.policy.positions:
                ex = self.policy.evaluate_exit(q)
                if ex.kind == "exit":
                    await self._execute_exit(q, ex)
                    acted = True

            # Shadow-evaluate on EVERY scan, trading or not. evaluate_entry
            # is pure, so this costs nothing, and it is the only way a
            # signal-only session learns which gate rejected which market.
            # Before this, signal-only runs wrote reject_gate: null on every
            # row — 528 of 528 in the 19AUG session — which made the
            # recordings useless for answering "why didn't it trade?".
            entry = self.policy.evaluate_entry(q, bankroll, now_ts=now_ts)
            executed = False
            if entry.kind == "enter" and not acted and self.cfg.strategy.auto_trade:
                await self._execute_entry(q, entry)
                executed = True

            self._record_decision(
                q, band, entry, nowcast,
                await self._book_depth(market.ticker),
                executed=executed,
            )

        self.state["signals"] = signals
        self.state["open_markets"] = markets_snapshot

    @staticmethod
    def _signal_label(q: MarketQuote, edge: float) -> str:
        if q.degenerate or q.recommended_side is None:
            return "NEUTRAL"
        strong = edge > 0.10
        side = q.recommended_side.upper()
        return f"{'STRONG' if strong else 'WEAK'} {side}"

    def _bankroll_usd(self) -> float:
        """Cash available for new positions."""
        return max(0.0, self.broker.cash_usd)

    async def _book_depth(self, ticker: str, levels: int = DECISION_BOOK_LEVELS) -> dict:
        """Top `levels` of each side as [[price_cents, qty], ...].

        Stamped into every decision row so the recordings can drive an
        offline fill simulation on their own. Bids descend (best first),
        asks ascend (best first) — the order a taker walks them.
        """
        book = await self._book_for(ticker)
        bids = sorted(book.get("yes_bids", {}).items(), key=lambda kv: -float(kv[0]))
        asks = sorted(book.get("yes_asks", {}).items(), key=lambda kv: float(kv[0]))
        return {
            "yes_bids": [[float(px), float(qty)] for px, qty in bids[:levels] if qty],
            "yes_asks": [[float(px), float(qty)] for px, qty in asks[:levels] if qty],
        }

    async def _book_for(self, ticker: str) -> dict:
        """Raw depth ladder for fill simulation."""
        async with self._market_cache._lock:
            ob = self._market_cache._orderbooks.get(ticker, {})
            return {
                "yes_bids": dict(ob.get("yes_bids") or {}),
                "yes_asks": dict(ob.get("yes_asks") or {}),
            }

    # ── Execution ────────────────────────────────────────────────────────────

    async def _execute_entry(self, q: MarketQuote, d) -> None:
        trade_id = uuid.uuid4().hex[:8]
        if self.cfg.strategy.paper_trade:
            book = await self._book_for(q.ticker)
            fill = self.broker.buy_ioc(
                ticker=q.ticker, side=d.side, contracts=d.contracts,
                limit_cents=d.limit_cents, book=book, trade_id=trade_id,
            )
            if not fill.filled:
                log.info(f"[REJECT] no fill {q.ticker} {d.side.upper()} — {fill.note}")
                return
        else:
            fill = await self._live_buy(q, d, trade_id)
            if fill is None:
                return

        pos = Position(
            ticker=q.ticker, side=d.side, contracts=fill.contracts,
            entry_cents=fill.avg_price_cents,
            cost_usd=abs(fill.cash_delta_usd), fees_usd=fill.fee_usd,
            opened_ts=fill.ts, trade_id=trade_id, strike=q.strike,
        )
        self.policy.record_open(pos)
        log.info(
            f"[ENTER] {q.ticker} {d.side.upper()} x{fill.contracts} @ "
            f"{fill.avg_price_cents:.1f}c fee=${fill.fee_usd:.2f} | {d.reason}"
        )
        self._log_trade(q.ticker, d.side, fill.contracts, fill.avg_price_cents,
                        "core/entry", trade_id)
        self._push_trade(q.ticker, d.side, fill.contracts, fill.avg_price_cents,
                         "core/entry", trade_id, None)

    async def _execute_exit(self, q: MarketQuote, d) -> None:
        pos = self.policy.positions.get(q.ticker)
        if pos is None:
            return
        if self.cfg.strategy.paper_trade:
            book = await self._book_for(q.ticker)
            fill = self.broker.sell_ioc(
                ticker=q.ticker, side=pos.side, contracts=pos.contracts,
                limit_cents=d.limit_cents, book=book,
                entry_cents=pos.entry_cents, entry_fee_usd=pos.fees_usd,
                trade_id=pos.trade_id,
            )
            if not fill.filled:
                return
        else:
            fill = await self._live_sell(q, pos, d)
            if fill is None:
                return
        # Net of both legs' fees — the same convention the broker uses.
        pnl = (
            (fill.avg_price_cents - pos.entry_cents) * fill.contracts / 100.0
            - fill.fee_usd
            - pos.fees_usd * (fill.contracts / pos.contracts)
        )

        self.policy.record_close(q.ticker, pnl, now_ts=time.time())
        log.info(
            f"[EXIT] {q.ticker} {pos.side.upper()} x{fill.contracts} @ "
            f"{fill.avg_price_cents:.1f}c pnl=${pnl:+.2f} | {d.reason}"
        )
        self._log_trade(q.ticker, f"{pos.side}_exit", fill.contracts,
                        fill.avg_price_cents, "core/exit", pos.trade_id)
        self._push_trade(q.ticker, f"{pos.side}_exit", fill.contracts,
                         fill.avg_price_cents, "core/exit", pos.trade_id, pnl)

    @staticmethod
    def _order_fill_cents(order: Order, side: str, fallback_cents: float) -> float:
        """Price WE paid/received per contract, in our side's terms.

        The V2 parser normalizes yes_price/no_price to the single-book
        invariant (yes + no = 100), so read our side's field directly and
        fall back to the submitted limit when the response omits both.
        """
        px = order.yes_price if side == "yes" else order.no_price
        return float(px) if px and px > 0 else float(fallback_cents)

    async def _live_buy(self, q: MarketQuote, d, trade_id: str):
        """Live IOC entry. Returns a paper.Fill-shaped object or None."""
        from btc15.core.paper import Fill
        try:
            placed = await self._kalshi.place_order(
                ticker=q.ticker,
                side=Side.YES if d.side == "yes" else Side.NO,
                contracts=d.contracts,
                price_cents=int(d.limit_cents),
                order_type=OrderType.LIMIT,
                client_order_id=f"btc15-{trade_id}",
                time_in_force=TimeInForce.IOC,
            )
        except Exception as e:
            log.warning(f"[ERROR] live buy failed {q.ticker}: {e}")
            return None
        filled = int(placed.filled_count or 0)
        if filled <= 0:
            log.info(f"[REJECT] live IOC unfilled {q.ticker}")
            return None
        avg = self._order_fill_cents(placed, d.side, d.limit_cents)
        reported_fee = float(placed.fees_paid_usd or 0.0)
        if reported_fee > 0:
            self.fee_calibrator.observe(
                fee_usd=reported_fee, price_cents=avg, contracts=filled,
            )
        fee = reported_fee or taker_fee_usd(
            avg, filled, self.cfg.core.fee_rate, self.cfg.core.fee_multiplier,
        )
        gross = avg * filled / 100.0
        self.broker.cash_usd -= (gross + fee)
        self.broker.fees_paid_usd += fee
        return Fill(
            ticker=q.ticker, side=d.side, action="buy", contracts=filled,
            avg_price_cents=avg, gross_usd=gross, fee_usd=fee,
            cash_delta_usd=-(gross + fee), requested_contracts=d.contracts,
            trade_id=trade_id, ts=time.time(), note="live",
        )

    async def _live_sell(self, q: MarketQuote, pos: Position, d):
        """Live exit. Sweeps the book (reduce_only IOC at the 1c floor) —
        Kalshi fills at the resting maker's price, so this takes the best
        available bids rather than rejecting liquidity below our limit."""
        from btc15.core.paper import Fill
        try:
            placed = await self._kalshi.sell_position_sweep(
                q.ticker,
                Side.YES if pos.side == "yes" else Side.NO,
                pos.contracts,
            )
        except Exception as e:
            log.warning(f"[ERROR] live sell failed {q.ticker}: {e}")
            return None
        filled = int(placed.filled_count or 0)
        if filled <= 0:
            log.warning(f"[EXIT] {q.ticker} sweep filled=0 — book empty, retry next scan")
            return None
        avg = self._order_fill_cents(placed, pos.side, d.limit_cents)
        reported_fee = float(placed.fees_paid_usd or 0.0)
        if reported_fee > 0:
            self.fee_calibrator.observe(
                fee_usd=reported_fee, price_cents=avg, contracts=filled,
            )
        fee = reported_fee or taker_fee_usd(
            avg, filled, self.cfg.core.fee_rate, self.cfg.core.fee_multiplier,
        )
        gross = avg * filled / 100.0
        self.broker.cash_usd += (gross - fee)
        self.broker.fees_paid_usd += fee
        return Fill(
            ticker=q.ticker, side=pos.side, action="sell", contracts=filled,
            avg_price_cents=avg, gross_usd=gross, fee_usd=fee,
            cash_delta_usd=(gross - fee), requested_contracts=pos.contracts,
            trade_id=pos.trade_id, ts=time.time(), note="live",
        )

    # ── Settlement ───────────────────────────────────────────────────────────

    async def _settlement_loop(self) -> None:
        while self.running:
            try:
                await asyncio.sleep(self.settlement_check_interval)
                await self._check_settlements()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug(f"Settlement loop error: {e}")

    async def _check_settlements(self) -> None:
        """Resolve held positions. Kalshi's official result is authoritative;
        the paper fallback is our own final-minute BRTI TWAP — the actual
        settlement instrument. We never mark a position off a drained book.
        """
        for ticker in list(self.policy.positions.keys()):
            pos = self.policy.positions[ticker]
            market = self._watched.get(ticker)
            result: Optional[str] = None

            try:
                md = await self._kalshi.get_market(ticker)
                if md.status in (MarketStatus.SETTLED, MarketStatus.FINALIZED) and md.result:
                    result = md.result
                elif md.seconds_remaining > 0:
                    continue
            except Exception:
                pass

            if result is None:
                if market is None:
                    continue
                close_time = market.close_time
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) <= close_time:
                    continue
                if self.cfg.strategy.paper_trade:
                    twap = self._settlement_twap(close_time.timestamp())
                    if twap is None:
                        continue
                    result = "yes" if twap >= market.strike_price else "no"
                    log.info(
                        f"[SETTLED] {ticker} paper TWAP=${twap:,.2f} vs "
                        f"strike=${market.strike_price:,.2f} -> {result.upper()}"
                    )
                else:
                    n = self._pending_settlement.get(ticker, 0) + 1
                    self._pending_settlement[ticker] = n
                    if n % 6 == 0:
                        log.warning(f"[SETTLED] {ticker}: awaiting Kalshi result ({n} checks)")
                    continue

            pnl = self.broker.settle(
                ticker=ticker, side=pos.side, contracts=pos.contracts,
                entry_cents=pos.entry_cents, result=result,
                entry_fee_usd=pos.fees_usd, trade_id=pos.trade_id,
            )
            self.policy.record_close(ticker, pnl, now_ts=time.time())
            self._pending_settlement.pop(ticker, None)
            self._settled[ticker] = time.time()
            settle_cents = 100 if result == pos.side else 0
            log.info(
                f"[SETTLED] {ticker} {pos.side.upper()} x{pos.contracts} "
                f"entry={pos.entry_cents:.1f}c -> {settle_cents}c "
                f"result={result.upper()} pnl=${pnl:+.2f}"
            )
            self._log_trade(ticker, f"{pos.side}_settled", pos.contracts,
                            settle_cents, f"settled/{result}", pos.trade_id)
            self._push_trade(ticker, f"{pos.side}_settled", pos.contracts,
                             settle_cents, f"settled/{result}", pos.trade_id, pnl)

    def _settlement_twap(self, close_ts: float, window_sec: float = 60.0) -> Optional[float]:
        """Mean BRTI over the final `window_sec` before close — the real
        settlement instrument, computed from our own tick history."""
        ticks = self.price_feed.recent_ticks(seconds=window_sec + 120.0)
        window = [t.price for t in ticks if close_ts - window_sec <= t.ts <= close_ts and t.price > 0]
        if len(window) < 5:
            return None
        return sum(window) / len(window)

    # ── Recording / state ────────────────────────────────────────────────────

    def _record_decision(
        self, q: MarketQuote, band: str, decision, nowcast=None,
        book: Optional[dict] = None, executed: bool = False,
    ) -> None:
        """One row per market per scan — the unit of everything offline.

        Always carries p_model AND p_market so core/score.py can measure us
        against the market. It also carries enough state to RE-DERIVE the
        decision under a different config: the sigma legs before and after
        clamping, the accrued settlement average, and the top of the book
        on both sides. That last part is what lets `sweep` simulate fills
        without reading kalshi_frames.jsonl.
        """
        try:
            row = {
                "ts": time.time(),
                "session": self._session_label,
                "ticker": q.ticker,
                "strike": q.strike,
                "spot": round(q.spot, 2),
                "secs": round(q.secs, 1),
                "phase": q.phase,
                "band": band,
                "p_model": round(q.prob_yes, 6),
                "p_market": (round(q.market_prob_yes, 6)
                             if q.market_prob_yes is not None else None),
                "yes_bid": q.yes_bid,
                "yes_ask": q.yes_ask,
                "sigma": round(q.sigma, 5),
                "z": round(q.z_score, 4),
                "degenerate": q.degenerate,
                "action": decision.kind if decision else "none",
                "executed": executed,
                "reject_gate": decision.reject_gate if decision else None,
                "ev_cents": decision.ev_cents if decision else None,
                "edge": decision.edge if decision else None,
                "contracts": decision.contracts if decision else 0,
                "side": (decision.side if decision and decision.side
                         else q.recommended_side),
                "crossed": q.crossed,
                "confidence": round(q.confidence, 5),
            }
            if nowcast is not None:
                # sigma_raw vs sigma is how you detect the floor binding —
                # the failure mode that manufactures 0.999 probabilities.
                row.update({
                    "sigma_raw": round(nowcast.sigma_raw, 5),
                        "sigma_clamped": nowcast.clamped,
                    "sigma_fast": round(nowcast.sigma_fast, 5),
                    "sigma_slow": round(nowcast.sigma_slow, 5),
                    "sigma_n_fast": nowcast.n_fast,
                    "sigma_n_slow": nowcast.n_slow,
                })
            if q.accrued_avg is not None:
                row.update({
                    "accrued_avg": round(q.accrued_avg, 2),
                    "accrued_n": q.accrued_count,
                    "locked_frac": round(q.locked_frac, 4),
                    "k_eff": round(q.k_eff, 2) if q.k_eff is not None else None,
                })
            if book:
                row["book"] = book
            self._recorder.write_decision(row)
        except Exception as e:
            log.debug(f"decision record failed: {e}")

    async def _state_loop(self) -> None:
        while self.running:
            try:
                await asyncio.sleep(1.0)
                self.state["feed_age_sec"] = self.price_feed.feed_age_sec()

                unrealized = 0.0
                positions = []
                for ticker, pos in self.policy.positions.items():
                    yes_bid, yes_ask = await self._market_cache.get_best_prices(ticker)
                    bid = yes_bid if pos.side == "yes" else (
                        100.0 - yes_ask if yes_ask is not None else None
                    )
                    pnl = pos.mark_to_market_usd(bid)
                    unrealized += pnl
                    positions.append({
                        "ticker": ticker, "side": pos.side,
                        "contracts": pos.contracts,
                        "entry_cents": round(pos.entry_cents),
                        "cost": round(pos.cost_usd, 2),
                        "pnl": round(pnl, 2), "source": "core",
                    })
                self.state["open_positions"] = positions
                self.state["unrealized_pnl"] = round(unrealized, 2)
                self.state["balance"] = {
                    "available": round(self.broker.cash_usd, 2),
                    "portfolio": round(
                        self.broker.cash_usd + self.policy.open_cost_basis_usd(), 2
                    ),
                }
                self.state["risk"] = {
                    "session_pnl": round(self.broker.realized_pnl_usd, 2),
                    "session_trades": self.broker.wins + self.broker.losses,
                    "open_positions": len(self.policy.positions),
                    "win_rate": self.broker.win_rate,
                    "halted": self.policy.halted,
                    "halt_reason": self.policy.halt_reason,
                    "fees_paid": round(self.broker.fees_paid_usd, 2),
                }
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.debug(f"State loop error: {e}")

    def _push_trade(self, ticker: str, side: str, contracts: int,
                    price_cents: float, source: str, trade_id: str,
                    pnl: Optional[float]) -> None:
        entry = {
            "ticker": ticker, "side": side, "contracts": contracts,
            "price_cents": round(price_cents),
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "source": source + ("/paper" if self.cfg.strategy.paper_trade else "/live"),
            "trade_id": trade_id,
        }
        if pnl is not None:
            entry["pnl"] = round(pnl, 3)
        self.state["recent_trades"].insert(0, entry)
        self.state["recent_trades"] = self.state["recent_trades"][:50]

    def _ensure_trade_log(self) -> None:
        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self._log_file.exists():
            with open(self._log_file, "w", newline="") as f:
                csv.writer(f).writerow([
                    "trade_id", "timestamp", "ticker", "side", "contracts",
                    "price_cents", "cost_usd", "source", "mode", "session",
                ])

    def _log_trade(self, ticker: str, side: str, contracts: int,
                   price_cents: float, source: str, trade_id: str) -> None:
        try:
            with open(self._log_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    trade_id,
                    datetime.now(timezone.utc).isoformat(),
                    ticker, side, contracts, round(price_cents),
                    round(contracts * price_cents / 100.0, 4),
                    source,
                    "paper" if self.cfg.strategy.paper_trade else "live",
                    self._session_label,
                ])
        except Exception as e:
            log.debug(f"trade log write failed: {e}")
