"""
Multi-persona trading strategy framework.

Three concurrent personas share the same engine but apply different logic:

  Sniper  — High-conviction directional bets (half-Kelly, aggressive fills)
  Scalper — Market maker posting both sides with post_only (zero fees)
  Arb     — Arbitrage: buy YES+NO when combined cost < $1, stat-arb divergences
"""
from __future__ import annotations

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from btc15.config import SniperConfig, ScalperConfig, ArbConfig
from btc15.kalshi.models import Side, TimeInForce, SelfTradePrevention
from btc15.models.ensemble import ModelOutput
from btc15.strategy.sizer import kelly_fraction_binary

log = logging.getLogger(__name__)


# ── Action: what a persona wants the engine to do ────────────────────────────

@dataclass
class Action:
    """A trade action requested by a persona."""
    persona: str               # "sniper" | "scalper" | "arb"
    action_type: str           # "buy" | "sell" | "amend" | "cancel" | "batch_buy"
    ticker: str
    side: Optional[str] = None          # "yes" | "no"
    contracts: int = 0
    price_cents: int = 0
    post_only: bool = False
    time_in_force: Optional[str] = None
    self_trade_prevention: Optional[str] = None
    order_id: Optional[str] = None      # for amend/cancel
    reason: str = ""
    # For batch_buy (arb): buy both sides
    batch_orders: list = field(default_factory=list)


# ── Base Persona ─────────────────────────────────────────────────────────────

class BasePersona(ABC):
    name: str = "base"
    tag: str = "[BASE]"

    def __init__(self):
        self.positions: dict[str, list[dict]] = {}  # ticker → [{side, entry_cents, contracts}, ...]
        self.resting_orders: dict[str, dict] = {}  # order_id → {ticker, side, price, contracts}
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0

    @abstractmethod
    def evaluate(
        self,
        ticker: str,
        market: dict,
        orderbook: dict,
        output: ModelOutput,
        bankroll_usd: float,
        engine_state: dict,
    ) -> list[Action]:
        """Return a list of actions to execute for this market."""
        ...

    def record_fill(self, ticker: str, side: str, contracts: int, price_cents: int, trade_id: str = ""):
        """Record that an order was filled (entry)."""
        entry = {"side": side, "entry_cents": price_cents, "contracts": contracts, "trade_id": trade_id}
        if ticker not in self.positions:
            self.positions[ticker] = []
        # Merge with existing entry on same side (add contracts, avg price)
        for pos in self.positions[ticker]:
            if pos["side"] == side:
                total = pos["contracts"] + contracts
                pos["entry_cents"] = round(
                    (pos["entry_cents"] * pos["contracts"] + price_cents * contracts) / total
                )
                pos["contracts"] = total
                self.daily_trades += 1
                return
        self.positions[ticker].append(entry)
        self.daily_trades += 1

    def record_exit(self, ticker: str, pnl: float, side: str | None = None):
        """Record that a position was closed."""
        if side and ticker in self.positions:
            self.positions[ticker] = [p for p in self.positions[ticker] if p["side"] != side]
            if not self.positions[ticker]:
                del self.positions[ticker]
        else:
            self.positions.pop(ticker, None)
        self.daily_pnl += pnl

    def record_order(self, order_id: str, ticker: str, side: str, price: int, contracts: int):
        self.resting_orders[order_id] = {
            "ticker": ticker, "side": side, "price": price, "contracts": contracts,
        }

    def remove_order(self, order_id: str):
        self.resting_orders.pop(order_id, None)

    def summary(self) -> dict:
        total_pos = sum(len(entries) for entries in self.positions.values())
        return {
            "name": self.name,
            "positions": total_pos,
            "resting_orders": len(self.resting_orders),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
        }


# ── Sniper: Aggressive Directional ──────────────────────────────────────────

class SniperPersona(BasePersona):
    name = "sniper"
    tag = "[SNP]"

    def __init__(self, cfg: SniperConfig):
        super().__init__()
        self.cfg = cfg
        self._stop_cooldown: dict[str, float] = {}  # ticker → timestamp of last stop-loss

    def evaluate(
        self,
        ticker: str,
        market: dict,
        orderbook: dict,
        output: ModelOutput,
        bankroll_usd: float,
        engine_state: dict,
    ) -> list[Action]:
        if not self.cfg.enabled:
            return []

        secs = market.get("seconds_left", 0)
        if not (self.cfg.min_seconds <= secs <= self.cfg.max_seconds):
            return []

        # Already have a position in this market — check exits
        if ticker in self.positions:
            return self._check_exits(ticker, market, orderbook, output)

        # Block re-entry after a stop-loss on this ticker
        cooldown = getattr(self.cfg, "stop_loss_cooldown_seconds", 180)
        last_stop = self._stop_cooldown.get(ticker, 0)
        if time.time() - last_stop < cooldown:
            return []

        # Need strong confidence AND edge
        if output.confidence < self.cfg.min_confidence:
            return []
        if not output.recommended_side:
            return []

        side = output.recommended_side
        edge = output.edge_yes if side == "yes" else output.edge_no
        if edge is None or edge < self.cfg.min_edge:
            return []

        # Kelly sizing at half-Kelly
        yes_bid = orderbook.get("yes_bid")
        yes_ask = orderbook.get("yes_ask")
        if side == "yes":
            if yes_ask is None:
                return []
            raw_price = int(yes_ask)
            prob_win = output.prob_yes
        else:
            if yes_bid is None:
                return []
            raw_price = int(100 - yes_bid)
            prob_win = output.prob_no

        raw_price = max(1, min(99, raw_price))
        order_price = max(1, min(99, raw_price + self.cfg.slippage_cents))

        frac = kelly_fraction_binary(prob_win, raw_price, self.cfg.kelly_fraction)
        if frac <= 0:
            return []

        budget = bankroll_usd * self.cfg.budget_pct
        dollar_amount = min(frac * bankroll_usd, budget)
        if dollar_amount < 1.0:
            return []

        contracts = int(dollar_amount / (raw_price / 100))
        if contracts <= 0:
            return []

        log.info(
            f"{self.tag} SIGNAL: {ticker} {side.upper()} | "
            f"conf={output.confidence:.0%} edge={edge:+.1%} "
            f"contracts={contracts} @ {order_price}¢"
        )

        return [Action(
            persona=self.name,
            action_type="buy",
            ticker=ticker,
            side=side,
            contracts=contracts,
            price_cents=order_price,
            post_only=False,
            reason=f"sniper conf={output.confidence:.0%} edge={edge:+.1%}",
        )]

    def _check_exits(self, ticker: str, market: dict, orderbook: dict, output: ModelOutput) -> list[Action]:
        actions = []
        for pos in self.positions[ticker]:
            side = pos["side"]
            entry = pos["entry_cents"]

            yes_bid = orderbook.get("yes_bid", 0)
            yes_ask = orderbook.get("yes_ask", 100)
            if side == "yes":
                current_bid = float(yes_bid or 0)
            else:
                current_bid = max(0.0, 100.0 - float(yes_ask or 100))

            if entry <= 0 or current_bid <= 0:
                continue

            pnl_pct = (current_bid - entry) / entry

            # Update trailing peak
            peak = max(pnl_pct, pos.get("peak_pnl_pct", 0.0))
            pos["peak_pnl_pct"] = peak

            exit_reason = None
            if pnl_pct >= self.cfg.take_profit_pct:
                exit_reason = f"sniper_tp ({pnl_pct:+.1%})"
            elif pnl_pct <= -self.cfg.stop_loss_pct:
                self._stop_cooldown[ticker] = time.time()
                exit_reason = f"sniper_sl ({pnl_pct:+.1%})"
            else:
                trail_activate = getattr(self.cfg, "trail_activate_pct", 0.20)
                trail_retrace = getattr(self.cfg, "trail_retracement_pct", 0.50)
                if peak >= trail_activate:
                    trail_floor = peak * (1.0 - trail_retrace)
                    if pnl_pct < trail_floor:
                        exit_reason = f"sniper_trail (peak={peak:+.1%} → floor={trail_floor:+.1%} now={pnl_pct:+.1%})"

            if exit_reason:
                actions.append(Action(
                    persona=self.name, action_type="sell", ticker=ticker,
                    side=side, contracts=pos["contracts"], price_cents=int(current_bid),
                    reason=exit_reason,
                ))
        return actions


# ── Scalper: Market Maker / Spread Capture ───────────────────────────────────

class ScalperPersona(BasePersona):
    name = "scalper"
    tag = "[SCA]"

    def __init__(self, cfg: ScalperConfig):
        super().__init__()
        self.cfg = cfg
        # Track inventory per market: {ticker: {"yes": N, "no": N}}
        self.inventory: dict[str, dict[str, int]] = {}

    def evaluate(
        self,
        ticker: str,
        market: dict,
        orderbook: dict,
        output: ModelOutput,
        bankroll_usd: float,
        engine_state: dict,
    ) -> list[Action]:
        if not self.cfg.enabled:
            return []

        secs = market.get("seconds_left", 0)

        # Approaching settlement: cancel resting orders AND exit any held positions
        if secs < self.cfg.cancel_before_seconds:
            actions = self._cancel_all_for_ticker(ticker)
            exit_actions = self._exit_positions_for_ticker(ticker, orderbook)
            actions.extend(exit_actions)
            return actions

        if not (self.cfg.min_seconds <= secs <= self.cfg.max_seconds):
            return []

        yes_bid = orderbook.get("yes_bid")
        yes_ask = orderbook.get("yes_ask")
        if yes_bid is None or yes_ask is None:
            return []

        spread = yes_ask - yes_bid
        if spread < self.cfg.min_spread_cents:
            return []

        # Use model's fair value as anchor (even at low confidence)
        fair_yes = output.prob_yes * 100  # convert to cents
        fair_yes = max(2, min(98, fair_yes))

        # Quote inside the spread, biased by model
        # YES buy: slightly below fair value
        yes_buy_price = int(max(1, min(fair_yes - 1, yes_ask - 2)))
        # NO buy: slightly below inverse fair value
        no_fair = 100 - fair_yes
        no_buy_price = int(max(1, min(no_fair - 1, (100 - yes_bid) - 2)))

        # Ensure we don't cross ourselves (YES buy + NO buy must be < 100)
        if yes_buy_price + no_buy_price >= 99:
            # Widen quotes
            yes_buy_price = int(fair_yes - 2)
            no_buy_price = int(no_fair - 2)
            if yes_buy_price + no_buy_price >= 99:
                return []

        yes_buy_price = max(1, min(98, yes_buy_price))
        no_buy_price = max(1, min(98, no_buy_price))

        # Check inventory imbalance
        inv = self.inventory.get(ticker, {"yes": 0, "no": 0})
        net_imbalance = abs(inv["yes"] - inv["no"])

        contracts = self.cfg.contracts_per_side
        actions = []

        # Check if we already have resting orders for this ticker
        existing_tickers = {v["ticker"] for v in self.resting_orders.values()}
        if ticker in existing_tickers:
            # Amend existing orders instead of posting new ones
            return self._amend_existing(ticker, yes_buy_price, no_buy_price)

        # Post YES side if not too imbalanced toward YES
        if inv["yes"] - inv["no"] < self.cfg.max_inventory_imbalance:
            actions.append(Action(
                persona=self.name,
                action_type="buy",
                ticker=ticker,
                side="yes",
                contracts=contracts,
                price_cents=yes_buy_price,
                post_only=True,
                self_trade_prevention="taker_at_cross",
                reason=f"scalper_quote spread={spread:.0f}¢",
            ))

        # Post NO side if not too imbalanced toward NO
        if inv["no"] - inv["yes"] < self.cfg.max_inventory_imbalance:
            actions.append(Action(
                persona=self.name,
                action_type="buy",
                ticker=ticker,
                side="no",
                contracts=contracts,
                price_cents=no_buy_price,
                post_only=True,
                self_trade_prevention="taker_at_cross",
                reason=f"scalper_quote spread={spread:.0f}¢",
            ))

        if actions:
            log.info(
                f"{self.tag} QUOTING: {ticker} | spread={spread:.0f}¢ | "
                f"YES@{yes_buy_price}¢ NO@{no_buy_price}¢ x{contracts} | "
                f"inv=Y{inv['yes']}/N{inv['no']}"
            )

        return actions

    def _amend_existing(self, ticker: str, yes_price: int, no_price: int) -> list[Action]:
        """Amend resting orders for this ticker to new prices."""
        actions = []
        for oid, info in list(self.resting_orders.items()):
            if info["ticker"] != ticker:
                continue
            new_price = yes_price if info["side"] == "yes" else no_price
            if abs(new_price - info["price"]) >= 1:  # only amend if price moved
                actions.append(Action(
                    persona=self.name,
                    action_type="amend",
                    ticker=ticker,
                    side=info["side"],
                    price_cents=new_price,
                    order_id=oid,
                    reason="scalper_reprice",
                ))
        return actions

    def _exit_positions_for_ticker(self, ticker: str, orderbook: dict) -> list[Action]:
        """Exit all held positions for a ticker approaching settlement."""
        if ticker not in self.positions:
            return []

        actions = []
        yes_bid = orderbook.get("yes_bid", 0)
        yes_ask = orderbook.get("yes_ask", 100)

        for pos in self.positions[ticker]:
            side = pos["side"]
            if side == "yes":
                current_bid = int(yes_bid or 0)
            else:
                current_bid = int(max(0, 100 - (yes_ask or 100)))

            if current_bid <= 0:
                continue

            log.info(
                f"{self.tag} EXIT BEFORE SETTLEMENT: {ticker} {side.upper()} "
                f"x{pos['contracts']} @ {current_bid}¢"
            )

            actions.append(Action(
                persona=self.name,
                action_type="sell",
                ticker=ticker,
                side=side,
                contracts=pos["contracts"],
                price_cents=current_bid,
                reason="scalper_pre_settlement",
            ))
        return actions

    def _cancel_all_for_ticker(self, ticker: str) -> list[Action]:
        """Cancel all resting orders for a ticker approaching settlement."""
        actions = []
        for oid, info in list(self.resting_orders.items()):
            if info["ticker"] == ticker:
                actions.append(Action(
                    persona=self.name,
                    action_type="cancel",
                    ticker=ticker,
                    order_id=oid,
                    reason="scalper_settlement_cancel",
                ))
        return actions

    def record_fill(self, ticker: str, side: str, contracts: int, price_cents: int, trade_id: str = ""):
        super().record_fill(ticker, side, contracts, price_cents, trade_id)
        if ticker not in self.inventory:
            self.inventory[ticker] = {"yes": 0, "no": 0}
        self.inventory[ticker][side] += contracts

    def record_exit(self, ticker: str, pnl: float, side: str | None = None):
        super().record_exit(ticker, pnl, side=side)
        if ticker not in self.positions:
            self.inventory.pop(ticker, None)

    def summary(self) -> dict:
        base = super().summary()
        total_inv = sum(v["yes"] + v["no"] for v in self.inventory.values())
        base["inventory"] = total_inv
        return base


# ── Arb: Arbitrage / Mispricing Hunter ───────────────────────────────────────

class ArbPersona(BasePersona):
    name = "arb"
    tag = "[ARB]"

    def __init__(self, cfg: ArbConfig):
        super().__init__()
        self.cfg = cfg
        self._stat_arb_tickers: set[str] = set()  # tickers with active stat-arb positions

    def evaluate(
        self,
        ticker: str,
        market: dict,
        orderbook: dict,
        output: ModelOutput,
        bankroll_usd: float,
        engine_state: dict,
    ) -> list[Action]:
        if not self.cfg.enabled:
            return []

        actions = []

        # 0. Check exits for open stat-arb positions (pure arb never exits early)
        if ticker in self.positions and ticker in self._stat_arb_tickers:
            exit_actions = self._check_stat_arb_exits(ticker, market, orderbook)
            if exit_actions:
                return exit_actions

        # 1. Pure arbitrage: YES_ask + NO_ask < 100¢
        arb_action = self._check_pure_arb(ticker, orderbook, bankroll_usd)
        if arb_action:
            actions.append(arb_action)
            return actions  # pure arb takes priority

        # 2. Statistical arbitrage: large model-vs-market divergence
        stat_action = self._check_stat_arb(ticker, market, orderbook, output, bankroll_usd)
        if stat_action:
            actions.append(stat_action)

        return actions

    def _check_stat_arb_exits(self, ticker: str, market: dict, orderbook: dict) -> list[Action]:
        """Check exits for open stat-arb positions: TP/SL on price, time-cut near settlement."""
        actions = []
        seconds_left = market.get("seconds_left", 999)

        for pos in self.positions[ticker]:
            side = pos["side"]
            entry = pos["entry_cents"]
            if not entry or entry <= 0:
                continue

            # Current liquidation value (best bid for our side)
            if side == "yes":
                current_bid = orderbook.get("yes_bid")
            else:
                yes_ask = orderbook.get("yes_ask")
                current_bid = (100 - yes_ask) if yes_ask is not None else None

            if current_bid is None:
                continue

            pnl_pct = (current_bid - entry) / entry

            # Update trailing peak
            peak = max(pnl_pct, pos.get("peak_pnl_pct", 0.0))
            pos["peak_pnl_pct"] = peak

            exit_reason = None
            if pnl_pct >= self.cfg.stat_arb_take_profit_pct:
                exit_reason = f"arb_tp ({pnl_pct:+.1%})"
            elif pnl_pct <= -self.cfg.stat_arb_stop_loss_pct:
                exit_reason = f"arb_sl ({pnl_pct:+.1%})"
            else:
                trail_activate = getattr(self.cfg, "trail_activate_pct", 0.20)
                trail_retrace = getattr(self.cfg, "trail_retracement_pct", 0.50)
                if peak >= trail_activate:
                    trail_floor = peak * (1.0 - trail_retrace)
                    if pnl_pct < trail_floor:
                        exit_reason = f"arb_trail (peak={peak:+.1%} → floor={trail_floor:+.1%} now={pnl_pct:+.1%})"

            if exit_reason:
                log.info(
                    f"{self.tag} STAT ARB EXIT: {ticker} {side.upper()} | "
                    f"entry={entry}¢ current={current_bid}¢ pnl={pnl_pct:+.1%} | {exit_reason}"
                )
                actions.append(Action(
                    persona=self.name, action_type="sell", ticker=ticker,
                    side=side, contracts=pos["contracts"], price_cents=int(current_bid),
                    reason=exit_reason,
                ))
                continue

            if seconds_left <= self.cfg.stat_arb_cut_before_seconds and pnl_pct < 0:
                # Near settlement and losing — sell now to recover whatever the bid offers
                # rather than riding to zero. If we're winning, let it settle for full $1.
                log.info(
                    f"{self.tag} STAT ARB TIME-CUT: {ticker} {side.upper()} | "
                    f"{seconds_left:.0f}s left | entry={entry}¢ current={current_bid}¢ pnl={pnl_pct:+.1%}"
                )
                actions.append(Action(
                    persona=self.name, action_type="sell", ticker=ticker,
                    side=side, contracts=pos["contracts"], price_cents=int(current_bid),
                    reason=f"arb_time_cut ({seconds_left:.0f}s left, {pnl_pct:+.1%})",
                ))
        return actions

    def record_exit(self, ticker: str, pnl: float, side: str | None = None):
        """Override to clean up _stat_arb_tickers when position is fully closed."""
        super().record_exit(ticker, pnl, side)
        if ticker not in self.positions:
            self._stat_arb_tickers.discard(ticker)

    def _check_pure_arb(self, ticker: str, orderbook: dict, bankroll_usd: float) -> Optional[Action]:
        """Buy both YES and NO when combined cost < $1.00 for guaranteed profit."""
        yes_ask = orderbook.get("yes_ask")
        yes_bid = orderbook.get("yes_bid")
        if yes_ask is None or yes_bid is None:
            return None

        # Cost to buy YES = yes_ask cents
        # Cost to buy NO = (100 - yes_bid) cents
        yes_cost = int(yes_ask)
        no_cost = int(100 - yes_bid)
        combined = yes_cost + no_cost

        profit_cents = 100 - combined
        if profit_cents < self.cfg.min_arb_cents:
            return None

        # Size: as many pairs as budget allows
        budget = bankroll_usd * self.cfg.budget_pct
        cost_per_pair = combined / 100  # dollars
        if cost_per_pair <= 0:
            return None
        contracts = min(
            int(budget / cost_per_pair),
            self.cfg.max_contracts,
        )
        if contracts <= 0:
            return None

        log.info(
            f"{self.tag} PURE ARB: {ticker} | "
            f"YES@{yes_cost}¢ + NO@{no_cost}¢ = {combined}¢ | "
            f"profit={profit_cents}¢/pair x{contracts} = ${contracts * profit_cents / 100:.2f}"
        )

        return Action(
            persona=self.name,
            action_type="batch_buy",
            ticker=ticker,
            contracts=contracts,
            reason=f"pure_arb profit={profit_cents}¢/pair",
            batch_orders=[
                {
                    "ticker": ticker,
                    "action": "buy",
                    "side": "yes",
                    "type": "limit",
                    "count": contracts,
                    "yes_price": yes_cost,
                },
                {
                    "ticker": ticker,
                    "action": "buy",
                    "side": "no",
                    "type": "limit",
                    "count": contracts,
                    "no_price": no_cost,
                },
            ],
        )

    def _check_stat_arb(
        self, ticker: str, market: dict, orderbook: dict,
        output: ModelOutput, bankroll_usd: float,
    ) -> Optional[Action]:
        """Bet on large model-vs-market divergence closing."""
        if ticker in self.positions:
            return None

        if output.confidence < self.cfg.stat_arb_min_confidence:
            return None

        yes_ask = orderbook.get("yes_ask")
        yes_bid = orderbook.get("yes_bid")
        if yes_ask is None or yes_bid is None:
            return None

        market_yes = (yes_bid + yes_ask) / 2 / 100  # 0-1 scale
        model_yes = output.prob_yes

        divergence = abs(model_yes - market_yes)
        if divergence < self.cfg.stat_arb_divergence:
            return None

        # Determine direction: if model says YES is underpriced, buy YES
        if model_yes > market_yes:
            side = "yes"
            price = int(yes_ask)
            prob_win = output.prob_yes
        else:
            side = "no"
            price = int(100 - yes_bid)
            prob_win = output.prob_no

        price = max(1, min(99, price))

        # Skip if the market has already priced this side too cheaply —
        # <min_price_cents means the market is 85%+ confident in the other direction.
        # The model divergence here is almost certainly the model being stale, not edge.
        if price < self.cfg.stat_arb_min_price_cents:
            log.debug(
                f"{self.tag} SKIP STAT ARB {ticker}: price={price}¢ < min={self.cfg.stat_arb_min_price_cents}¢ "
                f"(market consensus too strong)"
            )
            return None

        budget = bankroll_usd * self.cfg.budget_pct
        contracts = min(
            int(budget / (price / 100)),
            self.cfg.max_contracts,
        )
        if contracts <= 0:
            return None

        log.info(
            f"{self.tag} STAT ARB: {ticker} {side.upper()} | "
            f"model={model_yes:.1%} vs market={market_yes:.1%} "
            f"divergence={divergence:.1%} | x{contracts} @ {price}¢"
        )

        self._stat_arb_tickers.add(ticker)
        return Action(
            persona=self.name,
            action_type="buy",
            ticker=ticker,
            side=side,
            contracts=contracts,
            price_cents=price,
            reason=f"stat_arb div={divergence:.1%}",
        )
