"""Risk manager — enforces all risk limits before trade execution."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    ticker: str
    side: str
    contracts: int
    price_cents: float
    timestamp: datetime
    pnl: Optional[float] = None
    won: Optional[bool] = None
    persona: Optional[str] = None


@dataclass
class RiskState:
    """Live risk metrics."""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    open_positions: int = 0
    total_exposure_usd: float = 0.0
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=100))
    today: date = field(default_factory=date.today)
    halted: bool = False
    halt_reason: str = ""

    def reset_if_new_day(self):
        today = date.today()
        if today != self.today:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.today = today
            self.halted = False
            self.halt_reason = ""
            log.info("New trading day — daily stats reset")

    @property
    def win_rate(self) -> Optional[float]:
        settled = [t for t in self.recent_trades if t.won is not None]
        if not settled:
            return None
        return sum(1 for t in settled if t.won) / len(settled)

    @property
    def recent_win_rate(self) -> Optional[float]:
        settled = [t for t in list(self.recent_trades)[-20:] if t.won is not None]
        if len(settled) < 5:
            return None
        return sum(1 for t in settled if t.won) / len(settled)


class RiskManager:
    """
    Centralized risk gatekeeper.
    All trades must pass check_trade() before execution.

    The optional `persona` field on records is preserved purely as a
    label for the dashboard / trade log — it has no behavioral effect
    after the unified AutoTrader replaced the Sniper/Scalper/Arb trio.
    """

    def __init__(self, config):
        self.cfg = config
        self.state = RiskState()
        self._persona_pnl: dict[str, float] = {}      # persona → daily PnL
        self._persona_trades: dict[str, int] = {}      # persona → daily trade count

    def check_trade(
        self,
        ticker: str,
        side: str,
        contracts: int,
        price_cents: float,
        bankroll_usd: float,
        persona: Optional[str] = None,
    ) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        reason is empty string if allowed, explanation if blocked.
        """
        self.state.reset_if_new_day()

        if self.state.halted:
            return False, f"Trading halted: {self.state.halt_reason}"

        cost_usd = contracts * price_cents / 100

        # 1. Daily loss limit
        if self.state.daily_pnl <= -self.cfg.daily_loss_limit_usd:
            self._halt(f"Daily loss limit ${self.cfg.daily_loss_limit_usd:.0f} hit")
            return False, self.state.halt_reason

        # 2. Max simultaneous positions
        if self.state.open_positions >= self.cfg.max_open_positions:
            return False, f"Max open positions ({self.cfg.max_open_positions}) reached"

        # 3. Max trade size
        if cost_usd > self.cfg.max_trade_usd:
            return False, f"Trade ${cost_usd:.2f} exceeds max ${self.cfg.max_trade_usd:.2f}"

        # 4. Min trade size
        if cost_usd < self.cfg.min_trade_usd:
            return False, f"Trade ${cost_usd:.2f} below min ${self.cfg.min_trade_usd:.2f}"

        # 5. Max exposure
        if self.state.total_exposure_usd + cost_usd > self.cfg.max_position_per_market_usd * self.cfg.max_open_positions:
            return False, "Total exposure limit reached"

        # 6. Win rate floor
        recent_wr = self.state.recent_win_rate
        if recent_wr is not None and recent_wr < self.cfg.win_rate_min:
            self._halt(f"Win rate {recent_wr:.1%} below floor {self.cfg.win_rate_min:.1%}")
            return False, self.state.halt_reason

        # 7. Sufficient bankroll
        if cost_usd > bankroll_usd * 0.5:
            return False, f"Trade ${cost_usd:.2f} would use >50% of bankroll ${bankroll_usd:.2f}"

        return True, ""

    def record_open(self, ticker: str, side: str, contracts: int, price_cents: float, persona: Optional[str] = None):
        cost = contracts * price_cents / 100
        self.state.open_positions += 1
        self.state.total_exposure_usd += cost
        self.state.daily_trades += 1
        record = TradeRecord(
            ticker=ticker,
            side=side,
            contracts=contracts,
            price_cents=price_cents,
            timestamp=datetime.utcnow(),
            persona=persona,
        )
        self.state.recent_trades.append(record)
        if persona:
            self._persona_trades[persona] = self._persona_trades.get(persona, 0) + 1
        tag = f"[{persona.upper()[:3]}] " if persona else ""
        log.info(f"[RISK] {tag}Trade opened: {ticker} {side.upper()} {contracts}c @ {price_cents}¢ "
                 f"(${cost:.2f}) | Open: {self.state.open_positions} | "
                 f"Daily PnL: ${self.state.daily_pnl:.2f}")

    def record_close(self, ticker: str, won: bool, pnl: float, persona: Optional[str] = None):
        self.state.open_positions = max(0, self.state.open_positions - 1)
        self.state.daily_pnl += pnl
        if persona:
            self._persona_pnl[persona] = self._persona_pnl.get(persona, 0.0) + pnl
        # Update the most recent trade record for this ticker and release exposure
        for record in reversed(self.state.recent_trades):
            if record.ticker == ticker and record.won is None:
                record.won = won
                record.pnl = pnl
                cost = record.contracts * record.price_cents / 100
                self.state.total_exposure_usd = max(0.0, self.state.total_exposure_usd - cost)
                break
        tag = f"[{persona.upper()[:3]}] " if persona else ""
        log.info(f"[RISK] {tag}Trade closed: {ticker} | {'WIN' if won else 'LOSS'} | "
                 f"PnL=${pnl:.2f} | Daily PnL=${self.state.daily_pnl:.2f} | "
                 f"Win rate: {self._format_wr()}")

    def record_exposure_change(self, delta_usd: float):
        self.state.total_exposure_usd = max(0, self.state.total_exposure_usd + delta_usd)

    def _halt(self, reason: str):
        self.state.halted = True
        self.state.halt_reason = reason
        log.warning(f"[RISK] TRADING HALTED: {reason}")

    def resume(self):
        self.state.halted = False
        self.state.halt_reason = ""
        log.info("[RISK] Trading resumed manually")

    def _format_wr(self) -> str:
        wr = self.state.win_rate
        return f"{wr:.1%}" if wr is not None else "N/A"

    def summary(self) -> dict:
        return {
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
            "daily_pnl": self.state.daily_pnl,
            "daily_trades": self.state.daily_trades,
            "open_positions": self.state.open_positions,
            "total_exposure": self.state.total_exposure_usd,
            "win_rate": self.state.win_rate,
            "recent_win_rate": self.state.recent_win_rate,
            "persona_pnl": dict(self._persona_pnl),
            "persona_trades": dict(self._persona_trades),
        }
