"""
Unified AutoTrader — replaces Sniper / Scalper / Arb persona trio.

Three time phases drive all decisions based on seconds remaining in the
15-minute KXBTC window:

  Early  (>8 min left)  — GTC post_only at mid-1¢ (0% maker fee)
                           Market-make both sides if spread ≥ 5¢
  Prime  (3–8 min left) — IOC directional entry; escalates unfilled GTC orders
  Late   (<3 min left)  — No new entries; position management only

Exit policy (3 rules — no trailing stop, no profit decay):
  Rule 1: Pure arb pairs always hold to settlement ($1.00 guaranteed)
  Rule 2: Flip sides if model reversal edge > reversal_min_edge AND >5 min left
  Rule 3: Cut losses if pnl < -stop_loss_pct AND <4 min left
  Rule 4: Everything else → hold to settlement

Market making:
  When spread ≥ mm_min_spread_cents, post both YES and NO inside the spread
  with GTC post_only (0% maker fee). Cancel at <mm_cancel_before_seconds left.
  Inventory tracking prevents over-exposure on one side.

Arb:
  Always check YES_ask + NO_ask < 98¢ first. When found, IOC both sides
  immediately — guaranteed profit regardless of direction.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from btc15.config import TraderConfig
from btc15.models.ensemble import ModelOutput
from btc15.strategy.sizer import kelly_fraction_binary

log = logging.getLogger(__name__)


# ── Action: what the AutoTrader wants the engine to execute ──────────────────

@dataclass
class Action:
    """A trade action returned by AutoTrader for the engine to execute."""
    action_type: str            # "buy" | "sell" | "amend" | "cancel" | "batch_buy"
    ticker: str
    side: Optional[str] = None           # "yes" | "no"
    contracts: int = 0
    price_cents: int = 0
    post_only: bool = False
    time_in_force: Optional[str] = None  # "ioc" | "gtc" | None (defaults to ioc)
    self_trade_prevention: Optional[str] = None
    order_id: Optional[str] = None       # for amend / cancel
    reason: str = ""
    # For batch_buy (pure arb): buy both sides atomically
    batch_orders: list = field(default_factory=list)
    # Kept for engine compatibility
    persona: str = "auto"
    # Original GTC price when this action was escalated from a resting order.
    # Used by the engine to measure price drift at IOC failure time.
    original_price_cents: Optional[int] = None
    # Mid price at the moment the signal fired, attached to GTC entries so
    # the escalation path can measure how far the market drifted while we
    # rested. Adverse drift → informed flow moved us → halve or skip.
    signal_mid_cents: Optional[float] = None


# ── AutoTrader ───────────────────────────────────────────────────────────────

class AutoTrader:
    """
    Single unified trading logic class. Replaces Sniper, Scalper, and Arb.

    State:
      positions      — ticker → [{side, entry_cents, contracts, mode, trade_id, ...}]
      resting_orders — order_id → {ticker, side, price, contracts, placed_at, purpose}
    """

    name = "auto"
    tag = "[AUTO]"

    def __init__(self, cfg: TraderConfig):
        self.cfg = cfg
        # ticker → list of open position dicts
        self.positions: dict[str, list[dict]] = {}
        # order_id → {ticker, side, price, contracts, placed_at, purpose, mode}
        self.resting_orders: dict[str, dict] = {}
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        # cooldown after stop-loss on a ticker
        self._stop_cooldown: dict[str, float] = {}
        # cooldown after a reversal exit — prevents flipping back and forth on
        # a chop market. The atomic reversal itself (sell + buy in one evaluate
        # cycle) bypasses this; only SUBSEQUENT new entries are blocked.
        self._reversal_cooldown: dict[str, float] = {}
        # cooldown after a failed IOC escalation — prevents price-chase retry loops.
        # When GTC→IOC escalation fires and the IOC can't fill, it means the market
        # has moved significantly. Without this cooldown, the next scan posts a fresh
        # GTC which escalates again at an even higher price (TB76A0271 pattern).
        self._entry_retry_cooldown: dict[str, float] = {}
        # cumulative realized P&L per ticker this session
        self._ticker_session_pnl: dict[str, float] = {}
        # ticker → timestamp when a loss_cut condition was first detected.
        # Empirically (Friday paper data, 29 SO loss_cut events): 83% of cuts
        # were panic-flushes that recovered within 30s. We require the cut
        # condition to persist across a runway-scaled cool-off window before
        # firing. Cleared whenever the model returns to agreeing OR pnl
        # recovers above threshold. Does NOT apply to emergency_stop (-65%).
        self._pending_loss_cut: dict[str, float] = {}
        # ticker → timestamp when a pure-arb cross was first observed.
        # Arb only fires if the cross persists for at least two consecutive
        # scan cycles — filters transient 1-2¢ WS-delta crosses that are
        # already filled by the time our batch order would reach Kalshi.
        self._arb_first_seen: dict[str, float] = {}
        # ticker → number of reversal exits on this ticker this session.
        # Blocks pyramiding after ≥2 reversals — a choppy market that has
        # flipped us twice is not a market to double down in.
        self._reversal_count: dict[str, int] = {}

    # ── Main evaluation entry point ──────────────────────────────────────────

    def evaluate(
        self,
        ticker: str,
        market_info: dict,      # {seconds_left, volume, ticker, annual_vol}
        orderbook: dict,        # {yes_bid, yes_ask}
        output: ModelOutput,
        bankroll_usd: float,
    ) -> list[Action]:
        if not self.cfg.enabled:
            return []

        secs = float(market_info.get("seconds_left", 0))
        annual_vol = float(market_info.get("annual_vol") or 0.80)
        actions: list[Action] = []

        # ── Step 1: Manage existing positions ────────────────────────────────
        reversal_exit = False
        pyramid_ok = False
        if ticker in self.positions:
            exit_actions = self._evaluate_exits(ticker, secs, orderbook, output)
            actions.extend(exit_actions)
            reversal_exit = any("reversal" in (a.reason or "") for a in exit_actions)
            # Reversal: fall through to entry logic so the flip is atomic
            # (sell + buy in the same action list). Engine processes them
            # sequentially; if the sell fails, the buy is blocked by the
            # position guard in _execute_action.
            #
            # Pyramid: if position is profitable and model still agrees,
            # fall through to entry logic to add contracts.
            if (not reversal_exit
                    and not exit_actions
                    and getattr(self.cfg, "pyramid_enabled", False)):
                pyramid_ok = self._check_pyramid_eligible(
                    ticker, secs, orderbook, output
                )
            if ticker in self.positions and not reversal_exit and not pyramid_ok:
                return actions  # still holding, no flip/pyramid — no new entry

        # ── Step 2: Cancel MM orders approaching settlement ───────────────────
        if secs < self.cfg.mm_cancel_before_seconds:
            actions.extend(self._cancel_mm_orders(ticker))

        # ── Step 3: Escalate stale GTC entry orders → IOC ────────────────────
        escalate = self._check_gtc_escalation(ticker, orderbook, secs, annual_vol)
        if escalate:
            actions.extend(escalate)
            return actions  # wait for escalation result

        # ── Step 4: No new entries in late window ────────────────────────────
        # Exception: settlement lock entries when outcome is near-certain.
        # The BRTI settles on a 60s trailing average — with 30s left, half
        # is locked in. When BSM says ≥88%, BTC is well clear of the strike.
        if secs < self.cfg.late_window_min_seconds:
            lock_cfg_enabled = getattr(self.cfg, "settlement_lock_enabled", True)
            lock_min_secs = getattr(self.cfg, "settlement_lock_min_seconds", 20)
            lock_max_secs = getattr(self.cfg, "settlement_lock_max_seconds", 60)
            lock_min_prob = getattr(self.cfg, "settlement_lock_min_prob", 0.88)
            lock_min_conf = getattr(self.cfg, "settlement_lock_min_confidence", 0.50)

            if (lock_cfg_enabled
                    and lock_min_secs <= secs <= lock_max_secs
                    and ticker not in self.positions
                    and output.recommended_side
                    and output.confidence >= lock_min_conf):
                # Check BSM probability (the most informative model near settlement)
                bsm_prob = output.prob_binary_options
                if bsm_prob is not None:
                    bsm_extreme = bsm_prob >= lock_min_prob or bsm_prob <= (1 - lock_min_prob)
                    if bsm_extreme:
                        lock_entry = self._check_settlement_lock_entry(
                            ticker, orderbook, output, bankroll_usd, secs,
                        )
                        if lock_entry:
                            actions.append(lock_entry)
                            return actions
            return actions

        if ticker in self.positions and not reversal_exit and not pyramid_ok:
            return actions

        # ── Step 5: Stop-loss & whipsaw cooldowns ────────────────────────────
        # Stop cooldown blocks *any* new entry for N seconds after a loss cut;
        # reversal cooldown blocks re-entries on a ticker that just flipped.
        # The atomic reversal itself runs in the same evaluate cycle and
        # therefore bypasses — only the NEXT scan sees the cooldown.
        now = time.time()
        stop_ts = self._stop_cooldown.get(ticker, 0)
        stop_cd = getattr(self.cfg, "stop_cooldown_seconds", 90)
        if now - stop_ts < stop_cd:
            return actions

        if not reversal_exit:
            rev_ts = self._reversal_cooldown.get(ticker, 0)
            rev_cd = getattr(self.cfg, "reversal_cooldown_seconds", 60)
            if now - rev_ts < rev_cd:
                return actions

        # ── Step 6: Pure arb (guaranteed profit — always highest priority) ────
        # Guard: skip if arb legs already resting (same stacking problem as entries)
        has_resting_arb = any(
            v.get("ticker") == ticker and v.get("purpose") == "arb"
            for v in self.resting_orders.values()
        )
        if not has_resting_arb:
            arb = self._check_pure_arb(ticker, orderbook, bankroll_usd)
            if arb:
                actions.append(arb)
                return actions

        # ── Step 7: Market making in early window ─────────────────────────────
        if secs > self.cfg.early_window_min_seconds:
            mm = self._check_market_making(ticker, orderbook)
            if mm:
                actions.extend(mm)
                return actions

        # ── Step 7b: Block directional entry if one is already resting ───────
        # GTC fills arrive via WS *after* the order is placed, so positions is
        # empty while the order rests. Without this guard every 3-second scan
        # cycle places a new entry, stacking up N orders before the first fill
        # arrives — producing the rapid-fire / phantom-order pattern.
        has_resting_entry = any(
            v.get("ticker") == ticker and v.get("purpose") == "entry"
            for v in self.resting_orders.values()
        )
        if has_resting_entry:
            return actions

        # ── Step 8: Directional entry ─────────────────────────────────────────
        if secs <= self.cfg.max_entry_seconds:
            entry = self._check_directional_entry(
                ticker, orderbook, output, bankroll_usd, secs, annual_vol,
                is_reversal=reversal_exit,
            )
            if entry:
                actions.append(entry)

        # Arm the reversal cooldown AFTER the atomic flip completes so that
        # subsequent cycles are locked, but the current cycle's re-entry
        # (handled above) has already passed the cooldown gate.
        if reversal_exit:
            self._reversal_cooldown[ticker] = time.time()

        return actions

    # ── Exit logic ────────────────────────────────────────────────────────────

    def _evaluate_exits(
        self,
        ticker: str,
        secs: float,
        orderbook: dict,
        output: ModelOutput,
    ) -> list[Action]:
        """
        Three exit rules — everything else holds to settlement:

        Rule 1: Pure arb positions — never exit early (guaranteed $1.00).
        Rule 2: Signal reversal — flip when model flips strongly AND time permits.
        Rule 3: Loss cut — only near settlement to preserve capital.
        """
        actions = []
        yes_bid = float(orderbook.get("yes_bid") or 0)
        # Do NOT default yes_ask to 100 — that makes NO bid appear as 0¢ when data is missing.
        # Use 0 (unknown) and let the current_bid <= 0 guard skip the position safely.
        yes_ask = float(orderbook.get("yes_ask") or 0)

        for pos in list(self.positions.get(ticker, [])):
            side = pos["side"]
            entry = pos["entry_cents"]
            mode = pos.get("mode", "directional")

            if pos.get("settling"):
                continue
            if mode == "arb":
                # Guaranteed profit — let it settle
                continue

            if entry <= 0:
                continue  # corrupt position record — skip

            if side == "yes":
                current_bid = yes_bid
            else:
                # NO bid = 100 - YES ask (what a NO seller receives).
                # When yes_ask is 0 (empty book), use YES bid as conservative lower bound.
                if yes_ask > 0:
                    current_bid = max(0.0, 100.0 - yes_ask)
                elif yes_bid > 0:
                    current_bid = max(0.0, 100.0 - yes_bid)
                else:
                    current_bid = 0.0

            # IMPORTANT: do NOT skip when current_bid = 0.
            # A 0¢ bid means the market has moved fully against us — pnl = -100%.
            # Skipping here is what caused T13873491 and T0ACDC05C to never stop-loss:
            # YES buyers vanished, bid → 0, old guard triggered, position sat unmanaged
            # until settlement. Instead, compute pnl at 0¢ and let stop rules fire.
            # The engine's sell handler now performs a REST refresh on any 0¢ sell
            # action and retries next cycle if the book is still thin, rather than
            # latching `settling=True` — so spurious 0¢ reads no longer retire a
            # healthy position permanently.
            pnl_pct = (current_bid - entry) / entry  # -1.0 when bid=0 (full loss)

            # Emergency stop: fire immediately at any time if loss exceeds threshold.
            # Bypasses the time gate AND the cool-off below — a 65%+ loss is
            # unrecoverable regardless of how much time remains.
            if pnl_pct <= -self.cfg.emergency_stop_pct:
                self._stop_cooldown[ticker] = time.time()
                self._pending_loss_cut.pop(ticker, None)
                log.warning(
                    f"{self.tag} EMERGENCY STOP: {ticker} {side.upper()} | "
                    f"pnl={pnl_pct:+.1%} (threshold={-self.cfg.emergency_stop_pct:.0%}) | "
                    f"{secs:.0f}s left"
                )
                actions.append(Action(
                    action_type="sell", ticker=ticker, side=side,
                    contracts=pos["contracts"], price_cents=int(current_bid),
                    reason=f"emergency_stop pnl={pnl_pct:+.1%}",
                ))
                continue

            # Rule 2: Signal reversal with strong edge and time to recover
            if (output.recommended_side
                    and output.recommended_side != side
                    and secs > self.cfg.reversal_min_seconds):
                opp_edge = float(
                    (output.edge_no if side == "yes" else output.edge_yes) or 0
                )
                if opp_edge >= self.cfg.reversal_min_edge:
                    self._reversal_count[ticker] = self._reversal_count.get(ticker, 0) + 1
                    self._pending_loss_cut.pop(ticker, None)
                    log.info(
                        f"{self.tag} REVERSAL EXIT: {ticker} {side.upper()} | "
                        f"pnl={pnl_pct:+.1%} → flip to {output.recommended_side.upper()} "
                        f"(edge={opp_edge:+.1%}) | reversals_this_market={self._reversal_count[ticker]}"
                    )
                    actions.append(Action(
                        action_type="sell", ticker=ticker, side=side,
                        contracts=pos["contracts"], price_cents=int(current_bid),
                        reason=f"reversal→{output.recommended_side} edge={opp_edge:+.1%}",
                    ))
                    continue

            # Rule 2.5: Time-adaptive profit-take.
            # Near settlement, the risk/reward of holding flips: upside shrinks
            # but gamma risk from thin-book BTC ticks grows. Take profits more
            # aggressively as time runs down.
            if secs > 300:
                pt_bid_thresh, pt_pnl_thresh = 90, 0.30   # >5 min: original rule
            elif secs > 180:
                pt_bid_thresh, pt_pnl_thresh = 85, 0.25   # 3-5 min: slightly tighter
            else:
                pt_bid_thresh, pt_pnl_thresh = 80, 0.20   # <3 min: bank it

            if current_bid >= pt_bid_thresh and secs > 20 and pnl_pct >= pt_pnl_thresh:
                self._pending_loss_cut.pop(ticker, None)
                log.info(
                    f"{self.tag} PROFIT TAKE: {ticker} {side.upper()} | "
                    f"bid={current_bid:.0f}¢ pnl={pnl_pct:+.1%} | {secs:.0f}s left "
                    f"(thresh={pt_bid_thresh}¢/{pt_pnl_thresh:.0%})"
                )
                actions.append(Action(
                    action_type="sell", ticker=ticker, side=side,
                    contracts=pos["contracts"], price_cents=int(current_bid),
                    reason=f"profit_take pnl={pnl_pct:+.1%}",
                ))
                continue

            # Rule 3: Time-decayed loss cut — suppressed when model agrees,
            # gated by a runway-scaled cool-off when model disagrees.
            #
            # Near settlement, binary prices swing wildly on thin liquidity.
            # A NO contract can dip to 35¢ on a single BTC tick toward strike,
            # then settle at 100¢ when BTC doesn't actually cross.
            #
            # Suppression: model recommends our side with ≥40% confidence OR
            # our side still shows ≥+3% edge at current prices. The edge-floor
            # disjunction prevents a confidence-cliff effect (model at 38% conf
            # but +5% edge on our side is still saying we're ahead).
            #
            # Cool-off: empirically (Friday paper data, 29 SO loss_cut events)
            # 83% of stops were panic-flushes that recovered within 30s. When
            # there's runway, require the cut condition to persist across a
            # short cool-off window before firing. Wait scales with runway:
            # more time left = more patience; <240s = no wait, decisive cut.
            our_side_edge = output.edge_yes if side == "yes" else output.edge_no
            our_side_edge = float(our_side_edge) if our_side_edge is not None else 0.0
            model_agrees = (
                (output.recommended_side == side and output.confidence >= 0.40)
                or our_side_edge >= 0.03
            )

            # Snapshot model context for log lines (used in both branches).
            rec = output.recommended_side or "none"
            ctx = (f"rec={rec} conf={output.confidence:.0%} "
                   f"edge_our={our_side_edge:+.1%}")

            if model_agrees:
                # Suppression: model still likes our side. Clear any pending cut
                # — it would only have been there from a brief disagreement window.
                self._pending_loss_cut.pop(ticker, None)
                if secs > 480:
                    would_thresh = -0.55
                elif secs > 240:
                    would_thresh = -0.40
                else:
                    would_thresh = -0.25
                if pnl_pct <= would_thresh:
                    log.info(
                        f"{self.tag} STOP SUPPRESSED: {ticker} {side.upper()} | "
                        f"pnl={pnl_pct:+.1%} (would cut at {would_thresh:+.0%}) | "
                        f"{ctx} — holding"
                    )
            else:
                if secs > 480:
                    stop_thresh = -0.55  # >8min: lots of recovery time
                elif secs > 240:
                    stop_thresh = -0.40  # 4-8min: standard
                else:
                    stop_thresh = -0.25  # <4min: cut anything decisively underwater

                if pnl_pct <= stop_thresh:
                    # Cool-off: more runway → more confirmation required.
                    if secs > 480:
                        cool_off_secs = 6.0
                    elif secs > 240:
                        cool_off_secs = 3.0
                    else:
                        cool_off_secs = 0.0  # final 4 min: no wait

                    now = time.time()
                    pending_ts = self._pending_loss_cut.get(ticker)

                    if cool_off_secs > 0 and pending_ts is None:
                        # First detection — arm pending, don't cut yet.
                        self._pending_loss_cut[ticker] = now
                        log.info(
                            f"{self.tag} LOSS CUT PENDING ({cool_off_secs:.0f}s cool-off): "
                            f"{ticker} {side.upper()} | "
                            f"pnl={pnl_pct:+.1%} thresh={stop_thresh:+.0%} | "
                            f"{secs:.0f}s left | {ctx}"
                        )
                        continue
                    if cool_off_secs > 0 and pending_ts is not None \
                            and (now - pending_ts) < cool_off_secs:
                        # Still inside cool-off window — wait for next scan.
                        # (Don't log every tick to avoid spam; the PENDING line above
                        # marks the start.)
                        continue

                    # Cool-off elapsed (or zero), and the cut condition is still
                    # true on this tick → fire.
                    self._pending_loss_cut.pop(ticker, None)
                    self._stop_cooldown[ticker] = time.time()
                    cool_str = f" (after {cool_off_secs:.0f}s cool-off)" if cool_off_secs > 0 else ""
                    log.info(
                        f"{self.tag} LOSS CUT{cool_str}: {ticker} {side.upper()} | "
                        f"pnl={pnl_pct:+.1%} thresh={stop_thresh:+.0%} | "
                        f"{secs:.0f}s left | {ctx}"
                    )
                    actions.append(Action(
                        action_type="sell", ticker=ticker, side=side,
                        contracts=pos["contracts"], price_cents=int(current_bid),
                        reason=f"loss_cut pnl={pnl_pct:+.1%} {secs:.0f}s left",
                    ))
                else:
                    # pnl recovered above stop threshold — clear any pending cut.
                    self._pending_loss_cut.pop(ticker, None)

            # Rule 4: Hold to settlement (no trailing stop, no profit decay)

        return actions

    # ── Pyramid check ─────────────────────────────────────────────────────────

    def _check_pyramid_eligible(
        self,
        ticker: str,
        secs: float,
        orderbook: dict,
        output: ModelOutput,
    ) -> bool:
        """Check if an existing position qualifies for pyramiding (adding).

        Returns True if the position is profitable, model still agrees,
        and we haven't already added the max times.
        """
        min_pnl = getattr(self.cfg, "pyramid_min_pnl_pct", 0.10)
        min_conf = getattr(self.cfg, "pyramid_min_confidence", 0.55)
        min_edge = getattr(self.cfg, "pyramid_min_edge", 0.05)
        min_secs = getattr(self.cfg, "pyramid_min_seconds", 300)
        max_adds = getattr(self.cfg, "pyramid_max_adds", 1)

        if secs < min_secs:
            return False

        positions = self.positions.get(ticker, [])
        if not positions:
            return False

        # Block pyramiding on a ticker that has reversed ≥2 times — the market is
        # chopping and adding size only deepens the hole.
        rev_count = self._reversal_count.get(ticker, 0)
        if rev_count >= 2:
            log.debug(
                f"{self.tag} PYRAMID BLOCKED: {ticker} — {rev_count} reversals already on this market"
            )
            return False

        for pos in positions:
            if pos.get("mode") in ("arb", "mm_yes", "mm_no", "mm"):
                continue  # don't pyramid arb/MM positions

            side = pos["side"]
            entry = pos["entry_cents"]
            adds = pos.get("pyramid_adds", 0)

            if adds >= max_adds:
                continue

            # In the early window (>prime_min_seconds remaining), cap at 1 pyramid add
            # regardless of max_adds. Don't compound before the initial position is proven.
            in_early = secs > getattr(self.cfg, "prime_window_min_seconds", 180)
            if in_early and adds >= 1:
                continue

            if entry <= 0:
                continue

            # Check P&L
            yes_bid = float(orderbook.get("yes_bid") or 0)
            yes_ask = float(orderbook.get("yes_ask") or 0)
            if side == "yes":
                current_bid = yes_bid
            else:
                if yes_ask > 0:
                    current_bid = max(0.0, 100.0 - yes_ask)
                elif yes_bid > 0:
                    current_bid = max(0.0, 100.0 - yes_bid)
                else:
                    current_bid = 0.0

            if current_bid <= 0:
                continue

            pnl_pct = (current_bid - entry) / entry
            if pnl_pct < min_pnl:
                continue

            # Check model agreement
            if (output.recommended_side == side
                    and output.confidence >= min_conf):
                edge = (output.edge_yes if side == "yes" else output.edge_no) or 0
                if edge >= min_edge:
                    log.info(
                        f"{self.tag} PYRAMID eligible: {ticker} {side.upper()} | "
                        f"pnl={pnl_pct:+.1%} conf={output.confidence:.0%} "
                        f"edge={edge:+.1%} adds={adds}/{max_adds}"
                    )
                    return True

        return False

    # ── Pure arbitrage ────────────────────────────────────────────────────────

    def _check_pure_arb(
        self,
        ticker: str,
        orderbook: dict,
        bankroll_usd: float,
    ) -> Optional[Action]:
        """Buy both YES and NO when combined cost < 100¢ → guaranteed profit."""
        yes_ask = orderbook.get("yes_ask")
        yes_bid = orderbook.get("yes_bid")
        if yes_ask is None or yes_bid is None:
            return None

        # Use round() not int() — floor truncation was inflating apparent profit
        # by up to 2¢ on sub-cent prices, triggering false arbs on tiny crosses.
        yes_cost = round(yes_ask)
        no_cost = round(100 - yes_bid)
        combined = yes_cost + no_cost
        profit_cents = 100 - combined

        if profit_cents < self.cfg.min_arb_cents:
            # No arb — clear any pending confirmation so stale state doesn't linger.
            self._arb_first_seen.pop(ticker, None)
            return None

        # Persistence guard: require the cross to survive two consecutive scans
        # (~6s) before acting. The live WS delta stream sees real-time order flow
        # including transient crosses that the matching engine resolves in <100ms —
        # far faster than our batch-buy round-trip. Firing on those wastes the
        # arb budget and blocks directional logic every scan cycle.
        now = time.time()
        first_seen = self._arb_first_seen.get(ticker)
        if first_seen is None:
            self._arb_first_seen[ticker] = now
            log.debug(f"{self.tag} ARB CANDIDATE: {ticker} profit={profit_cents}¢ — confirming next scan")
            return None
        if now - first_seen < 5.0:
            return None
        # Cross persisted — clear candidate and proceed.
        self._arb_first_seen.pop(ticker, None)

        budget = bankroll_usd * self.cfg.budget_pct
        cost_per_pair = combined / 100
        if cost_per_pair <= 0:
            return None

        contracts = min(
            int(budget / cost_per_pair),
            self.cfg.max_arb_contracts,
        )
        if contracts <= 0:
            return None

        log.info(
            f"{self.tag} PURE ARB: {ticker} | "
            f"YES@{yes_cost}¢ + NO@{no_cost}¢ = {combined}¢ | "
            f"guaranteed profit={profit_cents}¢/pair × {contracts} = "
            f"${contracts * profit_cents / 100:.2f}"
        )

        return Action(
            action_type="batch_buy",
            ticker=ticker,
            contracts=contracts,
            reason=f"pure_arb profit={profit_cents}¢/pair",
            batch_orders=[
                {"ticker": ticker, "action": "buy", "side": "yes",
                 "type": "limit", "count": contracts,
                 "yes_price_dollars": f"{yes_cost/100:.2f}"},
                {"ticker": ticker, "action": "buy", "side": "no",
                 "type": "limit", "count": contracts,
                 "no_price_dollars": f"{no_cost/100:.2f}"},
            ],
        )

    # ── Market making ─────────────────────────────────────────────────────────

    def _check_market_making(
        self,
        ticker: str,
        orderbook: dict,
    ) -> list[Action]:
        """
        Post both sides inside the spread with GTC post_only (0% maker fee).
        Only when spread ≥ mm_min_spread_cents.
        Inventory tracking prevents runaway directional exposure.
        """
        yes_bid = orderbook.get("yes_bid")
        yes_ask = orderbook.get("yes_ask")
        if yes_bid is None or yes_ask is None:
            return []

        spread = float(yes_ask) - float(yes_bid)
        if spread < self.cfg.mm_min_spread_cents:
            return []

        # Inventory: count YES vs NO contracts held from MM mode
        inv_yes = sum(
            p["contracts"] for p in self.positions.get(ticker, [])
            if p.get("mode") in ("mm_yes", "mm")
        )
        inv_no = sum(
            p["contracts"] for p in self.positions.get(ticker, [])
            if p.get("mode") in ("mm_no", "mm")
        )

        # Quote inside the spread
        yes_buy = int(max(1, float(yes_ask) - self.cfg.mm_quote_offset_cents - 1))
        no_buy = int(max(1, (100 - float(yes_bid)) - self.cfg.mm_quote_offset_cents - 1))

        # Prevent quoting against ourselves (combined must be < 100¢)
        if yes_buy + no_buy >= 99:
            yes_buy = int(float(yes_bid) + 1)
            no_buy = int(100 - float(yes_ask) + 1)
            if yes_buy + no_buy >= 99:
                return []

        yes_buy = max(1, min(98, yes_buy))
        no_buy = max(1, min(98, no_buy))

        # Amend existing MM orders if prices have moved
        existing = {v["ticker"] for v in self.resting_orders.values()
                    if v.get("purpose") == "mm"}
        if ticker in existing:
            return self._amend_mm_orders(ticker, yes_buy, no_buy)

        contracts = self.cfg.mm_contracts_per_side
        actions = []

        if inv_yes - inv_no < self.cfg.mm_max_inventory:
            actions.append(Action(
                action_type="buy", ticker=ticker, side="yes",
                contracts=contracts, price_cents=yes_buy,
                post_only=True, time_in_force="gtc",
                self_trade_prevention="taker_at_cross",
                reason=f"mm_quote spread={spread:.0f}¢",
            ))

        if inv_no - inv_yes < self.cfg.mm_max_inventory:
            actions.append(Action(
                action_type="buy", ticker=ticker, side="no",
                contracts=contracts, price_cents=no_buy,
                post_only=True, time_in_force="gtc",
                self_trade_prevention="taker_at_cross",
                reason=f"mm_quote spread={spread:.0f}¢",
            ))

        if actions:
            log.info(
                f"{self.tag} MM QUOTE: {ticker} | spread={spread:.0f}¢ | "
                f"YES@{yes_buy}¢ NO@{no_buy}¢ ×{contracts} | "
                f"inv=Y{inv_yes}/N{inv_no}"
            )
        return actions

    def _amend_mm_orders(self, ticker: str, yes_price: int, no_price: int) -> list[Action]:
        actions = []
        for oid, info in list(self.resting_orders.items()):
            if info.get("ticker") != ticker or info.get("purpose") != "mm":
                continue
            new_price = yes_price if info["side"] == "yes" else no_price
            if abs(new_price - info["price"]) >= 2:
                actions.append(Action(
                    action_type="amend", ticker=ticker, side=info["side"],
                    price_cents=new_price, order_id=oid, reason="mm_reprice",
                ))
        return actions

    def _cancel_mm_orders(self, ticker: str) -> list[Action]:
        return [
            Action(
                action_type="cancel", ticker=ticker, order_id=oid,
                reason="mm_settlement_cancel",
            )
            for oid, info in list(self.resting_orders.items())
            if info.get("ticker") == ticker and info.get("purpose") == "mm"
        ]

    # ── Directional entry ─────────────────────────────────────────────────────

    def _check_directional_entry(
        self,
        ticker: str,
        orderbook: dict,
        output: ModelOutput,
        bankroll_usd: float,
        secs: float,
        annual_vol: float = 0.80,
        is_reversal: bool = False,
    ) -> Optional[Action]:
        """
        Phase-aware directional entry:

          Early phase (>prime_window threshold):
            GTC post_only at mid − 1¢ (0% maker fee; patient fill)
            Will escalate to IOC after gtc_escalate_seconds if unfilled.

          Prime phase (3–8 min left):
            IOC at ask + slippage (guaranteed fill or miss, no resting order).

          Strong-signal override:
            When confidence ≥ 0.65 AND |edge| ≥ 0.08, treat as prime regardless
            of seconds remaining — the move is happening NOW, no patience.

          Reversal re-entry:
            When is_reversal=True, the reversal already validated edge ≥ 0.10
            (double the normal threshold). Skip the confidence gate — edge is
            the binding constraint, not model agreement.
        """
        # Block entry if a prior IOC escalation recently failed for this ticker.
        # Prevents the price-chase loop where each failed IOC lets the next scan
        # post a new GTC that escalates at an even worse price.
        if time.time() < self._entry_retry_cooldown.get(ticker, 0):
            return None

        # Confidence gate: skip for reversal re-entries (edge already validated
        # at 0.10 — twice the normal bar). For normal entries, require min_confidence.
        if not is_reversal and output.confidence < self.cfg.min_confidence:
            return None
        if not output.recommended_side:
            return None

        side = output.recommended_side
        edge = output.edge_yes if side == "yes" else output.edge_no
        if edge is None or edge < self.cfg.min_edge:
            return None

        # Suppress high-edge, low-confidence entries: when the model shows extreme
        # edge (>25%) but confidence is near-neutral (<52%), Kalshi's market makers
        # are pricing a strong directional move our model can't confirm. This exact
        # pattern (e.g. edge=37.7%, conf=50%) was the largest single-session loss
        # driver in 21APR09:02 — the market knew; the model was late/wrong.
        if not is_reversal and edge > 0.25 and output.confidence < 0.52:
            log.info(
                f"{self.tag} ENTRY SUPPRESSED: {ticker} {side.upper()} "
                f"conf={output.confidence:.0%} edge={edge:+.1%} — high-edge/low-conf skip"
            )
            return None

        # Reversal re-entry orderbook confirmation.
        # Post-mortem: reversals exited at strong model edge but re-entries lost
        # because the orderbook hadn't flipped yet — the ensemble had, but the
        # tape was still chopping. Require prob_orderbook to actually agree in
        # the flip direction before buying into the reversal.
        if (is_reversal
                and getattr(self.cfg, "reversal_require_orderbook_confirm", True)):
            min_dev = getattr(self.cfg, "reversal_orderbook_min_dev", 0.10)
            p_ob = output.prob_orderbook
            if p_ob is None:
                log.info(
                    f"{self.tag} REVERSAL RE-ENTRY SKIPPED: {ticker} {side.upper()} "
                    f"— orderbook too thin to confirm flip"
                )
                return None
            dev = (p_ob - 0.5) if side == "yes" else (0.5 - p_ob)
            if dev < min_dev:
                log.info(
                    f"{self.tag} REVERSAL RE-ENTRY SKIPPED: {ticker} {side.upper()} "
                    f"— orderbook P={p_ob:.2f} not confirming flip "
                    f"(need dev≥{min_dev:.2f} in {side} direction)"
                )
                return None

        yes_bid = orderbook.get("yes_bid")
        yes_ask = orderbook.get("yes_ask")

        # Strong-signal override: when conviction is high, skip the patient
        # GTC and go straight to IOC. This is what makes earlier "aggressive"
        # iterations responsive — we stop watching the price run away while
        # a 12s GTC sits at mid-1¢.
        strong_signal = output.confidence >= 0.65 and abs(edge) >= 0.08
        in_prime = secs <= self.cfg.prime_window_min_seconds or strong_signal

        # Adaptive slippage by realized volatility — calm tape stays at 2¢,
        # storm tape pays up to 4¢ to actually fill.
        slip = self.cfg.slippage_cents
        if annual_vol > 1.0:
            slip += 1
        if annual_vol > 1.5:
            slip += 1

        if side == "yes":
            if yes_ask is None:
                return None
            prob_win = output.prob_yes
            if in_prime:
                raw_price = int(yes_ask) + slip
                use_gtc = False
            else:
                mid = ((float(yes_bid) + float(yes_ask)) / 2) if yes_bid else float(yes_ask)
                raw_price = max(1, int(mid) - 1)
                use_gtc = True
        else:
            if yes_bid is None:
                return None
            prob_win = output.prob_no
            if in_prime:
                raw_price = int(100 - float(yes_bid)) + slip
                use_gtc = False
            else:
                mid = ((float(yes_bid) + float(yes_ask)) / 2) if yes_ask else float(yes_bid)
                raw_price = max(1, int(100 - mid) - 1)
                use_gtc = True

        raw_price = max(1, min(99, raw_price))

        if raw_price < self.cfg.min_entry_price_cents:
            return None

        # Three-tier Kelly: strong signals get 0.75x to capitalize on
        # high-conviction opportunities. The max_single_trade_usd cap
        # still bounds absolute exposure.
        kelly_frac_strong = getattr(self.cfg, "kelly_fraction_strong", 0.75)
        if strong_signal:
            kelly_frac = kelly_frac_strong
        elif in_prime:
            kelly_frac = self.cfg.kelly_fraction_prime
        else:
            kelly_frac = self.cfg.kelly_fraction_early

        frac = kelly_fraction_binary(prob_win, raw_price, kelly_frac)
        if frac <= 0:
            return None

        dollar_amount = min(
            frac * bankroll_usd,
            bankroll_usd * self.cfg.budget_pct,
            self.cfg.max_single_trade_usd,
        )
        if dollar_amount < self.cfg.min_single_trade_usd:
            return None

        contracts = int(dollar_amount / (raw_price / 100))
        if contracts <= 0:
            return None

        phase = "prime" if in_prime else "early"
        order_mode = "IOC" if in_prime else "GTC"

        # Mid at signal time — used by escalation drift gate to measure
        # adverse selection if a GTC has to escalate.
        if yes_bid is not None and yes_ask is not None:
            if side == "yes":
                signal_mid = (float(yes_bid) + float(yes_ask)) / 2
            else:
                # NO side: mid of the implied NO price = 100 - (YES mid)
                signal_mid = 100 - ((float(yes_bid) + float(yes_ask)) / 2)
        else:
            signal_mid = float(raw_price)  # best we can do

        log.info(
            f"{self.tag} SIGNAL [{phase}|{order_mode}]: {ticker} {side.upper()} | "
            f"conf={output.confidence:.0%} edge={edge:+.1%} "
            f"×{contracts} @ {raw_price}¢ mid={signal_mid:.1f}¢"
        )

        return Action(
            action_type="buy",
            ticker=ticker,
            side=side,
            contracts=contracts,
            price_cents=raw_price,
            post_only=use_gtc,
            time_in_force="gtc" if use_gtc else "ioc",
            reason=f"dir_{phase} conf={output.confidence:.0%} edge={edge:+.1%}",
            signal_mid_cents=signal_mid,
        )

    # ── Settlement lock entry ────────────────────────────────────────────────

    def _check_settlement_lock_entry(
        self,
        ticker: str,
        orderbook: dict,
        output: ModelOutput,
        bankroll_usd: float,
        secs: float,
    ) -> Optional[Action]:
        """Late-window entry when BRTI settlement is near-certain.

        Only called when BSM probability is extreme (≥88% or ≤12%), meaning
        BTC is well clear of the strike with <60s remaining. The BRTI uses
        a 60-second trailing average — with 30s left, half is locked in.
        Everyone else fears late-window gamma, but gamma only bites when
        BTC is near the strike. When it's far away, this is free money.

        Always IOC, half-Kelly (bounded market, short hold, near-certain).
        """
        side = output.recommended_side
        if not side:
            return None

        edge = output.edge_yes if side == "yes" else output.edge_no
        if edge is None or edge < 0.03:  # lower edge bar — near-certainty
            return None

        yes_bid = orderbook.get("yes_bid")
        yes_ask = orderbook.get("yes_ask")

        if side == "yes":
            if yes_ask is None:
                return None
            prob_win = output.prob_yes
            raw_price = int(yes_ask) + 2  # pay up — speed matters
        else:
            if yes_bid is None:
                return None
            prob_win = output.prob_no
            raw_price = int(100 - float(yes_bid)) + 2

        raw_price = max(1, min(99, raw_price))

        # Half-Kelly for settlement locks — high certainty but short window
        frac = kelly_fraction_binary(prob_win, raw_price, self.cfg.kelly_fraction_prime)
        if frac <= 0:
            return None

        dollar_amount = min(
            frac * bankroll_usd,
            bankroll_usd * self.cfg.budget_pct,
            self.cfg.max_single_trade_usd,
        )
        if dollar_amount < self.cfg.min_single_trade_usd:
            return None

        contracts = int(dollar_amount / (raw_price / 100))
        if contracts <= 0:
            return None

        log.info(
            f"{self.tag} SETTLEMENT LOCK: {ticker} {side.upper()} | "
            f"bsm={output.prob_binary_options:.0%} conf={output.confidence:.0%} "
            f"edge={edge:+.1%} ×{contracts} @ {raw_price}¢ | {secs:.0f}s left"
        )

        return Action(
            action_type="buy",
            ticker=ticker,
            side=side,
            contracts=contracts,
            price_cents=raw_price,
            post_only=False,
            time_in_force="ioc",
            reason=f"settlement_lock bsm={output.prob_binary_options:.0%} {secs:.0f}s",
        )

    # ── GTC escalation ────────────────────────────────────────────────────────

    def _check_gtc_escalation(
        self,
        ticker: str,
        orderbook: dict,
        secs: float,
        annual_vol: float = 0.80,
    ) -> list[Action]:
        """
        If a GTC entry order has been open longer than gtc_escalate_seconds
        without filling, cancel it and place an IOC at the current ask.
        This handles the case where the early-window GTC missed and the market
        moved into the prime window.
        """
        # Never escalate in the late window — too close to settlement
        if secs < self.cfg.late_window_min_seconds:
            return []

        now = time.time()
        actions = []

        # Adaptive slippage by realized vol — same scale as directional entry.
        slip = self.cfg.slippage_cents
        if annual_vol > 1.0:
            slip += 1
        if annual_vol > 1.5:
            slip += 1

        halve_thresh = getattr(self.cfg, "escalation_drift_halve_cents", 2)
        skip_thresh = getattr(self.cfg, "escalation_drift_skip_cents", 5)

        for oid, info in list(self.resting_orders.items()):
            if info.get("ticker") != ticker or info.get("purpose") != "entry":
                continue

            age = now - info.get("placed_at", now)
            if age < self.cfg.gtc_escalate_seconds:
                continue

            side = info["side"]
            contracts = info["contracts"]
            yes_bid = orderbook.get("yes_bid")
            yes_ask = orderbook.get("yes_ask")

            if side == "yes":
                ioc_price = int(float(yes_ask or 99)) + slip
            else:
                ioc_price = int(100 - float(yes_bid or 1)) + slip
            ioc_price = max(1, min(99, ioc_price))

            # Adverse-selection drift gate. If the market moved substantially
            # past our signal-time mid while the GTC rested, informed flow has
            # already priced in what our model just saw — we're now the slow
            # money. Either shrink size or abort.
            signal_mid = info.get("signal_mid_cents")
            if signal_mid is not None:
                drift = ioc_price - float(signal_mid) - slip
                if drift > skip_thresh:
                    log.warning(
                        f"{self.tag} GTC→IOC SKIPPED: {ticker} {side.upper()} "
                        f"order {oid[:8]} — drift={drift:+.1f}¢ > "
                        f"{skip_thresh}¢ (mid@sig={signal_mid:.1f}¢ → IOC={ioc_price}¢). "
                        f"Cancelling GTC, entering retry cooldown."
                    )
                    actions.append(Action(
                        action_type="cancel", ticker=ticker, order_id=oid,
                        reason="gtc_drift_skip",
                    ))
                    # 30s retry cooldown so next scan doesn't re-post a fresh GTC
                    self._entry_retry_cooldown[ticker] = now + 30
                    continue
                elif drift > halve_thresh:
                    original = contracts
                    contracts = max(1, contracts // 2)
                    log.info(
                        f"{self.tag} GTC→IOC DRIFT HALVE: {ticker} {side.upper()} "
                        f"order {oid[:8]} — drift={drift:+.1f}¢ > "
                        f"{halve_thresh}¢ (mid@sig={signal_mid:.1f}¢ → IOC={ioc_price}¢). "
                        f"Size {original} → {contracts}."
                    )

            log.info(
                f"{self.tag} GTC→IOC ESCALATE: {ticker} {side.upper()} | "
                f"order {oid[:8]} age={age:.0f}s → IOC @ {ioc_price}¢ ×{contracts}"
            )

            actions.append(Action(
                action_type="cancel", ticker=ticker, order_id=oid,
                reason="gtc_escalate",
            ))
            actions.append(Action(
                action_type="buy", ticker=ticker, side=side,
                contracts=contracts, price_cents=ioc_price,
                post_only=False, time_in_force="ioc",
                reason=f"gtc_escalated after {age:.0f}s",
                original_price_cents=info.get("price"),  # original GTC price for drift check
                signal_mid_cents=signal_mid,  # preserve for any downstream logging
            ))

        return actions

    # ── Fill / exit recording ─────────────────────────────────────────────────

    def record_fill(
        self,
        ticker: str,
        side: str,
        contracts: int,
        price_cents: int,
        trade_id: str = "",
        mode: str = "directional",
    ):
        """Record that an entry order filled."""
        entry = {
            "side": side, "entry_cents": price_cents,
            "contracts": contracts, "trade_id": trade_id,
            "mode": mode,
        }
        if ticker not in self.positions:
            self.positions[ticker] = []
        # Merge with existing entry on same side + mode (average-in)
        for pos in self.positions[ticker]:
            if pos["side"] == side and pos.get("mode") == mode:
                total = pos["contracts"] + contracts
                pos["entry_cents"] = round(
                    (pos["entry_cents"] * pos["contracts"] + price_cents * contracts) / total
                )
                pos["contracts"] = total
                pos["pyramid_adds"] = pos.get("pyramid_adds", 0) + 1
                self.daily_trades += 1
                return
        self.positions[ticker].append(entry)
        self.daily_trades += 1

    def record_exit(self, ticker: str, pnl: float, side: Optional[str] = None):
        """Record that a position was closed (exit or settlement)."""
        if side and ticker in self.positions:
            self.positions[ticker] = [
                p for p in self.positions[ticker] if p["side"] != side
            ]
            if not self.positions[ticker]:
                del self.positions[ticker]
        else:
            self.positions.pop(ticker, None)
        self.daily_pnl += pnl
        self._ticker_session_pnl[ticker] = (
            self._ticker_session_pnl.get(ticker, 0.0) + pnl
        )

    def record_order(
        self,
        order_id: str,
        ticker: str,
        side: str,
        price: int,
        contracts: int,
        purpose: str = "entry",
        mode: str = "directional",
        signal_mid_cents: Optional[float] = None,
    ):
        """Track a resting GTC order."""
        self.resting_orders[order_id] = {
            "ticker": ticker, "side": side, "price": price,
            "contracts": contracts, "placed_at": time.time(),
            "purpose": purpose, "mode": mode,
            "signal_mid_cents": signal_mid_cents,
        }

    def remove_order(self, order_id: str):
        self.resting_orders.pop(order_id, None)

    def summary(self) -> dict:
        total_pos = sum(len(v) for v in self.positions.values())
        return {
            "name": "auto",
            "positions": total_pos,
            "resting_orders": len(self.resting_orders),
            "daily_pnl": round(self.daily_pnl, 2),
            "daily_trades": self.daily_trades,
        }
