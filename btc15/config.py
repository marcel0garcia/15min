"""Configuration loader — merges config.yaml + .env overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")


@dataclass
class KalshiConfig:
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    api_key: Optional[str] = None          # Key ID (UUID)
    rsa_key_path: Optional[str] = None     # Path to PEM private key file
    email: Optional[str] = None
    password: Optional[str] = None
    series_ticker: str = "KXBTC15M"
    order_type: str = "limit"
    limit_slippage_cents: int = 2


@dataclass
class FeedsConfig:
    binance_ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    coinbase_rest_url: str = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
    kraken_rest_url: str = "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
    bar_interval_sec: int = 60
    lookback_bars: int = 200
    # Phase 3: which source feeds the ensemble's current_price + bars.
    #   "brti"     consolidated median of Coinbase/Kraken/Bitstamp top-of-book
    #              (same algorithm as recording.brti). Default — what KXBTC
    #              actually settles against.
    #   "coinbase" single-venue Coinbase Exchange WS (legacy path; preserved
    #              as a fallback / A-B comparison option).
    price_source: str = "brti"


@dataclass
class ModelsConfig:
    min_confidence: float = 0.55
    # Ensemble weights — sum to 1.0. ml_model carries the trained LightGBM
    # prediction at 0.10; binary_options_model (BSM) was downweighted to
    # 0.00 because ML's training data already encodes Black-Scholes-style
    # information at a much finer resolution than the analytic BSM call.
    # The 0.10 freed up here is reallocated to ml_model.
    ensemble_weights: dict = field(default_factory=lambda: {
        "orderbook_imbalance": 0.25,
        "technical_momentum": 0.35,
        "trend_regression": 0.20,
        "binary_options_model": 0.00,
        "ml_model": 0.10,
    })
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0

    # ── EWMA signal smoothing — DISABLED BY DEFAULT ──────────────────────
    # EWMA was originally added to filter the per-tick noise from the fast
    # ensemble components (orderbook, BSM) when the slow components
    # (RSI/MACD/BB, trend, ML) were frozen between 1-min bar closes.
    #
    # After commit 4b69aa8 (per-second technical indicators via live
    # partial bar) the slow components also update every scan, providing
    # natural smoothing through component diversity. EWMA's denoising job
    # became redundant — empirically (May 28 sessions, EWMA off) the
    # ensemble output was stable enough without it.
    #
    # Half-life ≈ ln(2) / -ln(1-α) seconds at 1s scan interval:
    #   α=0.00 → disable (recommended; smoothed = raw passthrough)
    #   α=0.10 → half-life 6.6s
    #   α=0.20 → half-life 3.1s
    #   α=0.30 → half-life 2.0s
    # Flip back to a positive value if a future session shows excessive
    # tick-to-tick noise in conf/edge.
    signal_smoothing_alpha: float = 0.00
    # If we haven't seen this ticker for this many seconds, treat as cold
    # start and reset smoothed = raw. Prevents blending stale state across
    # WS gaps / disconnects.
    signal_smoothing_stale_sec: float = 5.0
    default_annual_vol: float = 0.80


@dataclass
class StrategyConfig:
    min_seconds_remaining: int = 60
    max_seconds_remaining: int = 870
    preferred_entry_window_sec: list = field(default_factory=lambda: [180, 600])
    auto_trade: bool = False
    paper_trade: bool = True
    max_open_positions: int = 4
    allowed_sides: str = "both"


@dataclass
class TraderConfig:
    """Unified auto-trader configuration (replaces Sniper/Scalper/Arb personas)."""
    enabled: bool = True
    budget_pct: float = 1.0              # fraction of bankroll available

    # ── Time-phase thresholds (seconds remaining) ──────────────────────────
    early_window_min_seconds: int = 480  # >8 min: market-make + pre-position GTC
    prime_window_min_seconds: int = 180  # 3–8 min: directional entries
    late_window_min_seconds: int = 60    # <1 min: no new entries
    max_entry_seconds: int = 870         # never enter beyond 14.5 min remaining

    # ── Entry thresholds ──────────────────────────────────────────────────
    min_confidence: float = 0.55         # minimum model confidence for any entry
    min_edge: float = 0.05               # minimum edge over market price (5%)
    min_entry_price_cents: int = 10      # skip contracts priced below this (fallback when by_phase empty)
    max_entry_price_cents: int = 72      # skip contracts priced above this (fallback when by_phase empty)

    # Phase-aware entry-price gates. Overrides the flat min/max above on a
    # per-phase basis. Empty dict {} falls back to the flat values.
    # Derived from 1,314-position tape audit (May 28). Early window keeps a
    # tight 10-60¢ band per user trader-intuition input; mid/late open up as
    # market crystallizes and prices become more informative.
    # Keys map to the same secs_remaining boundaries as min_confidence_by_phase.
    entry_price_by_phase: dict = field(default_factory=lambda: {
        "early": {"min": 10, "max": 60},   # >540s — first 6 min, market hasn't crystallized
        "mid":   {"min": 35, "max": 80},   # 300-540s — directional consensus forming
        "prime": {"min": 20, "max": 85},   # 180-300s — late but pre-settle
        "late":  {"min": 10, "max": 95},   # <180s — settlement near-certain at extremes
    })

    # ── Settlement lock (late-window near-certainty entries) ──────────────
    settlement_lock_enabled: bool = True
    settlement_lock_min_seconds: int = 20    # earliest: 20s before close
    settlement_lock_max_seconds: int = 60    # latest:  60s before close (below late window)
    settlement_lock_min_prob: float = 0.88   # BSM prob must be this extreme
    settlement_lock_min_confidence: float = 0.50  # model agreement required

    # ── GTC order management ──────────────────────────────────────────────
    gtc_escalate_seconds: float = 25.0   # escalate GTC entry to IOC after N seconds unfilled
    slippage_cents: int = 2              # IOC slippage above current ask

    # ── Market making ─────────────────────────────────────────────────────
    # Existing (conservative) knobs:
    mm_min_spread_cents: int = 5         # minimum spread to post both sides (used when mm_aggressive=false)
    mm_contracts_per_side: int = 2       # contracts per MM quote (used when mm_aggressive=false)
    mm_max_inventory: int = 8            # legacy NET inventory cap (still honored when mm_aggressive=false)
    mm_cancel_before_seconds: int = 120  # cancel all MM orders with <2 min left
    mm_quote_offset_cents: int = 2       # quote this many cents inside the best bid/ask

    # Aggressive-mode knobs (only active when mm_aggressive=true):
    # When true, MM uses the aggressive thresholds below INSTEAD OF the conservative
    # defaults above. Also enables concurrent firing with directional entry (the
    # mutual-exclusivity short-circuit in evaluate() is bypassed).
    # Default is false — pulling code is safe; flip in config.yaml to start
    # the experiment.
    mm_aggressive: bool = False
    mm_aggressive_min_spread_cents: int = 3      # was 5 — loosen to catch more markets
    mm_aggressive_contracts_per_side: int = 4    # was 2 — bigger quote, more spread captured per fill
    mm_window_min_seconds: int = 240             # was 480 (early_window) — extend to mid window too
    # Hard inventory caps. Per-side guards against single-side runaway during
    # persistent directional moves; net keeps overall exposure bounded.
    mm_per_side_max_inventory: int = 10
    mm_max_inventory_net: int = 15
    # Pre-emptive cancel when BTC moves > this many cents/sec — adverse-selection
    # insurance. Computed from the price_feed tick stream.
    mm_volatility_cancel_threshold_cents_per_sec: float = 5.0

    # ── Exit rules ────────────────────────────────────────────────────────
    emergency_stop_pct: float = 0.65    # cut IMMEDIATELY (any time) if losing >65%
    reversal_min_edge: float = 0.10      # edge required to flip sides
    reversal_min_seconds: int = 300      # only flip with >5 min left

    # ── Loss-cut cool-off (panic-flush guard) ─────────────────────────────
    # When a loss_cut would fire, we wait N seconds and re-check the condition.
    # If model returns to agreeing OR pnl recovers above stop_thresh, the cut
    # is cleared. Calibrated for the 1s scan interval — see commit d0a13be.
    cool_off_seconds_high_runway: float = 10.0  # used when >480s remaining to settle
    cool_off_seconds_mid_runway: float = 5.0    # used when 240-480s remaining
    # Below 240s remaining, cool-off is always 0 (late window = decisive cut).

    # ── Entry signal persistence (anti-noise gate for fresh entries) ──────
    # Under 1s scan, the model's tick-by-tick output is noisier than under
    # the old 3s scan (which implicitly required signals to hold for 3s just
    # to be observed). Require fresh entries to clear the conf+edge gate for
    # K consecutive scans before firing. Does NOT apply to reversal re-entries
    # (already double-gated by reversal_min_edge) or pyramid adds (existing
    # position implies the original signal was strong). K=1 = fire immediately
    # (old behavior); K=2 = ~2s of confirmation; K=3 = ~3s.
    entry_confirmation_ticks: int = 2

    # ── Position sizing ───────────────────────────────────────────────────
    kelly_fraction_early: float = 0.25   # quarter-Kelly for GTC early entries
    kelly_fraction_prime: float = 0.50   # half-Kelly for IOC prime-window entries
    kelly_fraction_strong: float = 0.75  # 3/4-Kelly for very strong signals
    max_single_trade_usd: float = 12.0   # hard cap per individual trade ($100 bankroll)
    min_single_trade_usd: float = 1.0    # skip if Kelly size falls below this

    # ── Pyramiding (add-to-winner) ────────────────────────────────────────
    pyramid_enabled: bool = True
    pyramid_min_pnl_pct: float = 0.10    # position must be ≥10% profitable
    pyramid_min_confidence: float = 0.55 # model must still be confident
    pyramid_min_edge: float = 0.05       # edge must still exist
    pyramid_min_seconds: int = 300       # ≥5 min left to add
    pyramid_max_adds: int = 1            # max 1 add per position

    # ── Stop-loss / whipsaw cooldowns ─────────────────────────────────────
    stop_cooldown_seconds: int = 90      # after a stop-loss, lock the ticker for N seconds
                                         # (was 45; session 12APR06:15 showed 4 stops in 10 min
                                         # on one market — doubling prevents re-entering chop)
    reversal_cooldown_seconds: int = 60  # after a reversal exit, lock the ticker for N seconds
                                         # (blocks second/third flips inside one window)

    # ── GTC→IOC escalation drift gate ─────────────────────────────────────
    # When an early-window GTC sits unfilled and the market has moved past our
    # resting price, measure how far the IOC would have to cross vs the mid
    # we saw at signal time. Adverse drift = informed flow already moved us.
    escalation_drift_halve_cents: int = 2  # IOC price > signal_mid + slip + 2 → halve size
    escalation_drift_skip_cents: int = 5   # IOC price > signal_mid + 5       → skip entirely

    # ── Reversal re-entry orderbook confirmation ──────────────────────────
    # After a reversal exit, don't re-enter the flipped side unless the live
    # orderbook imbalance agrees. Half of last session's reversal re-entries
    # lost immediately — the edge existed on model but the flow hadn't turned.
    reversal_require_orderbook_confirm: bool = True
    reversal_orderbook_min_dev: float = 0.10  # |prob_orderbook - 0.5| must exceed this
                                              # in the flip direction to allow re-entry

    # ── Phase-conditional entry confidence floor ──────────────────────────
    # Empirical settlement-conditional probabilities (tools/friday_market_tape.py
    # --dynamics over 737 markets) suggest signal quality varies by phase:
    # the 6-10min window has the cleanest directional moves, while 0-6min is
    # noisier and the final 3 min is decisive but thin-book-risky.
    # Defaults are conservative tightenings vs. flat 0.50; lower individual
    # values to "loosen" a phase or set the dict to {} to fall back to the
    # flat trader.min_confidence floor.
    min_confidence_by_phase: dict = field(default_factory=lambda: {
        "early": 0.55,    # secs > 540  (0-6 min from open: noisier)
        "mid":   0.48,    # secs 300-540 (6-10 min: best-WR window per dynamics data)
        "prime": 0.50,    # secs 180-300 (10-12 min: current flat default)
        "late":  0.55,    # secs < 180  (decisive but thin-book — tighten)
    })

    # ── Confidence-trend filter (denoise the 3-tick gate) ─────────────────
    # Beyond requiring K consecutive scans of clearance, also require the
    # confidence to NOT deteriorate by more than this many points from the
    # pending-window peak. Catches the "spiked then faded" pattern where
    # tick1=75, tick2=83, tick3 (fire)=57 — currently fires; with this gate
    # it would be rejected as a fading signal. Set to 1.0 to disable.
    entry_conf_fade_max: float = 0.05  # 5 percentage points

    # ── Raw-floor guard against EWMA dip-buoying ──────────────────────────
    # EWMA smoothing is symmetric — it pulls toward the moving average in
    # BOTH directions. That's intended for filtering brief noise SPIKES
    # (raw 80 % smoothed 60 → blocked). It's NOT intended for buoying
    # fading signals (raw 40 % smoothed 59 → fires anyway). The 25MAY22:07
    # session showed 2 of 14 fires came from this dip-buoying mode and
    # both lost. This guard rejects entries where smoothed cleared the
    # floor only because of stale-strong prior observations.
    #
    # Gate: when smoothing is active, also require
    #   raw_confidence >= (phase_min_confidence - smoothing_raw_margin)
    # Default margin of 0.10 means raw can drop 10pp below the smoothed
    # floor and we'll still trust it; further than that, we treat the
    # signal as dead even if smoothed says otherwise.
    # Set to 1.0 to disable (allow any raw value).
    # Set to 0.0 for the strictest version (raw must also clear floor).
    smoothing_raw_margin: float = 0.10

    # ── Trade-tape flow alignment gate (aggressive-taker direction) ───────
    # On every directional entry, sample the last `trade_flow_window_seconds`
    # of public trades for the ticker. The taker side of those trades
    # represents aggressive demand — if our intended side is taking less than
    # `trade_flow_required_alignment` of the recent volume, reject the entry
    # as misaligned with the prevailing tape. Skipped entirely when the
    # window has fewer than `trade_flow_min_volume` total contracts (no signal).
    # Set required_alignment to 0.0 to disable the gate (A/B testing).
    trade_flow_window_seconds: float = 30.0
    trade_flow_required_alignment: float = 0.30
    trade_flow_min_volume: float = 5.0

    # ── Arb ───────────────────────────────────────────────────────────────
    arb_enabled: bool = True             # master switch; disable for clean first-live debut
    min_arb_cents: int = 2               # YES+NO must cost ≤98¢ for guaranteed arb
    max_arb_contracts: int = 5           # max contracts per pure-arb pair


@dataclass
class RiskConfig:
    max_trade_usd: float = 12.00
    max_position_per_market_usd: float = 12.00
    daily_loss_limit_usd: float = 15.00
    kelly_fraction: float = 0.25
    min_trade_usd: float = 1.00
    win_rate_lookback: int = 20
    win_rate_min: float = 0.40
    max_open_positions: int = 3


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = "logs/bot.log"
    trade_log_file: str = "logs/trades.csv"
    max_log_size_mb: int = 50


@dataclass
class RecordingConfig:
    enabled: bool = True
    path: str = "data/recordings"
    venue_coinbase: bool = True
    venue_kraken: bool = True
    venue_bitstamp: bool = True
    venue_gemini: bool = False
    grid_interval_sec: float = 1.0
    venue_max_msg_per_sec: int = 10


@dataclass
class AppConfig:
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    feeds: FeedsConfig = field(default_factory=FeedsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    trader: TraderConfig = field(default_factory=TraderConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    database_path: str = "data/btc15.db"


def _merge(dataclass_instance, yaml_dict: dict):
    """Recursively update a dataclass from a dict."""
    for key, value in yaml_dict.items():
        if hasattr(dataclass_instance, key):
            attr = getattr(dataclass_instance, key)
            if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
                _merge(attr, value)
            else:
                setattr(dataclass_instance, key, value)


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    cfg = AppConfig()

    # Load YAML
    path = Path(config_path) if config_path else ROOT / "config.yaml"
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        _merge(cfg, raw)

    # Load secrets from env
    cfg.kalshi.api_key = os.getenv("KALSHI_API_KEY") or cfg.kalshi.api_key
    cfg.kalshi.rsa_key_path = os.getenv("KALSHI_RSA_KEY_PATH") or cfg.kalshi.rsa_key_path
    cfg.kalshi.email = os.getenv("KALSHI_EMAIL") or cfg.kalshi.email
    cfg.kalshi.password = os.getenv("KALSHI_PASSWORD") or cfg.kalshi.password

    # Env overrides for key risk params
    if v := os.getenv("BTC15_MAX_TRADE_USD"):
        cfg.risk.max_trade_usd = float(v)
    if v := os.getenv("BTC15_AUTO_TRADE"):
        cfg.strategy.auto_trade = v.lower() in ("1", "true", "yes")
    if v := os.getenv("BTC15_PAPER_TRADE"):
        cfg.strategy.paper_trade = v.lower() in ("1", "true", "yes")

    # Ensure directories exist
    Path(cfg.logging.log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.logging.trade_log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.database_path).parent.mkdir(parents=True, exist_ok=True)
    if cfg.recording.enabled:
        Path(cfg.recording.path).mkdir(parents=True, exist_ok=True)

    return cfg


# Global singleton — populated at startup
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
