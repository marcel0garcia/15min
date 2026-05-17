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


@dataclass
class ModelsConfig:
    min_confidence: float = 0.55
    ensemble_weights: dict = field(default_factory=lambda: {
        "orderbook_imbalance": 0.25,
        "technical_momentum": 0.35,
        "trend_regression": 0.20,
        "binary_options_model": 0.10,
        "ml_model": 0.10,
    })
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
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
    min_entry_price_cents: int = 10      # skip contracts priced below 10¢
    max_entry_price_cents: int = 72      # skip deep-ITM contracts — post-mortem showed
                                         # 67-82¢ entries dominated losses; 72¢ drops the tail

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
    mm_min_spread_cents: int = 5         # minimum spread to post both sides
    mm_contracts_per_side: int = 2       # contracts per MM quote
    mm_max_inventory: int = 8            # max net directional inventory before pausing one side
    mm_cancel_before_seconds: int = 120  # cancel all MM orders with <2 min left
    mm_quote_offset_cents: int = 2       # quote this many cents inside the best bid/ask

    # ── Exit rules ────────────────────────────────────────────────────────
    stop_loss_pct: float = 0.40          # cut if losing >40% AND inside stop_loss_min_seconds
    stop_loss_min_seconds: int = 240     # only cut if <4 min left (let positions breathe)
    emergency_stop_pct: float = 0.65    # cut IMMEDIATELY (any time) if losing >65%
    reversal_min_edge: float = 0.10      # edge required to flip sides
    reversal_min_seconds: int = 300      # only flip with >5 min left

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
class AppConfig:
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    feeds: FeedsConfig = field(default_factory=FeedsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    trader: TraderConfig = field(default_factory=TraderConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
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

    return cfg


# Global singleton — populated at startup
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
