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
class CoreConfig:
    """The v3 decision core. Every field here is a live knob in the policy
    (btc15/core/policy.py) — there are no dead fields, and anything added
    must be justified by measured evidence from core/score.py."""

    # ── 1. Window ─────────────────────────────────────────────────────────
    min_seconds: float = 30.0        # no entries inside the last 30s
    max_seconds: float = 840.0       # wait 60s after open for a real book

    # ── 2. Slices ─────────────────────────────────────────────────────────
    # "phase:band" keys, e.g. "prime:extreme". EMPTY = all slices enabled,
    # which is the correct R1 setting: observe everything, restrict later
    # using `python main.py score --suggest`.
    enabled_slices: list = field(default_factory=list)

    # ── 3. EV gate ────────────────────────────────────────────────────────
    # Required expected value per contract, in cents, AFTER the Kalshi
    # taker fee. This replaces the old flat "edge >= 5%" gate and is
    # automatically strict mid-range (fee peaks at 1.75c) and permissive at
    # the extremes (0.33c) — the structural shape of edge in this market.
    ev_margin_cents: float = 0.75
    min_confidence: float = 0.04     # |p - 0.5| * 2; "have some opinion"
    max_spread_cents: float = 8.0    # wider books can't be priced honestly

    # ── 4. Sizing / risk ──────────────────────────────────────────────────
    kelly_fraction: float = 0.25     # quarter-Kelly on the fee-adjusted edge
    max_single_trade_usd: float = 10.0
    min_single_trade_usd: float = 1.0
    max_per_market_usd: float = 10.0
    max_open_positions: int = 3
    daily_loss_limit_usd: float = 50.0

    # ── Exit (one rule) ───────────────────────────────────────────────────
    # Leave only when the model wants the OTHER side by this much EV and
    # there is still time for the flip to pay. No loss cuts, no profit
    # takes: a binary settles 0/100 and the June 6 post-mortem showed the
    # loss-cut tiers realized losses on positions that settled as wins.
    exit_flip_min_ev_cents: float = 2.0
    exit_min_seconds: float = 120.0

    # ── Execution ─────────────────────────────────────────────────────────
    slippage_cents: int = 1          # limit padding above the displayed ask

    # ── Paper simulation fidelity ─────────────────────────────────────────
    paper_starting_cash_usd: float = 100.0
    # Extra cents paid vs the displayed price, modeling the tick that moves
    # against us between decision and arrival. 0.0 = trust the displayed
    # book; 1.0 = pessimistic. Paper fills always walk real displayed depth.
    paper_adverse_cents: float = 0.0

    # ── Fee model ─────────────────────────────────────────────────────────
    # Kalshi taker fee = ceil_to_cent(rate * multiplier * n * p * (1-p)).
    # 0.07 is the documented standard-category rate. Reports of the July
    # 2026 revision describe per-category multipliers and suggest CRYPTO
    # may price above standard — Kalshi's own docs are unreachable from
    # the build environment, so this is a KNOB, and FeeCalibrator checks
    # it against real fills at runtime and warns if it is wrong.
    # VERIFY BEFORE LIVE TRADING: place one small order, compare the fee
    # Kalshi reports to our model, set fee_rate to the measured value.
    fee_rate: float = 0.07
    fee_multiplier: float = 1.0

    # ── Vol nowcast ───────────────────────────────────────────────────────
    sigma_fast_sec: float = 60.0
    sigma_slow_sec: float = 300.0
    sigma_fast_weight: float = 0.6


@dataclass
class StrategyConfig:
    """Mode flags only. All decision parameters live in CoreConfig — this
    exists so the CLI and dashboard can ask "are we live?" and "are we
    allowed to fire?" without reaching into the policy."""
    auto_trade: bool = False
    paper_trade: bool = True


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
    core: CoreConfig = field(default_factory=CoreConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
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
        cfg.core.max_single_trade_usd = float(v)
        cfg.core.max_per_market_usd = float(v)
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
