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
    min_confidence: float = 0.62
    min_edge: float = 0.04
    ensemble_weights: dict = field(default_factory=lambda: {
        "binary_options_model": 0.40,
        "technical_momentum": 0.30,
        "trend_regression": 0.20,
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
    min_seconds_remaining: int = 90
    max_seconds_remaining: int = 840
    preferred_entry_window_sec: list = field(default_factory=lambda: [300, 600])
    auto_trade: bool = False
    paper_trade: bool = True
    max_open_positions: int = 3
    allowed_sides: str = "both"
    # Active position management
    take_profit_pct: float = 0.65
    stop_loss_pct: float = 0.35
    flip_min_edge: float = 0.10
    lock_profit_seconds: int = 120
    # Trailing stop — activates once up trail_activate_pct, exits if retraces
    # more than trail_retracement_pct of the peak gain
    trail_activate_pct: float = 0.20   # start trailing once up 20%
    trail_retracement_pct: float = 0.50  # exit if peak gain retraces 50%
    # Re-entry protection
    min_entry_price_cents: int = 10       # skip if contract priced < 10¢ (market decided)
    stop_loss_cooldown_seconds: int = 120  # block re-entry for 2 min after stop-loss
    # Cheap-entry cost cap — Kelly explodes at penny prices; cap dollar exposure
    cheap_entry_threshold_cents: int = 25  # applies when price < this
    max_cost_cheap_entry_usd: float = 4.00
    # Per-candle loss brake — stop auto entries on a ticker that has lost this much
    per_candle_max_loss_usd: float = 8.00


@dataclass
class RiskConfig:
    max_trade_usd: float = 50.00
    max_position_per_market_usd: float = 200.00
    daily_loss_limit_usd: float = 150.00
    kelly_fraction: float = 0.25
    min_trade_usd: float = 5.00
    win_rate_lookback: int = 20
    win_rate_min: float = 0.45
    max_open_positions: int = 3


@dataclass
class SniperConfig:
    enabled: bool = True
    budget_pct: float = 0.50
    min_confidence: float = 0.75
    min_edge: float = 0.05
    kelly_fraction: float = 0.50
    slippage_cents: int = 3
    take_profit_pct: float = 0.50
    stop_loss_pct: float = 0.25
    min_seconds: int = 180       # 3 min
    max_seconds: int = 600       # 10 min
    stop_loss_cooldown_seconds: int = 180  # no re-entry on same ticker for 3 min after SL
    trail_activate_pct: float = 0.20
    trail_retracement_pct: float = 0.50


@dataclass
class ScalperConfig:
    enabled: bool = True
    budget_pct: float = 0.30
    min_spread_cents: int = 3
    contracts_per_side: int = 1
    max_inventory_imbalance: int = 10
    cancel_before_seconds: int = 120
    min_seconds: int = 120       # 2 min
    max_seconds: int = 720       # 12 min


@dataclass
class ArbConfig:
    enabled: bool = True
    budget_pct: float = 0.20
    min_arb_cents: int = 2       # YES+NO must cost < 98¢
    stat_arb_divergence: float = 0.15
    stat_arb_min_confidence: float = 0.40
    max_contracts: int = 20
    # Stat-arb entry filter
    stat_arb_min_price_cents: int = 20       # skip contracts already priced <20¢ (market has decided)
    # Stat-arb exit thresholds (pure arb always holds to settlement)
    stat_arb_take_profit_pct: float = 0.80   # exit when up 80%
    stat_arb_stop_loss_pct: float = 0.65     # exit when down 65% (widened from 50%)
    trail_activate_pct: float = 0.20
    trail_retracement_pct: float = 0.50
    stat_arb_cut_before_seconds: int = 90    # cut losing stat-arb with <90s left rather than ride to zero


@dataclass
class PersonasConfig:
    sniper: SniperConfig = field(default_factory=SniperConfig)
    scalper: ScalperConfig = field(default_factory=ScalperConfig)
    arb: ArbConfig = field(default_factory=ArbConfig)


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
    risk: RiskConfig = field(default_factory=RiskConfig)
    personas: PersonasConfig = field(default_factory=PersonasConfig)
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
    path = config_path or ROOT / "config.yaml"
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
