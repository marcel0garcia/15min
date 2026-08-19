"""Configuration loader — merges config.yaml + .env overrides."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

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
    # These three were declared here from the start but never reached the
    # engine — it called blended_sigma() with hard-coded defaults. They are
    # wired now; so are the clamp bounds, which were buried as module
    # constants in vol_nowcast.py.
    sigma_fast_sec: float = 60.0
    sigma_slow_sec: float = 300.0
    sigma_fast_weight: float = 0.6

    # THE FLOOR IS THE MOST DANGEROUS NUMBER IN THIS FILE. A floored sigma
    # understates settlement variance, which drives P(YES) toward 0/1 at
    # exactly the extreme strikes we want to trade. The 19AUG session ran
    # 27% of its scans at the 0.20 floor and priced 27.5% of rows beyond
    # 0.999 while the market quoted 0.99. BTC realized vol is typically
    # 0.40-0.60 annualized. Sweep this; do not trust the inherited value.
    sigma_floor: float = 0.20
    sigma_ceiling: float = 5.0
    sigma_min_samples: int = 10
    # Multiplicative correction applied after blending, before the clamp.
    # >1 widens the distribution (humbler probabilities), <1 sharpens it.
    # The one knob that trades conviction against calibration; set it from
    # `score --calibration`, not from intuition.
    sigma_scale: float = 1.0

    # ── Slice grid ────────────────────────────────────────────────────────
    # Phase and band boundaries define the (phase x band) buckets that R1
    # measures and R2 switches on. They were hard-coded in pricer.py, which
    # made the entire R1/R2 methodology depend on three unmeasured numbers.
    # Seconds remaining at each phase boundary:
    phase_early_sec: float = 540.0    # > this  -> "early"
    phase_mid_sec: float = 300.0      # > this  -> "mid"
    phase_prime_sec: float = 90.0     # > this  -> "prime", else "late"
    # Distance from the nearer end of the price range, in cents:
    band_extreme_cents: float = 15.0  # <= this -> "extreme"
    band_outer_cents: float = 30.0    # <= this -> "outer", else "middle"

    # ── Vol-nowcast sanity guards ─────────────────────────────────────────
    # Observed 2026-08-19: the engine entered 0.2s after startup on a sigma
    # pinned to its floor (no BRTI history), pricing 0.996 against a market
    # at 0.905, then flipped out four seconds later at a loss once sigma
    # caught up. Both legs were sigma noise. These guards close that.
    warmup_sec: float = 120.0          # BRTI history required before entering
    reject_clamped_sigma: bool = True  # never enter while sigma sits at a clamp

    # ── Entry cadence ─────────────────────────────────────────────────────
    # KXBTC15M lists exactly ONE market at a time (verified against the live
    # series: open_time == the previous window's close_time), so the ceiling
    # is 96 markets/day and max_open_positions can never bind. That makes
    # this the only lever that can lift trade count above one per window:
    # how many separate entries we are willing to make in the same market.
    # 1 preserves the original hold-to-settle behavior exactly.
    max_entries_per_market: int = 1
    # Seconds to wait after closing a position before re-entering the same
    # market. Only meaningful when max_entries_per_market > 1.
    entry_cooldown_sec: float = 0.0

    # ── Engine cadences ───────────────────────────────────────────────────
    # Were module constants in core/engine.py. Scan interval in particular
    # bounds how fast we can react inside the final minute, where the TWAP
    # lock edge lives.
    scan_interval_sec: float = 1.0
    ob_refresh_interval_sec: float = 4.0
    settlement_check_interval_sec: float = 5.0
    brti_hz: float = 4.0
    venue_staleness_sec: float = 5.0


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
    # Raw Kalshi WS frames are ~0.5 GB/hour — 12 GB/day, which makes 24/7
    # recording impossible. They exist so a book can be reconstructed
    # offline; now that every decision row carries the top of both sides
    # (DECISION_BOOK_LEVELS), the sweep no longer needs them. Turn this off
    # for long unattended runs; turn it on when you specifically want
    # microstructure detail between scans.
    kalshi_frames: bool = True


@dataclass
class AppConfig:
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)
    feeds: FeedsConfig = field(default_factory=FeedsConfig)
    core: CoreConfig = field(default_factory=CoreConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    database_path: str = "data/btc15.db"
    # Free-text label for this run, stamped into the session's meta.json.
    # The agent uses it to group A/B arms; nothing in the engine reads it.
    session_tag: Optional[str] = None


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


def _coerce(current, raw: str):
    """Cast a CLI string to the type the existing field already has.

    Deliberately type-directed rather than guess-the-literal: if the field
    is a float today, "0.5" must not silently become an int and change
    downstream arithmetic.
    """
    if isinstance(current, bool):
        low = raw.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, (list, tuple, set)):
        raw = raw.strip()
        if not raw:
            return []
        return [x.strip() for x in raw.split(",") if x.strip()]
    return raw


def apply_overrides(cfg: AppConfig, pairs: Iterable[str]) -> list[str]:
    """Apply `section.field=value` overrides in place; return what changed.

    This exists so a research session never has to edit config.yaml. The
    sweep runs many configs concurrently and the agent starts sessions in
    parallel — mutating a shared file for that is a race, and worse, it
    loses the record of what a given session actually ran under. Overrides
    are hashed into the session's meta.json like any other config.
    """
    changed: list[str] = []
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects section.field=value, got {pair!r}")
        path, raw = pair.split("=", 1)
        parts = [p for p in path.strip().split(".") if p]
        if not parts:
            raise ValueError(f"--set has an empty field path: {pair!r}")
        target = cfg
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"unknown config section {part!r} in {pair!r}")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise ValueError(f"unknown config field {leaf!r} in {pair!r}")
        before = getattr(target, leaf)
        after = _coerce(before, raw)
        setattr(target, leaf, after)
        changed.append(f"{path}: {before!r} -> {after!r}")
    return changed


# Global singleton — populated at startup
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = load_config()
    return _config
