"""Kalshi REST API client with auth, retry, and rate limiting."""
from __future__ import annotations

import asyncio
import base64
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from btc15.config import KalshiConfig
from btc15.kalshi.models import (
    Market, MarketStatus, Orderbook, Order, OrderType, Side, OrderStatus,
    Position, PortfolioBalance, Trade, TimeInForce, SelfTradePrevention
)

log = logging.getLogger(__name__)

# Kalshi rate limit: ~10 requests/sec
_RATE_LIMIT_INTERVAL = 0.12


class KalshiAuthError(Exception):
    pass


class KalshiAPIError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


def _load_rsa_key(pem_path: str):
    """Load RSA private key from PEM file."""
    from cryptography.hazmat.primitives import serialization
    path = Path(pem_path)
    if not path.is_absolute():
        path = Path(__file__).parent.parent.parent / pem_path
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _rsa_sign(private_key, timestamp_ms: int, method: str, path: str) -> str:
    """
    Kalshi RSA signature: RSASSA-PSS SHA-256
    Message = str(timestamp_ms) + method.upper() + path_without_query
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    msg = f"{timestamp_ms}{method.upper()}{path}".encode("utf-8")
    sig = private_key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


class KalshiClient:
    """Async Kalshi REST API client."""

    def __init__(self, config: KalshiConfig):
        self.cfg = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._token: Optional[str] = None          # for email/password auth
        self._token_expiry: float = 0.0
        self._rsa_key = None                        # for RSA auth
        self._last_request_time: float = 0.0

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def connect(self):
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={"Content-Type": "application/json"},
        )
        await self._ensure_auth()

    async def close(self):
        if self._session:
            await self._session.close()

    # ── Auth ───────────────────────────────────────────────────────────────

    @property
    def _using_rsa(self) -> bool:
        return bool(self.cfg.api_key and self.cfg.rsa_key_path)

    async def _ensure_auth(self):
        if self._using_rsa:
            if self._rsa_key is None:
                try:
                    self._rsa_key = _load_rsa_key(self.cfg.rsa_key_path)
                    log.info("RSA private key loaded")
                except Exception as e:
                    raise KalshiAuthError(f"Failed to load RSA key from '{self.cfg.rsa_key_path}': {e}")
            return  # RSA signs per-request, no session token needed
        if time.time() < self._token_expiry - 60:
            return
        await self._login()

    async def _login(self):
        if not self.cfg.email or not self.cfg.password:
            raise KalshiAuthError(
                "No auth method available. Add either:\n"
                "  KALSHI_RSA_KEY_PATH=kalshi_private_key.pem  (recommended for Google SSO)\n"
                "  KALSHI_EMAIL + KALSHI_PASSWORD  (for email/password accounts)"
            )
        url = urljoin(self.cfg.base_url + "/", "login")
        async with self._session.post(
            url,
            json={"email": self.cfg.email, "password": self.cfg.password},
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise KalshiAuthError(f"Login failed ({resp.status}): {text}")
            data = await resp.json()
        self._token = data.get("token") or data.get("access_token", "")
        self._token_expiry = time.time() + 23 * 3600
        log.info("Kalshi session token refreshed")

    def ws_auth_headers(self) -> dict:
        """RSA auth headers for the WebSocket upgrade handshake.
        Must be called fresh per connect (timestamp changes)."""
        if self._using_rsa and self._rsa_key:
            from urllib.parse import urlparse as _up
            ts = int(time.time() * 1000)
            ws_path = _up(self.cfg.ws_url).path  # /trade-api/ws/v2
            sig = _rsa_sign(self._rsa_key, ts, "GET", ws_path)
            return {
                "KALSHI-ACCESS-KEY": self.cfg.api_key,
                "KALSHI-ACCESS-SIGNATURE": sig,
                "KALSHI-ACCESS-TIMESTAMP": str(ts),
            }
        return {}

    def _auth_headers(self, method: str = "GET", path: str = "/") -> dict:
        if self._using_rsa and self._rsa_key:
            ts = int(time.time() * 1000)
            # Kalshi signs the FULL path including /trade-api/v2 prefix, no query string
            parsed = urlparse(self.cfg.base_url)
            base_path = parsed.path.rstrip("/")   # e.g. /trade-api/v2
            endpoint = path.split("?")[0]          # strip any query params
            if not endpoint.startswith("/"):
                endpoint = "/" + endpoint
            full_path = base_path + endpoint       # e.g. /trade-api/v2/portfolio/balance
            sig = _rsa_sign(self._rsa_key, ts, method, full_path)
            return {
                "KALSHI-ACCESS-KEY": self.cfg.api_key,
                "KALSHI-ACCESS-SIGNATURE": sig,
                "KALSHI-ACCESS-TIMESTAMP": str(ts),
            }
        elif self._token:
            return {"Authorization": self._token}
        return {}

    # ── Rate-limited request ───────────────────────────────────────────────

    async def _request(
        self, method: str, path: str, **kwargs
    ) -> Any:
        # Simple rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < _RATE_LIMIT_INTERVAL:
            await asyncio.sleep(_RATE_LIMIT_INTERVAL - elapsed)
        self._last_request_time = time.time()

        await self._ensure_auth()
        url = f"{self.cfg.base_url}/{path.lstrip('/')}"
        api_path = f"/{path.lstrip('/')}"
        kwargs.setdefault("headers", {}).update(self._auth_headers(method, api_path))

        async with self._session.request(method, url, **kwargs) as resp:
            if resp.status == 401 and not self._using_rsa:
                # Session token expired — re-login once (email/password auth only)
                await self._login()
                kwargs["headers"].update(self._auth_headers(method, api_path))
                async with self._session.request(method, url, **kwargs) as resp2:
                    return await self._parse(resp2)
            return await self._parse(resp)

    async def _parse(self, resp: aiohttp.ClientResponse) -> Any:
        if resp.status >= 400:
            text = await resp.text()
            raise KalshiAPIError(resp.status, text)
        if resp.status == 204:
            return {}
        return await resp.json()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def get(self, path: str, **params) -> Any:
        return await self._request("GET", path, params=params)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def post(self, path: str, body: dict) -> Any:
        return await self._request("POST", path, json=body)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=5),
        retry=retry_if_exception_type(aiohttp.ClientError),
    )
    async def delete(self, path: str) -> Any:
        return await self._request("DELETE", path)

    # ── Market endpoints ───────────────────────────────────────────────────

    async def get_markets(
        self,
        status: str = "open",
        series_ticker: Optional[str] = None,
        limit: int = 100,
    ) -> list[Market]:
        params: dict = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        data = await self.get("/markets", **params)
        return [self._parse_market(m) for m in data.get("markets", [])]

    async def get_market(self, ticker: str) -> Market:
        data = await self.get(f"/markets/{ticker}")
        return self._parse_market(data["market"])

    async def get_orderbook(self, ticker: str, depth: int = 10) -> Orderbook:
        data = await self.get(f"/markets/{ticker}/orderbook", depth=depth)
        ob = data.get("orderbook", {})
        yes_bids = [(int(p), int(s)) for p, s in ob.get("yes", [])]
        yes_asks = [(int(p), int(s)) for p, s in ob.get("no", [])]  # no side = yes sells
        return Orderbook(
            ticker=ticker,
            yes_bids=yes_bids,
            yes_asks=yes_asks,
            timestamp=datetime.now(timezone.utc),
        )

    def _parse_market(self, m: dict) -> Market:
        close_time_str = m.get("close_time") or m.get("expiration_time", "")
        try:
            close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        except Exception:
            close_time = datetime.now(timezone.utc)

        ticker = m.get("ticker", "")
        strike = self._parse_strike(ticker, m)

        # API returns prices in dollar format (0.42) or cent format (42)
        # Normalize everything to cents (0–100 scale)
        def to_cents(val) -> float:
            if val is None:
                return 0.0
            f = float(val)
            # Dollar format if value is <= 1.0; cent format if > 1.0
            return f * 100 if f <= 1.0 else f

        # New API field names use _dollars suffix; fall back to old names
        yes_bid = to_cents(m.get("yes_bid_dollars") or m.get("yes_bid"))
        yes_ask = to_cents(m.get("yes_ask_dollars") or m.get("yes_ask")) or 100.0
        no_bid  = to_cents(m.get("no_bid_dollars")  or m.get("no_bid"))
        no_ask  = to_cents(m.get("no_ask_dollars")  or m.get("no_ask")) or 100.0
        last    = to_cents(m.get("last_price_dollars") or m.get("last_price")) or 50.0

        # Volume: try fractional field first
        try:
            volume = int(float(m.get("volume_fp") or m.get("volume_24h_fp") or m.get("volume") or 0))
        except Exception:
            volume = 0
        try:
            oi = int(float(m.get("open_interest_fp") or m.get("open_interest") or 0))
        except Exception:
            oi = 0

        return Market(
            ticker=ticker,
            series_ticker=m.get("series_ticker", m.get("event_ticker", "")),
            title=m.get("title", ticker),
            status=MarketStatus(m.get("status", "open")),
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            no_bid=no_bid,
            no_ask=no_ask,
            last_price=last,
            volume=volume,
            open_interest=oi,
            strike_price=strike,
            close_time=close_time,
            result=m.get("result") or None,
        )

    def _parse_strike(self, ticker: str, m: dict) -> float:
        # Try dedicated field first (different API versions)
        for field in ("floor_strike", "cap_strike", "strike_price"):
            if m.get(field):
                try:
                    return float(m[field])
                except Exception:
                    pass
        # Parse from ticker suffix: KXBTC-25JAN1415-T96249.99 → 96249.99
        try:
            parts = ticker.split("-")
            raw = parts[-1].lstrip("T")
            return float(raw)
        except Exception:
            return 0.0

    # ── Portfolio endpoints ────────────────────────────────────────────────

    async def get_balance(self) -> PortfolioBalance:
        data = await self.get("/portfolio/balance")
        return PortfolioBalance(
            available_balance_cents=int(data.get("balance", data.get("available_balance", 0))),
            portfolio_value_cents=int(data.get("portfolio_value", 0)),
        )

    async def get_positions(self) -> list[Position]:
        data = await self.get("/portfolio/positions")
        positions = []
        for p in data.get("market_positions", []):
            positions.append(
                Position(
                    ticker=p["ticker"],
                    side=Side.YES if int(p.get("position", 0)) > 0 else Side.NO,
                    contracts=abs(int(p.get("position", 0))),
                    avg_price_cents=float(p.get("market_exposure", 0))
                    / max(abs(int(p.get("position", 1))), 1) * 100,
                    current_yes_bid=float(p.get("current_yes_bid", 0) or 0),
                    current_yes_ask=float(p.get("current_yes_ask", 100) or 100),
                )
            )
        return positions

    async def get_orders(self, ticker: Optional[str] = None) -> list[Order]:
        params: dict = {}
        if ticker:
            params["ticker"] = ticker
        data = await self.get("/portfolio/orders", **params)
        return [self._parse_order(o) for o in data.get("orders", [])]

    async def get_fills(self, ticker: Optional[str] = None) -> list[dict]:
        params: dict = {}
        if ticker:
            params["ticker"] = ticker
        data = await self.get("/portfolio/fills", **params)
        return data.get("fills", [])

    # ── Order management ───────────────────────────────────────────────────

    async def place_order(
        self,
        ticker: str,
        side: Side,
        contracts: int,
        price_cents: int,
        order_type: OrderType = OrderType.LIMIT,
        client_order_id: Optional[str] = None,
        post_only: bool = False,
        time_in_force: Optional[TimeInForce] = None,
        self_trade_prevention: Optional[SelfTradePrevention] = None,
    ) -> Order:
        body: dict = {
            "ticker": ticker,
            "action": "buy",
            "side": side.value,
            "type": order_type.value,
            "count": contracts,
        }
        if order_type == OrderType.LIMIT:
            if side == Side.YES:
                body["yes_price"] = price_cents
            else:
                body["no_price"] = price_cents
        if client_order_id:
            body["client_order_id"] = client_order_id
        if post_only:
            body["post_only"] = True
        if time_in_force:
            body["time_in_force"] = time_in_force.value
        if self_trade_prevention:
            body["self_trade_prevention"] = self_trade_prevention.value

        data = await self.post("/portfolio/orders", body)
        return self._parse_order(data["order"])

    async def amend_order(
        self,
        order_id: str,
        price_cents: Optional[int] = None,
        side: Optional[Side] = None,
        count: Optional[int] = None,
    ) -> Order:
        """Amend a resting order's price or count in-place (no cancel+replace)."""
        body: dict = {}
        if price_cents is not None and side is not None:
            if side == Side.YES:
                body["yes_price"] = price_cents
            else:
                body["no_price"] = price_cents
        if count is not None:
            body["count"] = count
        data = await self.post(f"/portfolio/orders/{order_id}/amend", body)
        return self._parse_order(data["order"])

    async def batch_place_orders(self, orders: list[dict]) -> list[Order]:
        """Place multiple orders atomically. Each dict should have:
        ticker, action, side, type, count, and optionally yes_price/no_price, post_only."""
        data = await self.post("/portfolio/orders/batched", {"orders": orders})
        return [self._parse_order(o) for o in data.get("orders", [])]

    async def sell_position(
        self,
        ticker: str,
        side: Side,
        contracts: int,
        price_cents: int,
    ) -> Order:
        """Close an existing position by placing a sell-side limit order."""
        body: dict = {
            "ticker": ticker,
            "action": "sell",
            "side": side.value,
            "type": "limit",
            "count": contracts,
        }
        if side == Side.YES:
            body["yes_price"] = price_cents
        else:
            body["no_price"] = price_cents
        data = await self.post("/portfolio/orders", body)
        return self._parse_order(data["order"])

    async def cancel_order(self, order_id: str) -> dict:
        return await self.delete(f"/portfolio/orders/{order_id}")

    async def cancel_all_orders(self, ticker: Optional[str] = None) -> int:
        orders = await self.get_orders(ticker=ticker)
        cancelled = 0
        for order in orders:
            if order.status == OrderStatus.RESTING:
                try:
                    await self.cancel_order(order.order_id)
                    cancelled += 1
                except Exception as e:
                    log.warning(f"Failed to cancel {order.order_id}: {e}")
        return cancelled

    def _parse_order(self, o: dict) -> Order:
        return Order(
            order_id=o.get("order_id", ""),
            ticker=o.get("ticker", ""),
            side=Side(o.get("side", "yes")),
            order_type=OrderType(o.get("type", "limit")),
            count=int(o.get("count", 0)),
            yes_price=int(o.get("yes_price", 0) or 0),
            no_price=int(o.get("no_price", 0) or 0),
            status=OrderStatus(o.get("status", "resting")),
            filled_count=int(o.get("filled_count", 0) or 0),
            remaining_count=int(o.get("remaining_count", 0) or 0),
        )
