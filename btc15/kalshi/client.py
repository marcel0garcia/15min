"""Kalshi REST API client with auth, retry, and rate limiting."""
from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
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
    Salt length must equal the hash digest size (32 bytes for SHA-256).
    Kalshi rejects signatures produced with PSS.MAX_LENGTH.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    msg = f"{timestamp_ms}{method.upper()}{path}".encode("utf-8")
    # Fresh SHA256 instances for each use — sign() consumes the algorithm object.
    # salt_length = digest_size (32 bytes) as required by Kalshi; PSS.MAX_LENGTH is rejected.
    sig = private_key.sign(
        msg,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=hashes.SHA256().digest_size,
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
        # API v2 uses "orderbook_fp" with "yes_dollars"/"no_dollars" string arrays.
        # Fall back to legacy "orderbook" with integer "yes"/"no" arrays.
        ob_fp = data.get("orderbook_fp", {})
        ob_legacy = data.get("orderbook", {})

        if ob_fp:
            # Current format: [[price_dollars_str, size_fp_str], ...]
            yes_bids = [
                (int(round(float(p) * 100)), int(round(float(s))))
                for p, s in ob_fp.get("yes_dollars", [])
            ]
            yes_asks = [
                (100 - int(round(float(p) * 100)), int(round(float(s))))
                for p, s in ob_fp.get("no_dollars", [])
            ]
        else:
            # Legacy format: [[price_cents_int, size_int], ...]
            yes_bids = [(int(p), int(s)) for p, s in ob_legacy.get("yes", [])]
            yes_asks = [(100 - int(p), int(s)) for p, s in ob_legacy.get("no", [])]

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
        yes_ask = to_cents(m.get("yes_ask_dollars") or m.get("yes_ask")) or 0.0
        no_bid  = to_cents(m.get("no_bid_dollars")  or m.get("no_bid"))
        no_ask  = to_cents(m.get("no_ask_dollars")  or m.get("no_ask")) or 0.0
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

        def _cents(val, default: float) -> float:
            """Normalize a price value that may be in dollar (0-1) or cent (0-100) format."""
            try:
                f = float(val or default)
                return f * 100 if f <= 1.0 else f
            except Exception:
                return float(default)

        for p in data.get("market_positions", []):
            # v2 API uses position_fp (string float): positive = net YES, negative = net NO
            pos_fp = float(p.get("position_fp", 0) or 0)
            contracts = abs(round(pos_fp))
            if contracts == 0:
                continue  # no active position

            side = Side.YES if pos_fp > 0 else Side.NO
            # market_exposure_dollars = net dollars at risk in the current position.
            # total_traded_dollars = gross trading volume (both YES and NO sides summed)
            # which can exceed 100¢ per net contract when both sides are held (e.g. scalper).
            exposure_dollars = float(p.get("market_exposure_dollars", 0) or 0)
            avg_price_cents = min(99.0, (exposure_dollars / max(contracts, 1)) * 100)

            positions.append(
                Position(
                    ticker=p["ticker"],
                    side=side,
                    contracts=contracts,
                    avg_price_cents=avg_price_cents,
                    current_yes_bid=_cents(p.get("current_yes_bid"), 0),
                    current_yes_ask=_cents(p.get("current_yes_ask"), 100),
                )
            )
        return positions

    async def get_orders(self, ticker: Optional[str] = None) -> list[Order]:
        params: dict = {}
        if ticker:
            params["ticker"] = ticker
        # Read endpoint stays on legacy /portfolio/orders — V2 either doesn't
        # expose a list-orders GET at /portfolio/events/orders (404) or uses
        # a different query convention. The deprecation wave targets the
        # write surface; legacy GET still works and is needed by the
        # reconciler to verify resting orders.
        data = await self.get("/portfolio/orders", **params)
        return [self._parse_order(o) for o in data.get("orders", [])]

    async def get_fills(self, ticker: Optional[str] = None) -> list[dict]:
        """
        Return normalized fills. Kalshi REST uses different field names than WS:
          REST: count_fp (str float), yes_price_dollars / no_price_dollars (str $)
          WS:   count (int),          yes_price / no_price (int cents)
        This method normalizes to: count (int), yes_price (int cents), no_price (int cents),
        plus raw fields preserved.
        """
        params: dict = {}
        if ticker:
            params["ticker"] = ticker
        data = await self.get("/portfolio/fills", **params)
        fills = data.get("fills", [])
        normalized = []
        for f in fills:
            count = round(float(f.get("count_fp", 0) or 0))
            yes_price = round(float(f.get("yes_price_dollars", 0) or 0) * 100)
            no_price  = round(float(f.get("no_price_dollars",  0) or 0) * 100)
            fee_cents = round(float(f.get("fee_cost", 0) or 0) * 100)
            normalized.append({
                **f,
                "count":     count,
                "yes_price": yes_price,
                "no_price":  no_price,
                "fee_cents": fee_cents,
            })
        return normalized

    # ── Order management ───────────────────────────────────────────────────

    @staticmethod
    def _price_dollars(cents: int) -> str:
        """Convert integer cents (1-99) to Kalshi fixed-point dollars string."""
        return f"{int(cents) / 100:.2f}"

    # ── V2 endpoint conventions ──────────────────────────────────────────
    # Kalshi V2 (/portfolio/events/orders) uses a single-book bid/ask model:
    # "bid is equivalent to yes, ask is equivalent to no" per the API changelog.
    # The legacy `action: buy/sell` field is gone — buy vs sell is inferred from
    # position state + the reduce_only flag.
    _V2_SIDE = {Side.YES: "bid", Side.NO: "ask"}
    _V2_BASE = "/portfolio/events/orders"

    @staticmethod
    def _count_fp(n: int) -> str:
        """V2 expects count as a fixed-point string with up to 2 decimals."""
        return f"{int(n):d}.00"

    def _v2_order_body(
        self,
        ticker: str,
        side: Side,
        contracts: int,
        price_cents: int,
        client_order_id: str,
        time_in_force: TimeInForce,
        self_trade_prevention: SelfTradePrevention,
        post_only: bool = False,
        reduce_only: bool = False,
        expiration_time: Optional[int] = None,
    ) -> dict:
        """Build a V2 create-order request body. Returns a dict; the caller
        POSTs it to either the single-order or batched endpoint."""
        body: dict = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": self._V2_SIDE[side],
            "count": self._count_fp(contracts),
            "price": self._price_dollars(price_cents),
            "time_in_force": time_in_force.value,
            "self_trade_prevention_type": self_trade_prevention.value,
            # Auto-cancel during exchange halts — free safety belt.
            "cancel_order_on_pause": True,
        }
        if post_only:
            body["post_only"] = True
        if reduce_only:
            body["reduce_only"] = True
        if expiration_time is not None:
            body["expiration_time"] = int(expiration_time)
        return body

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
        expiration_time: Optional[int] = None,
    ) -> Order:
        # V2 requires client_order_id; generate one if not supplied.
        if not client_order_id:
            client_order_id = f"btc15-auto-{uuid.uuid4().hex[:6]}"
        body = self._v2_order_body(
            ticker=ticker, side=side, contracts=contracts,
            price_cents=price_cents,
            client_order_id=client_order_id,
            time_in_force=time_in_force or TimeInForce.IOC,
            self_trade_prevention=self_trade_prevention or SelfTradePrevention.CANCEL_INCOMING,
            post_only=post_only,
            reduce_only=False,
            expiration_time=expiration_time,
        )
        log.info(f"[V2 PLACE REQ] {body}")
        data = await self.post(self._V2_BASE, body)
        log.info(f"[V2 PLACE RES] {data}")
        return self._parse_order(data.get("order") or data)

    async def amend_order(
        self,
        order_id: str,
        ticker: str,
        side: Side,
        price_cents: Optional[int] = None,
        count: Optional[int] = None,
        action: str = "buy",  # kept for caller compat; V2 doesn't use it
    ) -> Order:
        """Amend a resting order's price or count in-place (no cancel+replace).

        V2 path: POST /portfolio/events/orders/{id}/amend with the V2
        single-book body shape (side: bid/ask, single price field).
        The `action` parameter is kept in the signature for caller compatibility
        but is ignored — V2 infers buy/sell from position state.
        """
        body: dict = {
            "ticker": ticker,
            "side": self._V2_SIDE[side],
        }
        if price_cents is not None:
            body["price"] = self._price_dollars(price_cents)
        if count is not None:
            body["count"] = self._count_fp(count)
        data = await self.post(f"{self._V2_BASE}/{order_id}/amend", body)
        return self._parse_order(data.get("order") or data)

    async def batch_place_orders(self, orders: list[dict]) -> list[Order]:
        """Place multiple orders atomically via V2 batched endpoint.

        Accepts a list of dicts in the same shape produced by callers today
        (with `action`/`side: yes/no`/`yes_price_dollars`/etc.) and translates
        each to V2 shape before posting. Keeps the caller surface unchanged.
        """
        v2_orders = []
        for o in orders:
            # Translate legacy-shaped dicts (used by the engine's arb batch) to V2.
            side_str = (o.get("side") or "yes").lower()
            side_enum = Side.YES if side_str == "yes" else Side.NO
            # Pull price from whichever legacy field is set
            price_str = o.get("yes_price_dollars") or o.get("no_price_dollars")
            if price_str is None and "price" in o:
                price_str = o["price"]
            # Caller may pass price as a string ("0.65") or integer cents (65)
            if isinstance(price_str, (int, float)):
                price_cents = int(price_str) if price_str > 1 else int(round(price_str * 100))
            else:
                price_cents = int(round(float(price_str) * 100))
            cid = o.get("client_order_id") or f"btc15-batch-{uuid.uuid4().hex[:6]}"
            tif = o.get("time_in_force", TimeInForce.IOC.value)
            tif_enum = TimeInForce(tif) if isinstance(tif, str) else tif
            stp_val = o.get("self_trade_prevention_type", SelfTradePrevention.CANCEL_INCOMING.value)
            stp_enum = SelfTradePrevention(stp_val) if isinstance(stp_val, str) else stp_val
            v2_orders.append(self._v2_order_body(
                ticker=o["ticker"], side=side_enum,
                contracts=int(o.get("count", 0)),
                price_cents=price_cents,
                client_order_id=cid,
                time_in_force=tif_enum,
                self_trade_prevention=stp_enum,
                post_only=bool(o.get("post_only", False)),
                reduce_only=bool(o.get("reduce_only", False)),
                expiration_time=o.get("expiration_time"),
            ))
        data = await self.post(f"{self._V2_BASE}/batched", {"orders": v2_orders})
        return [self._parse_order(o) for o in data.get("orders", [])]

    async def sell_position(
        self,
        ticker: str,
        side: Side,
        contracts: int,
        price_cents: int,
    ) -> Order:
        """Close an existing position at a specific limit price.

        Uses IOC (Immediate or Cancel) so the order either fills on the spot
        or is cancelled — it never rests on the book. Resting GTC sell orders
        were the root cause of orphan positions: the bot removed the position
        from internal tracking assuming the sell filled, but the GTC limit
        sat unfilled on Kalshi, producing ghost positions that the orphan
        detector would later re-adopt with wrong contract counts.

        NOTE: This is a price-sensitive sell — the order only crosses bids
        ≥ `price_cents`. For stop-losses and emergency exits use
        `sell_position_sweep()` instead, which takes whatever the book gives.
        """
        body = self._v2_order_body(
            ticker=ticker, side=side, contracts=contracts,
            price_cents=price_cents,
            client_order_id=f"btc15-sell-{uuid.uuid4().hex[:6]}",
            time_in_force=TimeInForce.IOC,
            self_trade_prevention=SelfTradePrevention.CANCEL_INCOMING,
            reduce_only=True,  # exchange guarantee against accidental short
        )
        data = await self.post(self._V2_BASE, body)
        return self._parse_order(data.get("order") or data)

    async def sell_position_sweep(
        self,
        ticker: str,
        side: Side,
        contracts: int,
    ) -> Order:
        """Close a position by sweeping the book — take any bid ≥ 1¢.

        Kalshi limit orders fill at the RESTING maker's price, not the
        taker's limit, so submitting an IOC sell with `yes_price=1` (or
        `no_price=1`) tells Kalshi: "take whatever bids exist, highest
        first, until this order is filled." This is the correct primitive
        for stop-losses, emergency stops, reversals, and profit-takes:
        we care about getting out, not about the exact execution price.

        The old `sell_position(price_cents=6)` path was polite but brittle:
        if the top-of-book bid was a phantom at 6¢ and the real liquidity
        sat at 3¢ and 2¢, the IOC would reject those fills because they
        were below our stated limit. Positions then rode to 0¢ settlement
        even though the book had buyers the whole time.
        """
        body = self._v2_order_body(
            ticker=ticker, side=side, contracts=contracts,
            price_cents=1,  # floor at $0.01; Kalshi matches at best available bid
            client_order_id=f"btc15-sweep-{uuid.uuid4().hex[:6]}",
            time_in_force=TimeInForce.IOC,
            self_trade_prevention=SelfTradePrevention.CANCEL_INCOMING,
            reduce_only=True,
        )
        data = await self.post(self._V2_BASE, body)
        return self._parse_order(data.get("order") or data)

    async def cancel_order(self, order_id: str) -> dict:
        return await self.delete(f"{self._V2_BASE}/{order_id}")

    async def cancel_all_orders(self, ticker: Optional[str] = None) -> int:
        """Cancel every resting order via V2 batched endpoint.

        Uses `DELETE /portfolio/events/orders/batched` so a shutdown with
        many open orders does not chew through the rate limit one cancel
        at a time. Falls back to per-order V2 DELETE if the batched endpoint
        rejects the request.
        """
        resting = [
            o for o in await self.get_orders(ticker=ticker)
            if o.status == OrderStatus.RESTING
        ]
        if not resting:
            return 0

        cancelled = 0
        for i in range(0, len(resting), 20):
            chunk = resting[i:i + 20]
            try:
                await self._request(
                    "DELETE",
                    f"{self._V2_BASE}/batched",
                    json={"orders": [{"order_id": o.order_id} for o in chunk]},
                )
                cancelled += len(chunk)
            except Exception as e:
                log.warning(
                    f"Batched cancel failed for {len(chunk)} orders ({e}); "
                    "falling back to per-order DELETE"
                )
                for order in chunk:
                    try:
                        await self.cancel_order(order.order_id)
                        cancelled += 1
                    except Exception as inner:
                        log.warning(f"Failed to cancel {order.order_id}: {inner}")
        return cancelled

    def _parse_order(self, o: dict) -> Order:
        """Parse an order response across V1/V2/legacy shapes.

        V2 (/portfolio/events/orders):
          order_id, client_order_id, fill_count (str), remaining_count (str),
          average_fill_price (str dollars), average_fee_paid (str dollars),
          ts_ms. Side and ticker are NOT in the V2 response — those are
          inferred from the request context by the caller if needed.
        V1 fixed-point legacy:
          fill_count_fp, remaining_count_fp, initial_count_fp,
          yes_price_dollars, no_price_dollars.
        Older legacy:
          filled_count, remaining_count, count, yes_price/no_price (cents).
        Read all three shapes so any transition state is tolerated.
        """

        def _to_int(val, default: int = 0) -> int:
            try:
                return int(round(float(val)))
            except (TypeError, ValueError):
                return default

        def _first(*vals):
            """Return the first non-None value."""
            for v in vals:
                if v is not None:
                    return v
            return None

        def _price_cents(dollar_val, cent_val) -> int:
            """Prefer the *_dollars field; fall back to integer cents."""
            if dollar_val is not None:
                try:
                    return int(round(float(dollar_val) * 100))
                except (TypeError, ValueError):
                    pass
            try:
                return int(round(float(cent_val or 0)))
            except (TypeError, ValueError):
                return 0

        count = _to_int(_first(o.get("count"), o.get("initial_count_fp")))
        # V2 uses "fill_count" (plain); fall back to legacy names.
        filled = _to_int(_first(
            o.get("fill_count_fp"), o.get("fill_count"), o.get("filled_count")
        ))
        remaining = _to_int(_first(
            o.get("remaining_count_fp"), o.get("remaining_count")
        ))

        # V2 doesn't echo "status" in the lightweight create response.
        # Infer: if any contracts filled, executed; otherwise resting.
        # The caller can override interpretation based on time_in_force.
        raw_status = o.get("status")
        if raw_status:
            status = OrderStatus(raw_status)
        elif filled > 0 and remaining == 0:
            status = OrderStatus.EXECUTED
        elif filled > 0:
            status = OrderStatus.EXECUTED  # partial fill — treat as executed
        else:
            status = OrderStatus.RESTING

        # V2 returns a single average_fill_price; map it onto the right
        # side-specific field based on the order's side (if present).
        avg_fill = o.get("average_fill_price")
        side_str = o.get("side", "yes")
        # V2's side is "bid"/"ask"; map back to yes/no for internal use.
        if side_str == "bid":
            side_str = "yes"
        elif side_str == "ask":
            side_str = "no"
        try:
            side_enum = Side(side_str)
        except ValueError:
            side_enum = Side.YES

        yes_price = _price_cents(
            o.get("yes_price_dollars") or (avg_fill if side_enum == Side.YES else None),
            o.get("yes_price"),
        )
        no_price = _price_cents(
            o.get("no_price_dollars") or (avg_fill if side_enum == Side.NO else None),
            o.get("no_price"),
        )

        return Order(
            order_id=o.get("order_id", ""),
            ticker=o.get("ticker", ""),
            side=side_enum,
            order_type=OrderType(o.get("type", "limit")),
            count=count if count > 0 else (filled + remaining),
            yes_price=yes_price,
            no_price=no_price,
            status=status,
            filled_count=filled,
            remaining_count=remaining,
        )
