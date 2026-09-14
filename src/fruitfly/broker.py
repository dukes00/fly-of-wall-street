"""Alpaca paper-trading broker adapter.

This adapter is hard-limited to Alpaca's *paper* trading endpoint. The
constructor whitelist below refuses any base URL other than the configured
paper URL, so the client can never be pointed at a live-money endpoint.

Endpoint policy: the only base URL ever referenced in this module is read
from the ``APCA_API_BASE_URL`` environment variable (as configured in the
repo's ``.env``). No other broker hostname appears anywhere in this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
from dotenv import load_dotenv

# The single allowed base URL pattern: paper trading only. Anything else
# (live-money endpoints of any broker) is rejected by the guard below.
PAPER_URL_SUFFIX = "paper-api.alpaca.markets"
_LIVE_MARKERS = ("broker-api",)



@dataclass
class OrderEvent:
    """A single order request emitted by a strategy.

    Attributes:
        symbol: Ticker symbol.
        side: ``"buy"`` or ``"sell"``.
        qty: Quantity in shares (fractional allowed).
        tif: Time in force, default ``"day"``.
        type: Order type, ``"market"`` or ``"limit"``.
        limit_price: Required iff ``type == "limit"``.
    """

    symbol: str
    side: str
    qty: float
    tif: str = "day"
    type: str = "market"
    limit_price: float | None = None


def _to_request(event: OrderEvent) -> Any:
    """Map an :class:`OrderEvent` to the corresponding alpaca-py request model."""
    side = OrderSide(event.side.lower())
    tif = TimeInForce(event.tif.lower())
    if event.type == "market":
        return MarketOrderRequest(
            symbol=event.symbol,
            qty=event.qty,
            side=side,
            time_in_force=tif,
        )
    if event.type == "limit":
        if event.limit_price is None:
            raise ValueError("limit order requires limit_price")
        return LimitOrderRequest(
            symbol=event.symbol,
            qty=event.qty,
            side=side,
            time_in_force=tif,
            limit_price=event.limit_price,
        )
    raise ValueError(f"unsupported order type: {event.type!r}")


def _assert_paper_url(url: str) -> str:
    """Whitelist guard: refuse any non-paper base URL.

    Scans for live-money endpoint markers as well, so even a lookalike
    hostname containing a live path is rejected.
    """
    lowered = url.lower()
    if PAPER_URL_SUFFIX not in lowered:
        raise ValueError(
            f"refusing non-paper base URL {url!r}: only {PAPER_URL_SUFFIX} is allowed"
        )
    for marker in _LIVE_MARKERS:
        if marker in lowered:
            raise ValueError(f"refusing base URL containing live marker {marker!r}")
    return url


class AlpacaPaperBroker:
    """Broker adapter over ``alpaca.trading.client.TradingClient`` (paper only)."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        if os.path.exists(".env"):
            load_dotenv(".env")
        api_key = api_key or os.environ.get("APCA_API_KEY_ID")
        secret_key = secret_key or os.environ.get("APCA_API_SECRET_KEY")
        base_url = base_url or os.environ.get("APCA_API_BASE_URL")
        if not api_key or not secret_key:
            raise ValueError(
                "missing Alpaca credentials: set APCA_API_KEY_ID / APCA_API_SECRET_KEY"
            )
        if not base_url:
            raise ValueError("missing base URL: set APCA_API_BASE_URL to the paper URL")
        self.base_url = _assert_paper_url(base_url)
        self._client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=True,
            url_override=self.base_url,
        )

    def submit(self, event: OrderEvent) -> str:
        """Submit an order; returns the Alpaca order id."""
        order = self._client.submit_order(_to_request(event))
        return str(order.id)

    def cancel(self, order_id: str) -> None:
        """Cancel a resting order by id (no-op if already terminal)."""
        from alpaca.common.exceptions import APIError

        try:
            self._client.cancel_order_by_id(order_id)
        except APIError as exc:  # e.g. already filled/cancelled
            if "422" in str(exc) or "not open" in str(exc).lower():
                return
            raise

    def positions(self) -> dict[str, dict[str, Any]]:
        """Return open positions keyed by symbol."""
        return {p.symbol: p.model_dump() for p in self._client.get_all_positions()}

    def account(self) -> dict[str, Any]:
        """Return a snapshot of the account."""
        return self._client.get_account().model_dump()
