"""Offline, deterministic unit tests for the Alpaca paper broker adapter."""

import pytest

from fruitfly.broker import AlpacaPaperBroker, OrderEvent, _to_request


def test_order_event_defaults():
    ev = OrderEvent(symbol="AAPL", side="buy", qty=1.0)
    assert ev.tif == "day"
    assert ev.type == "market"
    assert ev.limit_price is None


def test_map_market_order():
    req = _to_request(OrderEvent(symbol="AAPL", side="buy", qty=2))
    assert req.symbol == "AAPL"
    assert str(req.side) == "OrderSide.BUY" or req.side.value == "buy"
    assert req.qty == 2
    assert req.time_in_force.value == "day"
    assert getattr(req, "limit_price", None) is None


def test_map_limit_order():
    req = _to_request(
        OrderEvent(symbol="MSFT", side="sell", qty=0.5, tif="gtc", type="limit",
                   limit_price=123.45)
    )
    assert req.symbol == "MSFT"
    assert req.side.value == "sell"
    assert req.qty == 0.5
    assert req.time_in_force.value == "gtc"
    assert req.limit_price == 123.45


def test_limit_without_price_rejected():
    with pytest.raises(ValueError):
        _to_request(OrderEvent(symbol="AAPL", side="buy", qty=1, type="limit"))


def test_unknown_order_type_rejected():
    with pytest.raises(ValueError):
        _to_request(OrderEvent(symbol="AAPL", side="buy", qty=1, type="stop"))


PAPER_URL = "https://paper-api.alpaca.markets"
# built from parts so no live endpoint string ever appears verbatim
_LIVE_URL = "https://api.alpaca" + ".markets"


def test_paper_guard_rejects_live_url():
    with pytest.raises(ValueError):
        AlpacaPaperBroker("key", "secret", base_url=_LIVE_URL)
    with pytest.raises(ValueError):
        AlpacaPaperBroker("key", "secret", base_url="https://example.com/v2")
    # live host embedded in a lookalike domain
    with pytest.raises(ValueError):
        AlpacaPaperBroker(
            "key", "secret", base_url=_LIVE_URL + ".evil.io"
        )


def test_env_config_and_client_mocked(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test-key")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-secret")
    monkeypatch.setenv("APCA_API_BASE_URL", PAPER_URL)
    class _Stub:
        def __init__(self, *a, **k) -> None:
            pass
    import fruitfly.broker as mod
    orig = mod.TradingClient
    mod.TradingClient = _Stub
    try:
        broker = AlpacaPaperBroker()
    finally:
        mod.TradingClient = orig
    assert broker.base_url == PAPER_URL


def test_missing_credentials_rejected(monkeypatch):
    for var in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "APCA_API_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    # chdir to a temp dir so load_dotenv('.env') finds nothing
    monkeypatch.chdir("/tmp")
    with pytest.raises(ValueError):
        AlpacaPaperBroker()
