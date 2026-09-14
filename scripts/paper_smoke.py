#!/usr/bin/env python
"""Real-network smoke test against the Alpaca PAPER account (never live).

Lifecycle verified here, in order:
  1. print an account snapshot,
  2. submit one small odd-lot limit order (1 share AAPL at $1.00) that rests
     far below the market and cannot fill,
  3. confirm the order is accepted (open at Alpaca) via get_orders,
  4. cancel it and confirm the status is cancelled.

NOTE: the market is closed until the next session open (2026-09-15 13:30 UTC),
so FILL behaviour cannot be verified tonight; this smoke validates the order
lifecycle up to and including cancellation. Fill verification happens at the
next open.

Run from the repo root:  uv run python scripts/paper_smoke.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from fruitfly.broker import AlpacaPaperBroker, OrderEvent  # noqa: E402


def main() -> None:
    broker = AlpacaPaperBroker()

    acct = broker.account()
    print("== Account snapshot ==")
    for field in (
        "account_number", "status", "currency", "cash", "buying_power",
        "equity", "paper_account",
    ):
        if field in acct:
            print(f"  {field}: {acct[field]}")

    print("\n== Submitting resting limit order (1 AAPL @ $1.00, day) ==")
    order_id = broker.submit(
        OrderEvent(symbol="AAPL", side="buy", qty=1, type="limit", limit_price=1.00)
    )
    print(f"  order id: {order_id}")

    orders = broker._client.get_orders()
    ours = [o for o in orders if str(o.id) == order_id]
    assert ours, "submitted order not found via get_orders"
    status_accepted = ours[0].status.value in ("new", "accepted", "pending_new", "held")
    print(f"  status after submit: {ours[0].status.value} (accepted: {status_accepted})")
    assert status_accepted, f"order not accepted: {ours[0].status.value}"

    print("\n== Cancelling ==")
    broker.cancel(order_id)
    fetched = broker._client.get_order_by_id(order_id)
    final_status = fetched.status.value
    print(f"  status after cancel: {final_status}")
    assert final_status in ("canceled", "cancelled", "pending_cancel"), (
        f"unexpected final status: {final_status}"
    )

    print("\n== Smoke PASSED ==")
    print("  Order accepted, then cancelled, on the paper account.")
    print(
        "  NOTE: market is closed until the next session open "
        "(2026-09-15 13:30 UTC); fills cannot occur now. "
        "Fill verification happens at next open."
    )


if __name__ == "__main__":
    main()
