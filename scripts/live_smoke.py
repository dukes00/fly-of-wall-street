#!/usr/bin/env python
"""T15 live-paper smoke: submit ONE paper order at the next market open.

PENDING-MARKET-OPEN (see note): the market is closed until 2026-09-15
13:30 UTC, so this script cannot complete tonight. Run it any time — it
computes the next NYSE regular-session open from the shared calendar helpers
(:func:`fruitfly.adult.next_session_open`), sleeps until then (pass
``--no-wait`` to fire immediately when already in a session), submits ONE
small market order (1 share SPY) through the paper-only
:class:`fruitfly.broker.AlpacaPaperBroker`, polls the fill, and cancels if
the order is still resting. It exists to prove the live seam end-to-end:
broker credentials -> order accepted -> fill observed. It does NOT run the
full adult fly — that is ``python -m fruitfly adult --mode live-paper``.

Run from the repo root:  uv run python scripts/live_smoke.py
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd  # noqa: E402

from fruitfly.adult import next_session_open  # noqa: E402
from fruitfly.broker import AlpacaPaperBroker, OrderEvent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="do not sleep until the next open; require a session now",
    )
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--qty", type=int, default=1)
    parser.add_argument("--poll-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    args = parser.parse_args()

    broker = AlpacaPaperBroker()
    acct = broker.account()
    print("== Account snapshot ==")
    for key in ("account_number", "status", "cash", "buying_power", "equity"):
        print(f"  {key}: {acct.get(key)}")

    now = pd.Timestamp.now(tz="UTC")
    open_ts = next_session_open(now)
    in_session = now.time() >= open_ts.time() and now.time() <= pd.Timestamp(
        open_ts.date(), tz="UTC"
    ) + pd.Timedelta(hours=20) - pd.Timedelta(minutes=1)
    if now < open_ts or not in_session:
        if args.no_wait:
            raise SystemExit(
                f"--no-wait given but the market is closed; next open is {open_ts} (UTC)"
            )
        print(f"\nMarket closed; waiting until the next session open: {open_ts} (UTC)")
        time.sleep(max(0.0, (open_ts - now).total_seconds()) + 2.0)
    else:
        print(f"\nSession is open now (opened {open_ts} UTC)")

    print(f"== Submitting ONE paper market order: {args.qty} {args.symbol} ==")
    order_id = broker.submit(
        OrderEvent(symbol=args.symbol, side="buy", qty=args.qty, tif="day", type="market")
    )
    print(f"  order id: {order_id}")

    deadline = time.monotonic() + args.timeout_s
    status, filled_at = None, None
    while time.monotonic() < deadline:
        order = broker._client.get_order_by_id(order_id)
        status = order.status.value
        filled_at = order.filled_avg_price
        if status in ("filled", "canceled", "cancelled", "expired", "rejected"):
            break
        time.sleep(args.poll_s)
    else:
        print("  poll timed out; cancelling the resting order")
        broker.cancel(order_id)
        status = broker._client.get_order_by_id(order_id).status.value

    if status == "filled":
        print(f"  FILLED at {filled_at}")
    else:
        print(f"  final status: {status} (not filled — see notes below)")
    print("== Live smoke DONE ==")
    print(
        "NOTE: pending-market-open verification — the first real invocation "
        "completes at the next session open (2026-09-15 13:30 UTC or later)."
    )


if __name__ == "__main__":
    main()
