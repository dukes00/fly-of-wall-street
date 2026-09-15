"""Offline tests for scripts/eval_brains.py (brain-shootout harness).

Synthetic bars/receipts only — no real cache data and no backtests. Covers
the metric math (return/drawdown), held-out day selection, and the report
renderer's determinism + PENDING behavior.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_brains", Path(__file__).resolve().parents[1] / "scripts" / "eval_brains.py"
)
assert _SPEC is not None and _SPEC.loader is not None
eval_brains = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_brains)


# ---------------------------------------------------------------------------
# Synthetic receipts
# ---------------------------------------------------------------------------


def _receipt(chassis: str, days: list[str], seed: int = 7) -> dict:
    per_day = [
        {
            "day": day,
            "final_return_pct": 0.25 * (i + 1),
            "max_drawdown_pct": 1.5 + 0.1 * i,
            "trades": 10 + i,
            "deaths": i % 2,
        }
        for i, day in enumerate(days)
    ]
    return {
        "arm": chassis,
        "chassis": chassis,
        "artifact": f"data/fly-{chassis}-weights.npz",
        "artifact_md5": "0" * 32,
        "artifact_meta": {"seed": str(seed), "start": "2026-03-02", "end": "2026-06-30"},
        "seed": seed,
        "std_beta": None,
        "std_tau_rec_ms": None,
        "days": per_day,
        "totals": {
            "mean_return_pct": sum(d["final_return_pct"] for d in per_day) / len(per_day),
            "max_drawdown_pct": max(d["max_drawdown_pct"] for d in per_day),
            "trades": sum(d["trades"] for d in per_day),
            "deaths": sum(d["deaths"] for d in per_day),
        },
    }


DAYS = ["2026-08-28", "2026-08-31", "2026-09-02", "2026-09-04"]


@pytest.fixture()
def fake_spx(monkeypatch):
    """Deterministic stand-in for the real SPX benchmark (no cache reads)."""
    monkeypatch.setattr(
        eval_brains, "spx_daily_returns",
        lambda days: {
            "daily_return_pct": [0.1 * (i + 1) for i in range(len(days))],
            "window_return_pct": 1.234,
            "max_drawdown_pct": 0.567,
        },
    )


# ---------------------------------------------------------------------------
# Held-out day selection
# ---------------------------------------------------------------------------


def test_select_days_full_window():
    assert eval_brains.select_days(DAYS, 4) == DAYS


def test_select_days_takes_most_recent_pool():
    avail = [f"2026-08-{d:02d}" for d in range(3, 15)]  # 12 days
    got = eval_brains.select_days(avail, 3)
    assert got[0] == avail[-10]  # window starts at the 10th-most-recent day
    assert got[-1] == avail[-1]


def test_select_days_caps_at_the_recent_pool():
    avail = [f"2026-08-{d:02d}" for d in range(3, 15)]  # 12 days
    with pytest.raises(ValueError, match="exceeds the 10"):
        eval_brains.select_days(avail, 11)


def test_select_days_evenly_spaced_keeps_endpoints():
    got = eval_brains.select_days(DAYS, 2)
    assert got == [DAYS[0], DAYS[-1]]
    got3 = eval_brains.select_days(DAYS, 3)
    assert got3[0] == DAYS[0] and got3[-1] == DAYS[-1]
    assert got3[1] in DAYS[1:-1]


def test_select_days_rejects_out_of_range():
    with pytest.raises(ValueError, match="must be >= 1"):
        eval_brains.select_days(DAYS, 0)
    with pytest.raises(ValueError, match="exceeds"):
        eval_brains.select_days(DAYS, 5)


# ---------------------------------------------------------------------------
# Per-day metric math
# ---------------------------------------------------------------------------


def _write_equity_csv(path: Path, equities: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "timestamp,equity,cash,n_positions\n"
        + "\n".join(
            f"2026-08-28T{i:02d}:00:00+00:00,{e:.2f},0.00,0"
            for i, e in enumerate(equities)
        )
        + "\n"
    )


def test_day_metrics_return_and_drawdown(tmp_path):
    # +1% final, with a 2% peak-to-trough dip along the way.
    equities = [100_000.0, 102_000.0, 99_960.0, 101_000.0]
    equity_csv = tmp_path / "equity.csv"
    _write_equity_csv(equity_csv, equities)
    m = eval_brains.day_metrics(equity_csv, final_equity=101_000.0,
                                n_orders=7, n_deaths=0, initial_cash=100_000.0)
    assert m["final_return_pct"] == pytest.approx(1.0)
    # (1 - 99960/102000) * 100 = 2.0
    assert m["max_drawdown_pct"] == pytest.approx(2.0)
    assert m["trades"] == 7 and m["deaths"] == 0


def test_day_metrics_flat_day_is_zero(tmp_path):
    equity_csv = tmp_path / "equity.csv"
    _write_equity_csv(equity_csv, [100_000.0, 100_000.0])
    m = eval_brains.day_metrics(equity_csv, final_equity=100_000.0,
                                n_orders=0, n_deaths=0, initial_cash=100_000.0)
    assert m["final_return_pct"] == 0.0
    assert m["max_drawdown_pct"] == 0.0


def test_day_metrics_monotonic_loss(tmp_path):
    equity_csv = tmp_path / "equity.csv"
    _write_equity_csv(equity_csv, [100_000.0, 99_000.0, 98_000.0])
    m = eval_brains.day_metrics(equity_csv, final_equity=98_000.0,
                                n_orders=3, n_deaths=1, initial_cash=100_000.0)
    assert m["final_return_pct"] == pytest.approx(-2.0)
    assert m["max_drawdown_pct"] == pytest.approx(2.0)
    assert m["deaths"] == 1


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def test_render_report_deterministic(tmp_path, fake_spx):
    stripped, whole = _receipt("stripped", DAYS), _receipt("whole", DAYS)
    one = eval_brains.render_report({"stripped": stripped, "whole": whole},
                                    tmp_path / "a.md")
    two = eval_brains.render_report({"stripped": stripped, "whole": whole},
                                    tmp_path / "b.md")
    assert one == two and (tmp_path / "a.md").read_text() == one


def test_render_report_both_arms_fills_every_cell(tmp_path, fake_spx):
    text = eval_brains.render_report(
        {"stripped": _receipt("stripped", DAYS), "whole": _receipt("whole", DAYS)},
        tmp_path / "r.md",
    )
    assert "PENDING" not in text
    for day in DAYS:
        assert f"| {day} |" in text
    assert "| TOTAL |" in text
    # 11 columns: day + 2 SPX + 4 per arm x2
    data_row = next(line for line in text.splitlines() if line.startswith(f"| {DAYS[0]}"))
    assert data_row.count("|") == 12


def test_render_report_pending_whole_arm(tmp_path, fake_spx):
    text = eval_brains.render_report({"stripped": _receipt("stripped", DAYS)},
                                     tmp_path / "r.md")
    # Arms row + 4 PENDING cells per data row + the TOTAL row; the prose
    # mentions PENDING only while the whole-fly arm is actually missing.
    table_pending = sum(
        line.count("PENDING")
        for line in text.splitlines()
        if line.startswith("| ")
    )
    assert table_pending == 4 * (len(DAYS) + 1) + 1
    row = next(line for line in text.splitlines() if line.startswith(f"| {DAYS[0]}"))
    assert row.count("PENDING") == 4


def test_render_report_rejects_missing_stripped(tmp_path):
    with pytest.raises(ValueError, match="stripped"):
        eval_brains.render_report({}, tmp_path / "r.md")


def test_render_report_rejects_day_mismatch(tmp_path, fake_spx):
    other_days = DAYS[:3]
    with pytest.raises(ValueError, match="different days"):
        eval_brains.render_report(
            {"stripped": _receipt("stripped", DAYS),
             "whole": _receipt("whole", other_days)},
            tmp_path / "r.md",
        )


def test_receipt_roundtrip_matches_render(tmp_path, fake_spx):
    """A receipt written as JSON re-renders the identical report."""
    stripped = _receipt("stripped", DAYS)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "results_stripped.json").write_text(
        json.dumps(stripped, indent=2, sort_keys=True) + "\n"
    )
    loaded = eval_brains.load_receipts(results_dir)
    direct = eval_brains.render_report({"stripped": stripped}, tmp_path / "a.md")
    via_json = eval_brains.render_report(loaded, tmp_path / "b.md")
    assert direct == via_json


# ---------------------------------------------------------------------------
# Totals aggregation
# ---------------------------------------------------------------------------


def test_receipt_totals_math():
    rc = _receipt("stripped", DAYS)
    totals = rc["totals"]
    assert totals["trades"] == sum(d["trades"] for d in rc["days"])
    assert totals["deaths"] == sum(d["deaths"] for d in rc["days"])
    assert math.isclose(
        totals["mean_return_pct"],
        sum(d["final_return_pct"] for d in rc["days"]) / len(rc["days"]),
    )
    assert totals["max_drawdown_pct"] == max(
        d["max_drawdown_pct"] for d in rc["days"]
    )


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------


def test_main_rejects_nonpositive_days():
    with pytest.raises(ValueError, match="--days"):
        eval_brains.main(["--chassis", "stripped", "--artifact", "x.npz",
                          "--days", "0"])
