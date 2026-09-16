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

import pandas as pd
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
    """Deterministic stand-in for the benchmark (no cache reads).

    The seam is ``benchmark_daily_returns`` (^GSPC daily close when
    covered, else the SPY proxy from the fetched cache).
    """
    monkeypatch.setattr(
        eval_brains, "benchmark_daily_returns",
        lambda days: {
            "source": "spx",
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
    avail = [f"2026-{m:02d}-{d:02d}" for m in (7, 8) for d in range(1, 24)]
    assert len(avail) == 46  # > RECENT_POOL = 40
    got = eval_brains.select_days(avail, 3)
    assert got[0] == avail[-40]  # window starts at the 40th-most-recent day
    assert got[-1] == avail[-1]


def test_select_days_caps_at_the_recent_pool():
    avail = [f"2026-{m:02d}-{d:02d}" for m in (7, 8) for d in range(1, 24)]
    with pytest.raises(ValueError, match="exceeds the 40"):
        eval_brains.select_days(avail, 41)


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




def test_select_days_within_bounded_pool():
    """A start/end-bounded available list selects only inside the bound."""
    avail = [f"2026-08-{d:02d}" for d in range(3, 15)]  # 12 days
    bounded = [d for d in avail if "2026-08-05" <= d <= "2026-08-10"]
    got = eval_brains.select_days(bounded, 2)
    assert got == ["2026-08-05", "2026-08-10"]


def _install_fake_run(monkeypatch, tmp_path, calls, meta=None):
    """Wire every run seam: bars days, chassis, weights, run_backtest.

    ``meta`` overrides the fake artifact's meta (default: a legacy
    pre-A9 artifact with no recorded basket or knobs).
    """
    from types import SimpleNamespace

    import fruitfly.connectome
    import fruitfly.data
    import fruitfly.loop
    import fruitfly.train

    def fake_run(config):
        calls.append({
            "day": config.start,
            "data_basket": list(fruitfly.data.BASKET),
            "loop_basket": list(fruitfly.loop.BASKET),
        })
        config.out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"equity": [100_000.0, 101_000.0]}).to_csv(
            config.out_dir / "equity.csv", index=False)
        return SimpleNamespace(run_dir=config.out_dir, final_equity=101_000.0,
                               n_orders=3, n_deaths=0)

    monkeypatch.setattr(fruitfly.loop, "run_backtest", fake_run)
    if meta is None:
        meta = {"end": "2026-01-01", "seed": "7"}
    monkeypatch.setattr(
        fruitfly.train, "load_larval_weights",
        lambda artifact, brain: SimpleNamespace(weights=None, meta=meta))
    monkeypatch.setattr(fruitfly.connectome, "load_stripped_chassis",
                        lambda: object())
    cache_days = [f"2026-08-{d:02d}" for d in range(3, 15)]
    monkeypatch.setattr(eval_brains, "cache_trading_days",
                        lambda: list(cache_days))
    return cache_days


def test_main_basket_and_period_reach_seam_and_receipt(tmp_path, monkeypatch,
                                                       fake_spx):
    from fruitfly import data as fdata
    from fruitfly import loop as floop
    calls: list[dict] = []
    # A custom --basket is a non-default config: the fake artifact's meta
    # must record the same basket (A9 fail-closed rule for legacy artifacts).
    _install_fake_run(
        monkeypatch, tmp_path, calls,
        meta={"end": "2026-01-01", "seed": "7",
              "basket": json.dumps(["AAA", "BBB"])})
    orig_data, orig_loop = list(fdata.BASKET), list(floop.BASKET)
    results_dir, report = tmp_path / "res", tmp_path / "r.md"
    (tmp_path / "x.npz").write_bytes(b"fake")
    eval_brains.main([
        "--chassis", "stripped", "--artifact", str(tmp_path / "x.npz"),
        "--days", "2", "--std-beta", "0.1", "--std-tau-rec-ms", "500",
        "--start", "2026-08-05", "--end", "2026-08-10",
        "--basket", "AAA,BBB",
        "--results-dir", str(results_dir), "--report", str(report),
    ])
    receipt = json.loads((results_dir / "results_stripped.json").read_text())
    assert receipt["period"] == {"start": "2026-08-05", "end": "2026-08-10"}
    assert receipt["basket"] == ["AAA", "BBB"]
    assert all(c["data_basket"] == ["AAA", "BBB"] for c in calls)
    assert all(c["loop_basket"] == ["AAA", "BBB"] for c in calls)
    assert all("2026-08-05" <= c["day"] <= "2026-08-10" for c in calls)
    assert list(fdata.BASKET) == orig_data
    assert list(floop.BASKET) == orig_loop
    text = report.read_text()
    assert "period=2026-08-05..2026-08-10" in text
    assert "basket=AAA,BBB" in text


def test_main_default_receipt_has_no_period_or_basket(tmp_path, monkeypatch,
                                                      fake_spx):
    calls: list[dict] = []
    _install_fake_run(monkeypatch, tmp_path, calls)
    results_dir, report = tmp_path / "res", tmp_path / "r.md"
    (tmp_path / "x.npz").write_bytes(b"fake")
    eval_brains.main([
        "--chassis", "stripped", "--artifact", str(tmp_path / "x.npz"),
        "--days", "2",
        "--results-dir", str(results_dir), "--report", str(report),
    ])
    receipt = json.loads((results_dir / "results_stripped.json").read_text())
    assert "period" not in receipt and "basket" not in receipt
    assert "period=" not in report.read_text()
    assert "basket=" not in report.read_text()


def test_main_rejects_start_after_end():
    with pytest.raises(ValueError, match="must be <= --end"):
        eval_brains.main(["--chassis", "stripped", "--artifact", "x.npz",
                          "--start", "2026-09-02", "--end", "2026-08-28"])


def test_main_rejects_single_symbol_basket():
    with pytest.raises(ValueError, match="--basket"):
        eval_brains.main(["--chassis", "stripped", "--artifact", "x.npz",
                          "--basket", "AAA"])


def test_basket_override_reaches_cache_trading_days(tmp_path, monkeypatch):
    """The day pool uses the overridden basket via fruitfly.data.BASKET."""
    import fruitfly.data
    frames = {}
    for symbol in ("AAA", "BBB"):
        path = tmp_path / f"{symbol}.parquet"
        pd.DataFrame({"timestamp": pd.to_datetime(
            ["2026-08-05", "2026-08-06"], utc=True)}).to_parquet(path)
        frames[symbol] = path
    monkeypatch.setattr(
        fruitfly.data, "cache_path",
        lambda symbol: frames.get(symbol, tmp_path / "missing.parquet"))
    with eval_brains._basket_override(["AAA", "BBB"]):
        assert eval_brains.cache_trading_days() == [
            "2026-08-05", "2026-08-06"]
    with pytest.raises(ValueError, match="no trading days"):
        eval_brains.cache_trading_days()


def test_basket_override_restores_original():
    import fruitfly.data
    import fruitfly.loop
    orig_data, orig_loop = fruitfly.data.BASKET, fruitfly.loop.BASKET
    with eval_brains._basket_override(["AAA", "BBB"]):
        assert fruitfly.data.BASKET == ["AAA", "BBB"]
        assert fruitfly.loop.BASKET == ["AAA", "BBB"]
    assert fruitfly.data.BASKET is orig_data
    assert fruitfly.loop.BASKET is orig_loop
    with pytest.raises(ValueError, match="--days"):
        eval_brains.main(["--chassis", "stripped", "--artifact", "x.npz",
                          "--days", "0"])
