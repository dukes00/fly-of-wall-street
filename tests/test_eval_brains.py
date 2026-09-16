"""Offline tests for the TRAINING2 eval-protocol additions in
``scripts/eval_brains.py``.

Synthetic receipts/meta only — no real cache data and no backtests. Covers
the A9 artifact-meta assertion (mismatch -> MetaMismatch, legacy tolerance,
never-seen guard), the RECENT_POOL contract, and the per-cell report
statistics (paired-t, up/down-capture).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "eval_brains", Path(__file__).resolve().parents[1] / "scripts" / "eval_brains.py"
)
assert _SPEC is not None and _SPEC.loader is not None
eval_brains = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(eval_brains)


# ---------------------------------------------------------------------------
# Fixtures: BacktestConfig + artifact meta builders (synthetic)
# ---------------------------------------------------------------------------


def _config(**overrides):
    from fruitfly.loop import BacktestConfig

    defaults = dict(seed=7, start="2026-08-18", end="2026-08-19",
                    chassis="stripped")
    return BacktestConfig(**{**defaults, **overrides})


def _meta_for(config, **extra):
    """Artifact meta whose A9 echo matches ``config`` exactly."""
    from fruitfly.train import META_KNOBS

    meta = {
        name: str(getattr(config, name))
        for name in META_KNOBS if hasattr(config, name)
    }
    meta.update(extra)
    return meta


DAYS = ["2026-08-28", "2026-08-31", "2026-09-02", "2026-09-04"]


def _receipt(arm: str, days: list[str], returns: list[float] | None = None,
             seed: int = 7) -> dict:
    per_day = [
        {
            "day": day,
            "final_return_pct": returns[i] if returns else 0.25 * (i + 1),
            "max_drawdown_pct": 1.5 + 0.1 * i,
            "trades": 10 + i,
            "deaths": i % 2,
        }
        for i, day in enumerate(days)
    ]
    return {
        "arm": arm,
        "chassis": "stripped",
        "artifact": f"data/fly-{arm}-weights.npz",
        "artifact_md5": "0" * 32,
        "artifact_meta": {"seed": str(seed), "start": "2026-03-02",
                          "end": "2026-06-30"},
        "seed": seed,
        "std_beta": None,
        "std_tau_rec_ms": None,
        "days": per_day,
        "totals": {
            "mean_return_pct": sum(d["final_return_pct"] for d in per_day)
            / len(per_day),
            "max_drawdown_pct": max(d["max_drawdown_pct"] for d in per_day),
            "trades": sum(d["trades"] for d in per_day),
            "deaths": sum(d["deaths"] for d in per_day),
        },
    }


# ---------------------------------------------------------------------------
# A9: artifact meta <-> eval runtime config
# ---------------------------------------------------------------------------


def test_meta_match_passes():
    config = _config()
    meta = _meta_for(config, basket=json.dumps(["AAA", "BBB"]))
    eval_brains.assert_meta_matches(meta, config, ["AAA", "BBB"], artifact="x.npz")


def test_meta_mismatch_lists_diff_and_is_a_value_error():
    config = _config()
    meta = _meta_for(config, r_scale="0.005")
    bad = _config(r_scale=0.01)
    with pytest.raises(eval_brains.MetaMismatch) as excinfo:
        eval_brains.assert_meta_matches(meta, bad, ["AAA", "BBB"],
                                        artifact="x.npz")
    msg = str(excinfo.value)
    assert "r_scale" in msg and "0.005" in msg and "0.01" in msg
    assert "x.npz" in msg
    # The module's __main__ guard converts every ValueError into exit 2.
    assert isinstance(excinfo.value, ValueError)


def test_basket_mismatch_fails():
    config = _config()
    meta = _meta_for(config, basket=json.dumps(["AAA", "BBB"]))
    with pytest.raises(eval_brains.MetaMismatch, match="basket"):
        eval_brains.assert_meta_matches(meta, config, ["XXX", "YYY"])


def test_malformed_meta_basket_fails_closed():
    config = _config()
    with pytest.raises(eval_brains.MetaMismatch, match="malformed"):
        eval_brains.assert_meta_matches(_meta_for(config, basket="not json"),
                                        config, ["AAA", "BBB"])


def test_legacy_artifact_default_config_warns_and_continues():
    import fruitfly.data

    config = _config()
    with pytest.warns(UserWarning, match="legacy artifact"):
        eval_brains.assert_meta_matches(
            {}, config, list(fruitfly.data.BASKET), artifact="legacy.npz")


def test_legacy_artifact_non_default_config_fails_closed():
    from dataclasses import fields

    import fruitfly.data
    from fruitfly.loop import BacktestConfig

    present = {f.name for f in fields(BacktestConfig)}
    for knob, value in (("id_scale", 0.0), ("entry_credit", "forecast"),
                        ("horizon_bars", 15)):
        if knob not in present:
            continue  # knob not on the installed BacktestConfig yet
        probe = _config(**{knob: value})
        with pytest.raises(eval_brains.MetaMismatch, match="no recorded"):
            eval_brains.assert_meta_matches(
                {}, probe, list(fruitfly.data.BASKET), artifact="legacy.npz")
        return
    pytest.skip("BacktestConfig does not carry A9 knobs yet")


def test_legacy_artifact_custom_basket_fails_closed():
    config = _config()
    with pytest.raises(eval_brains.MetaMismatch, match="no recorded basket"):
        eval_brains.assert_meta_matches({}, config, ["XXX", "YYY"])


# ---------------------------------------------------------------------------
# Never-seen basket guard
# ---------------------------------------------------------------------------


def test_load_never_seen_parses_comments_and_blanks(tmp_path):
    path = tmp_path / "eval20-neverseen.txt"
    path.write_text("# eval20 — never train on these\n\nZZZ  # mid-cap\nWWW\n")
    assert eval_brains.load_never_seen(path) == frozenset({"ZZZ", "WWW"})


def test_load_never_seen_missing_file_is_inert(tmp_path):
    assert eval_brains.load_never_seen(tmp_path / "absent.txt") == frozenset()


def test_guard_fires_when_artifact_was_trained_on_never_seen(tmp_path):
    path = tmp_path / "eval20-neverseen.txt"
    path.write_text("ZZZ\nWWW\n")
    with pytest.raises(eval_brains.NeverSeenGuardError, match="ZZZ"):
        eval_brains.assert_never_seen(["AAA", "ZZZ"], ["WWW", "ZZZ", "QQQ"],
                                      path=path)


def test_guard_allows_never_seen_basket_on_untrained_artifact(tmp_path):
    path = tmp_path / "eval20-neverseen.txt"
    path.write_text("ZZZ\nWWW\n")
    eval_brains.assert_never_seen(["AAA", "BBB"], ["ZZZ", "WWW"], path=path)


def test_guard_ignores_disjoint_eval_basket(tmp_path):
    path = tmp_path / "eval20-neverseen.txt"
    path.write_text("ZZZ\n")
    eval_brains.assert_never_seen(["ZZZ"], ["AAA", "BBB"], path=path)


# ---------------------------------------------------------------------------
# RECENT_POOL + per-cell statistics
# ---------------------------------------------------------------------------


def test_recent_pool_is_40():
    assert eval_brains.RECENT_POOL == 40


def _patch_benchmark(monkeypatch):
    monkeypatch.setattr(
        eval_brains, "benchmark_daily_returns",
        lambda days: {
            "source": "spy",
            "daily_return_pct": [1.0, -2.0, 3.0, -1.0][:len(days)],
            "window_return_pct": 0.5,
            "max_drawdown_pct": 3.0,
        },
    )


def test_paired_t_math():
    # config-minus-incumbent diffs [1.0, 0.5, 0.0, 1.0]:
    # mean 0.625, sd sqrt(0.6875/3), t ~ 2.61.
    t = eval_brains.paired_t([2.0, -0.5, 3.0, -1.0], [1.0, -1.0, 3.0, -2.0])
    assert t == pytest.approx(2.61, abs=0.01)

def test_paired_t_degenerate_series_is_none():
    assert eval_brains.paired_t([1.0], [2.0]) is None
    assert eval_brains.paired_t([1.0, 2.0], [1.0, 2.0]) is None  # zero variance
    assert eval_brains.paired_t([1.0, 2.0], [1.0]) is None  # length mismatch


def test_capture_ratios_math():
    cap = eval_brains.capture_ratios(
        [2.0, -0.5, 3.0, -1.0], [1.0, -2.0, 3.0, -1.0])
    assert cap["up_capture"] == pytest.approx((2.0 + 3.0) / (1.0 + 3.0))
    assert cap["down_capture"] == pytest.approx((-0.5 - 1.0) / (-2.0 - 1.0))


def test_capture_ratios_none_without_benchmark_moves():
    cap = eval_brains.capture_ratios([1.0, 2.0], [1.0, 2.0])  # no down days
    assert cap["up_capture"] == pytest.approx(1.0)
    assert cap["down_capture"] is None
    assert eval_brains.capture_ratios([1.0], []) == {
        "up_capture": None, "down_capture": None,
    }


def test_render_report_per_cell_statistics(tmp_path, monkeypatch):
    _patch_benchmark(monkeypatch)
    stripped = _receipt("stripped", DAYS, [1.0, -1.0, 3.0, -2.0])
    cfg = _receipt("cfg-a", DAYS, [2.0, -0.5, 3.0, -1.0])
    text = eval_brains.render_report(
        {"stripped": stripped, "cfg-a": cfg}, tmp_path / "r.md")
    assert "## Per-cell statistics" in text
    assert "| arm | shared days | paired-t | up-capture | down-capture |" in text
    # diffs [1.0, 0.5, 0.0, 1.0] -> t = 2.61; capture 1.25 / 0.50.
    assert "| cfg-a | 4 | 2.61 | 1.25 | 0.50 |" in text
    # Incumbent baseline row has no paired-t.
    assert text.count("| stripped | 4 | — |") == 1
    # The arms table echoes the meta knobs so configs are labeled.
    stripped["meta_knobs"] = {"entry_credit": "forecast", "id_scale": "0.0"}
    stripped["meta_basket"] = ["AAA", "BBB"]
    labeled = eval_brains.render_report(
        {"stripped": stripped, "cfg-a": cfg}, tmp_path / "r2.md")
    assert "cfg entry_credit=forecast id_scale=0.0" in labeled
    assert "trained_basket=AAA,BBB" in labeled


def test_render_report_deterministic_with_per_cell(tmp_path, monkeypatch):
    _patch_benchmark(monkeypatch)
    receipts = {"stripped": _receipt("stripped", DAYS),
                "cfg-a": _receipt("cfg-a", DAYS)}
    one = eval_brains.render_report(receipts, tmp_path / "a.md")
    two = eval_brains.render_report(receipts, tmp_path / "b.md")
    assert one == two and (tmp_path / "a.md").read_text() == one


def test_load_receipts_keys_by_arm(tmp_path):
    results = tmp_path / "res"
    results.mkdir()
    for arm in ("stripped", "cfg-a"):
        (results / f"results_{arm}.json").write_text(
            json.dumps(_receipt(arm, DAYS), indent=2, sort_keys=True) + "\n")
    loaded = eval_brains.load_receipts(results)
    assert sorted(loaded) == ["cfg-a", "stripped"]


# ---------------------------------------------------------------------------
# End-to-end path: evaluate_arm enforces the contract (fake run seams)
# ---------------------------------------------------------------------------


def _install_fake_run(monkeypatch, tmp_path, meta):
    import fruitfly.connectome
    import fruitfly.loop
    import fruitfly.train

    def fake_run(config):
        config.out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"equity": [100_000.0, 101_000.0]}).to_csv(
            config.out_dir / "equity.csv", index=False)
        return SimpleNamespace(run_dir=config.out_dir, final_equity=101_000.0,
                               n_orders=3, n_deaths=0)

    monkeypatch.setattr(fruitfly.loop, "run_backtest", fake_run)
    monkeypatch.setattr(
        fruitfly.train, "load_larval_weights",
        lambda artifact, brain: SimpleNamespace(weights=None, meta=meta))
    monkeypatch.setattr(fruitfly.connectome, "load_stripped_chassis",
                        lambda: object())
    monkeypatch.setattr(eval_brains, "cache_trading_days",
                        lambda: list(DAYS))


def test_evaluate_arm_echoes_meta_and_accepts_matching_config(
        tmp_path, monkeypatch):
    config_probe = _config(entry_credit="forecast", id_scale=0.0)
    from fruitfly.train import META_KNOBS
    meta = {
        name: str(getattr(config_probe, name))
        for name in META_KNOBS if hasattr(config_probe, name)
    }
    meta["basket"] = json.dumps(["AAA", "BBB"])
    meta["start"], meta["end"] = "2026-01-01", "2026-08-01"
    (tmp_path / "x.npz").write_bytes(b"fake")
    _install_fake_run(monkeypatch, tmp_path, meta)
    results = tmp_path / "res"
    receipt = eval_brains.evaluate_arm(
        "stripped", tmp_path / "x.npz", DAYS[:2], seed=7,
        std_beta=None, std_tau_rec_ms=None, period=None,
        basket=["AAA", "BBB"], results_dir=results, arm="cfg-a",
        knobs={"entry_credit": "forecast", "id_scale": 0.0})
    assert receipt["meta_knobs"]["entry_credit"] == "forecast"
    assert receipt["meta_basket"] == ["AAA", "BBB"]
    assert receipt["arm"] == "cfg-a"
    assert (results / "results_cfg-a.json").exists()


def test_evaluate_arm_rejects_mismatched_meta(tmp_path, monkeypatch):
    config_probe = _config()
    from fruitfly.train import META_KNOBS
    meta = {
        name: str(getattr(config_probe, name))
        for name in META_KNOBS if hasattr(config_probe, name)
    }
    meta["basket"] = json.dumps(["AAA", "BBB"])
    meta["r_scale"] = "0.005"
    meta["start"], meta["end"] = "2026-01-01", "2026-08-01"
    (tmp_path / "x.npz").write_bytes(b"fake")
    _install_fake_run(monkeypatch, tmp_path, meta)
    with pytest.raises(eval_brains.MetaMismatch, match="r_scale"):
        eval_brains.evaluate_arm(
            "stripped", tmp_path / "x.npz", DAYS[:2], seed=7,
            std_beta=None, std_tau_rec_ms=None, period=None,
            basket=["AAA", "BBB"], results_dir=tmp_path / "res",
            knobs={"r_scale": 0.01})


def test_main_enforces_contract_through_the_cli(tmp_path, monkeypatch):

    meta = {"start": "2026-01-01", "end": "2026-08-01",
            "basket": json.dumps(["AAA", "BBB"])}
    _install_fake_run(monkeypatch, tmp_path, meta)
    (tmp_path / "x.npz").write_bytes(b"fake")
    # Legacy meta (no A9 knobs) + all-default config -> warn and continue.
    with pytest.warns(UserWarning, match="legacy artifact"):
        eval_brains.main([
            "--chassis", "stripped", "--artifact", str(tmp_path / "x.npz"),
            "--days", "2", "--basket", "AAA,BBB",
            "--results-dir", str(tmp_path / "res"),
            "--report", str(tmp_path / "r.md"),
        ])
    # A recorded knob that disagrees with the eval config -> MetaMismatch
    # (a ValueError; the __main__ guard maps that to exit 2).
    meta["r_scale"] = "0.005"
    _install_fake_run(monkeypatch, tmp_path, meta)
    (tmp_path / "x2.npz").write_bytes(b"fake")
    with pytest.raises(eval_brains.MetaMismatch, match="r_scale"):
        eval_brains.main([
            "--chassis", "stripped", "--artifact", str(tmp_path / "x2.npz"),
            "--days", "2", "--basket", "AAA,BBB", "--r-scale", "0.01",
            "--results-dir", str(tmp_path / "res2"),
            "--report", str(tmp_path / "r2.md"),
        ])
