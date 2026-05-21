"""Tests for src/ml/data_store.py, src/ml/automl.py, src/ml/adaptive_features.py."""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.features.ml_features import ML_FEATURE_COLS
from src.ml.adaptive_features import AdaptiveFeatureSelector
from src.ml.automl import _defaults, autotune_lgbm, merge_params
from src.ml.data_store import TradeDataStore, _safe_float


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_Xy(n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, len(ML_FEATURE_COLS))), columns=ML_FEATURE_COLS)
    y = pd.Series(rng.integers(0, 2, n), name="label")
    return X, y


def _mock_model(importances: dict | None = None):
    m = MagicMock()
    if importances is None:
        importances = {feat: float(i) for i, feat in enumerate(ML_FEATURE_COLS)}
    m.feature_importances_ = importances
    return m


# ── TradeDataStore ────────────────────────────────────────────────────────────

class TestTradeDataStore:
    def test_log_entry_creates_pending_record(self, tmp_path):
        store = TradeDataStore(path=str(tmp_path / "db.sqlite"))
        feats = {col: 0.1 for col in ML_FEATURE_COLS}
        store.log_entry("BTCUSDT", "2024-01-01", 1, feats, ticket=1001)
        s = store.stats()
        assert s["total_trades"] == 1
        assert s["pending"] == 1
        assert s["wins"] == 0

    def test_update_outcome_win(self, tmp_path):
        store = TradeDataStore(path=str(tmp_path / "db.sqlite"))
        feats = {col: 0.5 for col in ML_FEATURE_COLS}
        store.log_entry("BTCUSDT", "2024-01-01", 1, feats, ticket=42)
        store.update_outcome(42, outcome=1, pnl=100.0)
        s = store.stats()
        assert s["wins"] == 1
        assert s["losses"] == 0
        assert s["pending"] == 0

    def test_update_outcome_loss(self, tmp_path):
        store = TradeDataStore(path=str(tmp_path / "db.sqlite"))
        feats = {col: 0.0 for col in ML_FEATURE_COLS}
        store.log_entry("ETHUSDT", "2024-01-02", -1, feats, ticket=99)
        store.update_outcome(99, outcome=0, pnl=-50.0)
        s = store.stats()
        assert s["losses"] == 1
        assert s["win_rate"] == 0.0

    def test_get_training_data_below_min_returns_none(self, tmp_path):
        store = TradeDataStore(path=str(tmp_path / "db.sqlite"))
        feats = {col: 0.1 for col in ML_FEATURE_COLS}
        for i in range(10):
            store.log_entry("BTCUSDT", f"2024-01-0{i+1}", 1, feats, ticket=i)
            store.update_outcome(i, outcome=1, pnl=10.0)
        result = store.get_training_data(min_samples=50)
        assert result is None

    def test_get_training_data_returns_Xy(self, tmp_path):
        store = TradeDataStore(path=str(tmp_path / "db.sqlite"))
        feats = {col: float(i) for i, col in enumerate(ML_FEATURE_COLS)}
        for i in range(60):
            store.log_entry("BTCUSDT", f"2024-01-{i:02d}", 1, feats, ticket=i)
            store.update_outcome(i, outcome=i % 2, pnl=10.0 if i % 2 else -10.0)
        result = store.get_training_data(min_samples=50)
        assert result is not None
        X, y = result
        assert len(X) == 60
        assert list(X.columns) == ML_FEATURE_COLS
        assert y.nunique() == 2

    def test_duplicate_ticket_ignored(self, tmp_path):
        store = TradeDataStore(path=str(tmp_path / "db.sqlite"))
        feats = {col: 0.0 for col in ML_FEATURE_COLS}
        store.log_entry("BTCUSDT", "2024-01-01", 1, feats, ticket=1)
        store.log_entry("BTCUSDT", "2024-01-01", 1, feats, ticket=1)  # duplicate
        assert store.stats()["total_trades"] == 1

    def test_safe_float_handles_nan_and_invalid(self):
        assert _safe_float(float("nan")) == 0.0
        assert _safe_float("bad") == 0.0
        assert _safe_float(3.14) == pytest.approx(3.14)
        assert _safe_float(0) == 0.0


# ── autotune_lgbm ─────────────────────────────────────────────────────────────

class TestAutotuneLgbm:
    def test_returns_required_keys(self):
        X, y = _make_Xy(100)
        params = autotune_lgbm(X, y, n_trials=5, n_splits=2)
        for key in ("lgbm_num_leaves", "lgbm_learning_rate",
                    "lgbm_n_estimators", "lgbm_min_child_samples"):
            assert key in params

    def test_insufficient_data_returns_defaults(self):
        X, y = _make_Xy(10)
        params = autotune_lgbm(X, y, n_trials=5)
        assert params == _defaults()

    def test_single_class_returns_defaults(self):
        X, y = _make_Xy(100)
        y[:] = 1  # only class 1
        params = autotune_lgbm(X, y, n_trials=5)
        assert params == _defaults()

    def test_values_in_valid_range(self):
        X, y = _make_Xy(120)
        params = autotune_lgbm(X, y, n_trials=5, n_splits=2)
        assert 15 <= params["lgbm_num_leaves"] <= 127
        assert 0.01 <= params["lgbm_learning_rate"] <= 0.15
        assert 100 <= params["lgbm_n_estimators"] <= 400
        assert 5 <= params["lgbm_min_child_samples"] <= 50

    def test_merge_params_does_not_mutate_original(self):
        import copy
        cfg = {"ml": {"lgbm_num_leaves": 50}}
        original = copy.deepcopy(cfg)
        best = {"lgbm_num_leaves": 80, "lgbm_learning_rate": 0.02}
        result = merge_params(cfg, best)
        assert cfg == original
        assert result["ml"]["lgbm_num_leaves"] == 80
        assert result["ml"]["lgbm_learning_rate"] == pytest.approx(0.02)


# ── AdaptiveFeatureSelector ───────────────────────────────────────────────────

class TestAdaptiveFeatureSelector:
    def test_all_features_active_initially(self, tmp_path):
        sel = AdaptiveFeatureSelector(path=str(tmp_path / "feat.db"))
        assert sel.get_active_features() == ML_FEATURE_COLS

    def test_update_records_cycle(self, tmp_path):
        sel = AdaptiveFeatureSelector(path=str(tmp_path / "feat.db"))
        model = _mock_model()
        sel.update(model)
        df = sel.report()
        assert not df.empty

    def test_model_without_importances_is_skipped(self, tmp_path):
        sel = AdaptiveFeatureSelector(path=str(tmp_path / "feat.db"))
        m = MagicMock()
        m.feature_importances_ = None
        sel.update(m)
        assert sel.report().empty

    def test_consistently_zero_feature_gets_dropped(self, tmp_path):
        """A feature with zero importance for min_cycles consecutive cycles is dropped."""
        sel = AdaptiveFeatureSelector(
            path=str(tmp_path / "feat.db"),
            drop_percentile=0.10,
            min_cycles_to_drop=3,
        )
        target = ML_FEATURE_COLS[0]
        for _ in range(3):
            importances = {
                feat: (0.0 if feat == target else 100.0)
                for feat in ML_FEATURE_COLS
            }
            sel.update(_mock_model(importances))

        assert target in sel.dropped_features()
        assert target not in sel.get_active_features()

    def test_feature_not_dropped_before_min_cycles(self, tmp_path):
        sel = AdaptiveFeatureSelector(
            path=str(tmp_path / "feat.db"),
            drop_percentile=0.10,
            min_cycles_to_drop=3,
        )
        target = ML_FEATURE_COLS[0]
        # Only 2 cycles — not enough to drop
        for _ in range(2):
            importances = {f: (0.0 if f == target else 100.0) for f in ML_FEATURE_COLS}
            sel.update(_mock_model(importances))

        assert target not in sel.dropped_features()

    def test_active_features_are_subset_of_ml_feature_cols(self, tmp_path):
        sel = AdaptiveFeatureSelector(path=str(tmp_path / "feat.db"))
        for _ in range(3):
            sel.update(_mock_model())
        active = sel.get_active_features()
        assert all(f in ML_FEATURE_COLS for f in active)
