"""Tests for src/ml/purged_kfold.py — Purged K-Fold mit Embargo."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.purged_kfold import PurgedKFold
from src.ml.sample_weights import get_t1


@pytest.fixture
def X_and_t1():
    idx = pd.date_range("2022-01-01", periods=300, freq="1h", tz="UTC")
    X = pd.DataFrame({"feat": np.random.randn(300)}, index=idx)
    signal_idx = idx[::10]
    t1 = get_t1(signal_idx, idx, max_bars=15)
    # Align t1 to X index
    t1_full = pd.Series(index=idx, dtype="datetime64[ns, UTC]")
    t1_full.update(t1)
    t1_full = t1_full.dropna()
    return X, t1_full


class TestPurgedKFold:
    def test_yields_correct_number_of_splits(self, X_and_t1):
        X, t1 = X_and_t1
        pkf = PurgedKFold(n_splits=5, t1=t1)
        splits = list(pkf.split(X))
        assert len(splits) == 5

    def test_train_test_disjoint(self, X_and_t1):
        X, t1 = X_and_t1
        pkf = PurgedKFold(n_splits=5, t1=t1)
        for train_idx, test_idx in pkf.split(X):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_test_indices_cover_all_data(self, X_and_t1):
        X, t1 = X_and_t1
        pkf = PurgedKFold(n_splits=5, t1=t1)
        covered = set()
        for _, test_idx in pkf.split(X):
            covered.update(test_idx.tolist())
        assert covered == set(range(len(X)))

    def test_train_always_smaller_than_full(self, X_and_t1):
        X, t1 = X_and_t1
        pkf = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.02)
        for train_idx, test_idx in pkf.split(X):
            assert len(train_idx) < len(X)
            assert len(test_idx) > 0

    def test_no_t1_raises(self, X_and_t1):
        X, _ = X_and_t1
        pkf = PurgedKFold(n_splits=3, t1=None)
        with pytest.raises(ValueError, match="t1"):
            list(pkf.split(X))

    def test_get_n_splits_returns_correct_value(self, X_and_t1):
        X, t1 = X_and_t1
        pkf = PurgedKFold(n_splits=4, t1=t1)
        assert pkf.get_n_splits() == 4

    def test_embargo_reduces_training_set(self, X_and_t1):
        """Higher embargo → fewer training samples."""
        X, t1 = X_and_t1
        pkf_no_emb  = PurgedKFold(n_splits=3, t1=t1, pct_embargo=0.0)
        pkf_emb     = PurgedKFold(n_splits=3, t1=t1, pct_embargo=0.05)
        train_sizes_no_emb = [len(tr) for tr, _ in pkf_no_emb.split(X)]
        train_sizes_emb    = [len(tr) for tr, _ in pkf_emb.split(X)]
        avg_no_emb = np.mean(train_sizes_no_emb)
        avg_emb    = np.mean(train_sizes_emb)
        assert avg_no_emb >= avg_emb
