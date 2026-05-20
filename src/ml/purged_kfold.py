from __future__ import annotations

from typing import Generator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """
    Sklearn-compatible CV splitter that:
    1. Purges training observations whose labels overlap with the test fold
    2. Appends an embargo period after the test fold to prevent leakage

    Args:
        n_splits:    Number of folds
        t1:          pd.Series(label_end_time, index=observation_time)
        pct_embargo: Embargo fraction of total data (e.g. 0.01 = 1%)
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: Optional[pd.Series] = None,
        pct_embargo: float = 0.01,
    ):
        super().__init__()
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(
        self,
        X: pd.DataFrame,
        y=None,
        groups=None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yields (train_indices, test_indices) with purging + embargo."""
        if self.t1 is None:
            raise ValueError("PurgedKFold benötigt t1 (label end times).")

        indices = np.arange(len(X))
        embargo_size = int(len(X) * self.pct_embargo)

        test_splits = np.array_split(indices, self.n_splits)

        for test_idx in test_splits:
            if len(test_idx) == 0:
                continue

            test_start_time = X.index[test_idx[0]]
            test_end_time = X.index[test_idx[-1]]

            # Embargo: exclude bars immediately after test fold
            embargo_end_pos = min(test_idx[-1] + embargo_size, len(X) - 1)
            embargo_end_time = X.index[embargo_end_pos]

            # Build train set: exclude test fold + embargo + purge overlaps
            train_idx = []
            for i in indices:
                if i in set(test_idx):
                    continue
                # Embargo: skip bars right after test fold
                if X.index[i] > test_end_time and X.index[i] <= embargo_end_time:
                    continue
                # Purge: skip if this observation's label overlaps test start
                obs_time = X.index[i]
                if obs_time in self.t1.index:
                    label_end = self.t1[obs_time]
                    if label_end >= test_start_time:
                        continue
                train_idx.append(i)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield np.array(train_idx), test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
