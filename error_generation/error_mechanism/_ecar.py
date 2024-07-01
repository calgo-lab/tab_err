from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from error_generation.utils import get_column

from ._base import ErrorMechanism

if TYPE_CHECKING:
    import pandas as pd


class ECAR(ErrorMechanism):
    def _sample(self: ECAR, data: pd.DataFrame, column: str | int, error_rate: float, error_mask: pd.DataFrame) -> pd.DataFrame:
        se_data = get_column(data, column)
        se_mask = get_column(error_mask, column)

        if self.condition_to_column is not None:
            warnings.warn("'condition_to_column' is set but will be ignored by ECAR.", stacklevel=1)

        n_errors = int(se_data.size * error_rate)

        # randomly choose error-cells
        error_indices = np.random.default_rng(seed=self.seed).choice(se_data.size, n_errors, replace=False)
        se_mask[error_indices] = True
        return error_mask
