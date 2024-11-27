from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from tab_err.utils import get_column

from ._base import ErrorMechanism

if TYPE_CHECKING:
    import pandas as pd


class ECAR(ErrorMechanism):
    def _sample(self: ECAR, data: pd.DataFrame, column: str | int, error_rate: float, error_mask: pd.DataFrame) -> pd.DataFrame:
        se_data = get_column(data, column)  # noqa: F841
    # TODO(seja): Docs
        se_mask = get_column(error_mask, column)

        se_mask_error_free = se_mask[~se_mask]

        if self.condition_to_column is not None:
            warnings.warn("'condition_to_column' is set but will be ignored by ECAR.", stacklevel=1)

        n_errors = int(se_mask.size * error_rate)

        if len(se_mask_error_free) < n_errors:
            msg = f"The error rate of {error_rate} requires {n_errors} error-free cells. "
            msg += f"However, only {len(se_mask_error_free)} error-free cells are available."
            raise ValueError(msg)

        # randomly choose error-cells
        error_indices = np.random.default_rng(seed=self.seed).choice(se_mask_error_free.index, n_errors, replace=False)
        se_mask[error_indices] = True
        return error_mask
