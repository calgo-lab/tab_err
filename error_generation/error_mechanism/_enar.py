from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from error_generation.utils import get_column

from ._base import ErrorMechanism

if TYPE_CHECKING:
    import pandas as pd


class ENAR(ErrorMechanism):
    def _sample(self: ENAR, data: pd.DataFrame, column: str | int, error_rate: float, error_mask: pd.DataFrame | None = None) -> pd.DataFrame:
        se_data = get_column(data, column)
        se_mask = get_column(error_mask, column)

        if self.condition_to_column is not None:
            warnings.warn("'condition_to_column' is set but will be ignored by ENAR.", stacklevel=1)

        n_errors = int(len(se_data) * error_rate)

        # if mid-level or high-level API call ENAR, the error_mask already contains errors. Below we make sure that we only sample rows that do not
        # already contain errors.
        se_data_error_free = se_data[~se_mask]

        if len(se_data_error_free) < n_errors:
            msg = f"The error rate of {error_rate} requires {n_errors} error-free cells. "
            msg += f"However, only {len(se_data_error_free)} error-free cells are available."
            raise ValueError(msg)

        if len(se_data_error_free) != n_errors:  # noqa: SIM108
            lower_error_index = np.random.default_rng(seed=self.seed).integers(0, len(se_data_error_free) - n_errors)
        else:
            lower_error_index = 0
        error_index_range = range(lower_error_index, lower_error_index + n_errors)
        selected_rows = se_data_error_free.sort_values().iloc[error_index_range]

        se_mask.loc[selected_rows.index] = True

        return error_mask
