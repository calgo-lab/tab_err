from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from error_generation.utils import get_column, get_column_str

from ._base import ErrorMechanism

if TYPE_CHECKING:
    import pandas as pd


class EAR(ErrorMechanism):
    def _sample(self: EAR, data: pd.DataFrame, column: str | int, error_rate: float, error_mask: pd.DataFrame) -> pd.DataFrame:
        if len(data.columns) < 2:  # noqa: PLR2004
            msg = "The table into which error at random (EAR) are to be injected requies at least 2 columns."
            raise ValueError(msg)

        if self.condition_to_column is None:
            col = get_column_str(data, column)
            column_selection = [x for x in data.columns if x != col]
            condition_to_column = np.random.default_rng(self.seed).choice(column_selection)
            warnings.warn(
                "The user did not specify 'condition_to_column', the column on which the EAR Mechanism conditions the error distribution. "
                f"Randomly select column '{condition_to_column}'.",
                stacklevel=1,
            )
        else:
            condition_to_column = get_column_str(data, self.condition_to_column)

        se_data = get_column(data, column)
        se_mask = get_column(error_mask, column)
        n_errors = int(se_data.size * error_rate)

        upper_bound = len(se_data) - n_errors
        lower_error_index = np.random.default_rng(self.seed).integers(0, upper_bound) if upper_bound > 0 else 0
        error_index_range = range(lower_error_index, lower_error_index + n_errors)

        se_mask.loc[data.sort_values(by=condition_to_column).index[error_index_range]] = True

        return error_mask
