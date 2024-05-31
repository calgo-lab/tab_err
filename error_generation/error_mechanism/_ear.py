from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas._typing import Dtype
from ._base import ErrorMechanism


class EAR(ErrorMechanism):
    @staticmethod
    def _sample(data: pd.DataFrame, error_rate: float, condition_to_column: Dtype | None = None, seed: int | None = None) -> pd.DataFrame:
        if len(data.columns) < 2:  # noqa: PLR2004
            msg = "'data' need at least 2 columns."
            raise ValueError(msg)

        if condition_to_column is None:
            condition_to_column = np.random.default_rng(seed).choice(data.columns)

        # NOTE: We do not perturb the 'condition_to_column' column, so we need to handle the following edge case:
        # If 'columns_to_create_errors' don't have enough cells to reach 'error_rate',
        # we simply perturb as many as possible, i.e., all cells in 'columns_to_create_errors'
        columns_to_create_errors = data.columns.drop(condition_to_column)
        if error_rate >= len(columns_to_create_errors) / len(data.columns):
            warnings.warn(
                "Given 'error_rate' can not be fulfilled because EAR does not perturb the 'condition_to_column'. Will perturb as many cells as possible.",
                stacklevel=1,
            )
            is_error_rate_too_large = True
            how_many_error_cells_for_each_column = len(data)

        else:
            is_error_rate_too_large = False
            how_many_error_cells_for_each_column = int(data.size * error_rate / len(columns_to_create_errors))

        error_mask = pd.DataFrame(data=False, index=data.index, columns=data.columns)

        # distribute errors equally over all columns but not 'condition_to_column'
        for column in columns_to_create_errors:
            lower_error_index = np.random.default_rng(seed).integers(
                0,
                1 if is_error_rate_too_large else len(data) - how_many_error_cells_for_each_column,
            )
            error_index_range = range(lower_error_index, lower_error_index + how_many_error_cells_for_each_column)

            error_mask.loc[data.sort_values(by=condition_to_column).index[error_index_range], column] = True

            # avoid sample same indices for each column
            if isinstance(seed, int):
                seed += 1

        return error_mask
