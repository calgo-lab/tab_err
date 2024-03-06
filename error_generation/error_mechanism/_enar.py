from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._base import ErrorMechanism

if TYPE_CHECKING:
    from pandas._typing import Dtype


class ENAR(ErrorMechanism):
    @staticmethod
    def _sample(data: pd.DataFrame, error_rate: float, condition_to_column: Dtype | None = None, seed: int | None = None) -> pd.DataFrame:
        if condition_to_column is not None:
            warnings.warn("'condition_to_column' is set but will be ignored by ENAR.", stacklevel=1)

        error_mask = pd.DataFrame(data=False, index=data.index, columns=data.columns)

        # distribute errors equally over all columns
        how_many_error_cells_for_each_column = int(data.size * error_rate / len(data.columns))
        for column in data.columns:
            lower_error_index = np.random.default_rng(seed=seed).integers(0, len(data) - how_many_error_cells_for_each_column)
            error_index_range = range(lower_error_index, lower_error_index + how_many_error_cells_for_each_column)

            error_mask.loc[data.sort_values(by=column).index[error_index_range], column] = True

            # avoid sample same indices for each column
            if isinstance(seed, int):
                seed += 1

        return error_mask
