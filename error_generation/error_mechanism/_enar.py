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

        # distribute errors equally over all columns
        n_errors = int(se_data.size * error_rate)

        if n_errors < len(se_data):  # noqa: SIM108
            lower_error_index = np.random.default_rng(seed=self.seed).integers(0, len(se_data) - n_errors)
        else:  # all cells are errors
            lower_error_index = 0
        error_index_range = range(lower_error_index, lower_error_index + n_errors)

        se_mask.loc[se_data.sort_values().index[error_index_range]] = True

        # TODO(PJ): Remember to run if isinstance(seed, int): seed += 1 in mid-level API

        return error_mask
