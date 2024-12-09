from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from tab_err._utils import get_column, get_column_str

from ._error_mechanism import ErrorMechanism

if TYPE_CHECKING:
    import pandas as pd


class EAR(ErrorMechanism):
    # TODO(seja): Docs
    def _sample(self: EAR, data: pd.DataFrame, column: str | int, error_rate: float, error_mask: pd.DataFrame) -> pd.DataFrame:
        if len(data.columns) < 2:  # noqa: PLR2004
            msg = "The data into which error at random (EAR) are to be injected requires at least 2 columns."
            raise ValueError(msg)

        if self.condition_to_column is None:
            col = get_column_str(data, column)
            column_selection = [x for x in data.columns if x != col]
            condition_to_column = self._random_generator.choice(column_selection)
            warnings.warn(
                "The user did not specify 'condition_to_column', the column on which the EAR Mechanism conditions the error distribution. "
                + f"Randomly select column '{condition_to_column}'.",
                stacklevel=1,
            )
        else:
            condition_to_column = get_column_str(data, self.condition_to_column)

        se_data = get_column(data, column)
        se_mask = get_column(error_mask, column)
        n_errors = int(se_data.size * error_rate)

        se_mask_error_free = se_mask[~se_mask]
        data_column_error_free = data.loc[se_mask_error_free.index, :]

        if len(se_mask_error_free) < n_errors:
            msg = f"The error rate of {error_rate} requires {n_errors} error-free cells. "
            msg += f"However, only {len(se_mask_error_free)} error-free cells are available."
            raise ValueError(msg)

        # we offset the upper bound of the lower_error_index by a) the existing number of errors in the row, and b) the number of errors to-be generated.
        upper_bound = len(se_data) - sum(se_mask) - n_errors
        lower_error_index = self._random_generator.integers(0, upper_bound) if upper_bound > 0 else 0
        error_index_range = range(lower_error_index, lower_error_index + n_errors)
        selected_rows = data_column_error_free.sort_values(by=condition_to_column).iloc[error_index_range, :]

        se_mask.loc[selected_rows.index] = True

        return error_mask
