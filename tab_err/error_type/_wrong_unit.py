from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import narwhals as nw

import numpy as np

from tab_err._utils import get_column, is_numeric_dtype, new_series_like, select_numeric_columns

from ._error_type import ErrorType


class WrongUnit(ErrorType):
    """Simulate a column containing values that are scaled because they are not stored in the same unit."""

    @staticmethod
    def _check_type(data: nw.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not is_numeric_dtype(series):
            msg = f"Column {column} with dtype: {series.dtype} does not contain scalars. Cannot apply a wrong unit."
            raise TypeError(msg)

    def _get_valid_columns(self: WrongUnit, data: nw.DataFrame) -> list[str | int]:
        """Returns all column names with numeric dtype elements."""
        return select_numeric_columns(data)

    def _apply(self: WrongUnit, data: nw.DataFrame, error_mask: nw.DataFrame, column: int | str) -> nw.Series:
        """Applies the WrongUnit ErrorType to a column of data.

        Args:
            data (nw.DataFrame): DataFrame containing the column to add errors to.
            error_mask (nw.DataFrame): A DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Returns:
            nw.Series: The data column, 'column', after Replace errors at the locations specified by 'error_mask' are introduced.
        """
        if self.config.wrong_unit_scaling is None:
            msg = "No scaling function was supplied for WrongUnit, defaulting to multiplication by 10.0."
            warnings.warn(msg, stacklevel=2)
            self.config.wrong_unit_scaling = lambda x: 10.0 * x

        series = get_column(data, column)
        series_mask = get_column(error_mask, column)

        # Get numpy arrays
        data_arr = series.to_numpy().copy().astype(np.float64)
        mask_arr = series_mask.to_numpy()

        # Apply scaling function where mask is True
        for i in range(len(data_arr)):
            if mask_arr[i]:
                data_arr[i] = self.config.wrong_unit_scaling(data_arr[i])

        return new_series_like(data, column, data_arr)
