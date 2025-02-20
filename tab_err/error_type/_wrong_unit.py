from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pandas.api.types import is_numeric_dtype

from tab_err._utils import get_column

from ._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


class WrongUnit(ErrorType):
    """Simulate a column containing values that are scaled because they are not stored in the same unit."""

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not is_numeric_dtype(series):
            msg = f"Column {column} with dtype: {series.dtype} does not contain scalars. Cannot apply a wrong unit."
            raise TypeError(msg)

    def _get_valid_columns(self: WrongUnit, data: pd.DataFrame) -> list[str | int]:
        """Returns all column names with numeric dtype elements."""
        return data.select_dtypes(include=["number"]).columns.tolist()

    def _apply(self: WrongUnit, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the WrongUnit ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Returns:
            pd.Series: The data column, 'column', after Replace errors at the locations specified by 'error_mask' are introduced.
        """
        if self.config.wrong_unit_scaling is None:
            msg = "No scaling function was supplied for WrongUnit, defaulting to multiplication by 10.0."
            warnings.warn(msg, stacklevel=2)
            self.config.wrong_unit_scaling = lambda x: 10.0 * x

        series = get_column(data, column).copy()
        series_mask = get_column(error_mask, column)

        series.loc[series_mask] = series.loc[series_mask].apply(self.config.wrong_unit_scaling)
        return series
