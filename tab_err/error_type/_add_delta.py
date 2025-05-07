from __future__ import annotations

import warnings

import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_numeric_dtype

from tab_err._utils import get_column

from ._error_type import ErrorType


class AddDelta(ErrorType):
    """Adds a delta to values in a column."""

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not (is_numeric_dtype(series) or is_datetime64_dtype(series)):
            msg = f"Column {column} with dtype: {series.dtype} does not contain numeric or datetime64 values. Cannot apply AddDelta."
            raise TypeError(msg)

    def _get_valid_columns(self: AddDelta, data: pd.DataFrame) -> list[str | int]:
        """Returns all column names with numeric dtype elements."""
        return data.select_dtypes(include=["number", "datetime64"]).columns.tolist()

    def _apply(self: AddDelta, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the AddDelta ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            ValueError: If the add_delta_value is None, a ValueError will be thrown.

        Returns:
            pd.Series: The data column, 'column', after AddDelta errors at the locations specified by 'error_mask' are introduced.
        """
        series = get_column(data, column).copy()
        series_mask = get_column(error_mask, column)
        was_datetime = False  # Default was_datetime to false -- changes occur only in the special case of datetime

        if is_datetime64_dtype(series):  # Convert to int (number of seconds) if datetime
            series = series.astype("int64") // 10**9
            was_datetime = True

        if self.config.add_delta_value is None:
            msg = f"self.config.add_delta_value is none, sampling a random delta value uniformly from the range of column: {column}."
            warnings.warn(msg, stacklevel=2)
            self.config.add_delta_value = (
                self._random_generator.choice(series) - series.mean()
            ) / series.std()  # Ensures a smaller value than uniform sampling

        series = series.where(~series_mask, series + self.config.add_delta_value)  # Avoids in-place modification

        if was_datetime:  # Convert back to datetime if it was initially
            series = pd.to_datetime(series, unit="s")

        return series
