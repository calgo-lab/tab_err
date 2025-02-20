from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pandas.api.types import is_numeric_dtype

from tab_err._utils import get_column

from ._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


class AddDelta(ErrorType):
    """Adds a delta to values in a column."""

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not is_numeric_dtype(series):
            msg = f"Column {column} with dtype: {series.dtype} does not contain numeric values. Cannot apply AddDelta."
            raise TypeError(msg)

    def _get_valid_columns(self:AddDelta, data: pd.DataFrame) -> list[str | int]:
        """Returns all column names with numeric dtype elements."""
        return data.select_dtypes(include=["number"]).columns.tolist()

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

        if self.config.add_delta_value is None:
            msg = f"self.config.add_delta_value is none, sampling a random delta value uniformly from the range of column: {column}"
            warnings.warn(msg, stacklevel=2)
            self.config.add_delta_value = (self._random_generator.choice(series) - series.mean())/series.std()  # Ensures a smaller value than uniform sampling

        series.loc[series_mask] = series.loc[series_mask].apply(lambda x: x + self.config.add_delta_value)
        return series
