from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from pandas.api.types import is_string_dtype

from tab_err._utils import get_column

from ._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


class Replace(ErrorType):
    """Replace a part of strings within a column."""

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not is_string_dtype(series):
            msg = f"Column {column} does not contain values of the string dtype. Cannot Permutate values."
            raise TypeError(msg)

    def _get_valid_columns(self: Replace, data: pd.DataFrame) -> list[str | int]:
        """Returns column names with string dtype elements."""
        return data.select_dtypes(include=["string", "object"]).columns.to_list()

    def _apply(self: Replace, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the Replace ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Returns:
            pd.Series: The data column, 'column', after Replace errors at the locations specified by 'error_mask' are introduced.
        """
        series = get_column(data, column).copy()
        series_mask = get_column(error_mask, column)

        if self.config.replace_what is None:
            msg = "The 'replace_what' parameter is not configured, defaulting to a random character from the given series. Replacements are not guaranteed."
            warnings.warn(msg, stacklevel=2)
            random_row = self._random_generator.choice(series.index)
            self.config.replace_what = self._random_generator.choice(list(series[random_row]))

        series.loc[series_mask] = series.loc[series_mask].apply(lambda x: x.replace(self.config.replace_what, self.config.replace_with))
        return series
