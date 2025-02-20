from __future__ import annotations

import pandas as pd
from pandas.api.types import is_string_dtype

from tab_err._utils import get_column

from ._error_type import ErrorType


class MissingValue(ErrorType):
    """Insert missing values into a column.

    Missing value handling is not a solved problem in pandas and under active development.
    Today, the best heuristic for inserting missing values is to assign None to the value.
    Pandas will choose the missing value sentinel based on the column dtype
    (https://pandas.pydata.org/docs/user_guide/missing_data.html#inserting-missing-data).
    """

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        # all dtypes are supported
        pass

    def _get_valid_columns(self: MissingValue, data: pd.DataFrame) -> list[str | int]:
        """If the config mising value is None, returns all columns. Otherwise, only the columns with the same type."""
        return data.columns.to_list() if self.config.missing_value is None else data.select_dtypes(include=["object", "string"]).columns.to_list()

    def _apply(self: MissingValue, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the MissingValue ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Returns:
            pd.Series: The data column, 'column', after MissingValue errors at the locations specified by 'error_mask' are introduced.
        """
        series = get_column(data, column).copy()
        series_mask = get_column(error_mask, column)

        if is_string_dtype(series) and self.config.missing_value is None:  # Strings are finicky
            series[series_mask] = pd.NA
            series = series.astype(str)
        else:
            series[series_mask] = self.config.missing_value

        return series
