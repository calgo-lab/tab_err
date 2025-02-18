from __future__ import annotations

from typing import TYPE_CHECKING

from tab_err._utils import get_column

from ._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


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

    def _get_valid_columns(self:MissingValue, data: pd.DataFrame, preserve_dtypes = True) -> list[str | int]:
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
        # TODO(nich): add a functionality to add a nan if the datatype is numeric and none is in the config -- string otherwise
        series = get_column(data, column).copy()
        series_mask = get_column(error_mask, column)
        series[series_mask] = self.config.missing_value
        return series
