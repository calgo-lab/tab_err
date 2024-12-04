from __future__ import annotations

from typing import TYPE_CHECKING

from tab_err.error_type import ErrorType
from tab_err.utils import get_column

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
    def _check_type(table: pd.DataFrame, column: int | str) -> None:
        # all dtypes are supported
        pass

    def _apply(self: MissingValue, table: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        series = get_column(table, column).copy()
        series_mask = get_column(error_mask, column)
        series[series_mask] = self.config.na_value
        return series
