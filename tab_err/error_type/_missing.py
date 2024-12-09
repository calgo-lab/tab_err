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

    def _apply(self: MissingValue, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        series = get_column(data, column).copy()
        series_mask = get_column(error_mask, column)
        series[series_mask] = self.config.missing_value
        return series
