from __future__ import annotations

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

    def _apply(self: Replace, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        series = get_column(data, column).copy()
        series_mask = get_column(error_mask, column)

        if self.config.replace_what is None:
            msg = "The 'replace_what' parameter is required to use the Replace Error Type, but it has not been configured."
            raise ValueError(msg)

        series.loc[series_mask] = series.loc[series_mask].apply(lambda x: x.replace(self.config.replace_what, self.config.replace_with))
        return series
