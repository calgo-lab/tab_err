from __future__ import annotations

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
            msg = f"Column {column} does not contain scalars. Cannot apply a wrong unit."
            raise TypeError(msg)

    def _apply(self: WrongUnit, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        if self.config.wrong_unit_scaling is None:
            msg = f"Cannot apply wrong unit to column {column} because no scaling function wrong_unit_scaling was defined in the ErrorTypeConfig."
            raise ValueError(msg)

        series = get_column(data, column).astype("object").copy()
        series_mask = get_column(error_mask, column)

        series.loc[series_mask] = series.loc[series_mask].apply(self.config.wrong_unit_scaling)
        return series
