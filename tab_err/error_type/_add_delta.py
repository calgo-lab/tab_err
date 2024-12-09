from __future__ import annotations

from typing import TYPE_CHECKING

from tab_err._utils import get_column

from ._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


class AddDelta(ErrorType):
    """Adds a delta to values in a column."""

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        # all data types are fine
        pass

    def _apply(self: AddDelta, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        # cast to object because our operation potentially changes the type of a column.
        series = get_column(data, column).copy().astype("object")
        series_mask = get_column(error_mask, column)

        if self.config.add_delta_value is None:
            msg = "No add_delta_value has been configured. Please add it to the ErrorTypeConfig."
            raise ValueError(msg)

        series.loc[series_mask] = series.loc[series_mask].apply(lambda x: x + self.config.add_delta_value)
        return series
