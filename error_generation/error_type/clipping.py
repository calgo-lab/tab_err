from __future__ import annotations

from typing import TYPE_CHECKING

from pandas.api.types import is_numeric_dtype

from error_generation.error_type import ErrorType
from error_generation.utils import get_column

if TYPE_CHECKING:
    import pandas as pd

class ValueClipping(ErrorType):
    """Simulate a column containing values that are clipped at a specified lower and upper bound.
    
    Values below the lower bound are set to the lower bound, values above the upper bound are set to the upper bound.
    If any of the bounds are None, the corresponding clipping is not applied.
    """

    @staticmethod
    def _check_type(table: pd.DataFrame, column: int | str) -> None:
        series = get_column(table, column)

        if not is_numeric_dtype(series):
            msg = f"Column {column} does not contain numeric values. Cannot apply value clipping."
            raise TypeError(msg)

    def _apply(self: ValueClipping, table: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        # Get the column series and mask
        series = get_column(table, column).copy()
        series_mask = get_column(error_mask, column)

        # Apply clipping to the values where the mask is True
        series.loc[series_mask] = series.loc[series_mask].clip(lower=self.config.clip_lower_bound, upper=self.config.clip_upper_bound)
        
        return series