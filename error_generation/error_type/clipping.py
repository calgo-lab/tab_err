from __future__ import annotations

from typing import TYPE_CHECKING

from pandas.api.types import is_numeric_dtype

from error_generation.error_type import ErrorType
from error_generation.utils import get_column

if TYPE_CHECKING:
    import pandas as pd

class ValueClipping(ErrorType):
    """Simulate a column containing values that are clipped at specified quantiles.
    
    Values below the lower quantile are set to the value at that quantile, and values above the upper quantile are set to the value at that quantile.
    If any of the quantiles are None, the corresponding clipping is not applied.
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

        # Calculate the quantile-based bounds
        lower_bound = series.quantile(self.config.clip_lower_quantile) if self.config.clip_lower_quantile is not None else None
        upper_bound = series.quantile(self.config.clip_upper_quantile) if self.config.clip_upper_quantile is not None else None

        # Apply clipping to the values where the mask is True
        series.loc[series_mask] = series.loc[series_mask].clip(lower=lower_bound, upper=upper_bound)
        
        return series