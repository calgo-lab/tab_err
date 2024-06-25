from __future__ import annotations

import pandas as pd

from error_generation.error_type import ErrorType
from error_generation.utils import get_column


class Mistype(ErrorType):
    """Insert incorrectly typed values into a column.

    - String / Object is the dead end of typing
    In an effort to keep the code relatively simple, we cast the corrupted column to an Object Dtype.
    """

    @staticmethod
    def _check_type(table: pd.DataFrame, column: int | str) -> None:
        # all dtypes are supported
        pass

    def _apply(self: Mistype, table: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        series = get_column(table, column).copy()

        if self.config.mistype_dtype is not None:
            is_valid_type = pd.api.types.is_dtype_registered(self.config.mistype_dtype)
            if not is_valid_type:
                msg = f"Invalid user-specified dtype {self.config.mistype_dtype}"
                raise TypeError(msg)
            target_dtype = self.config.mistype_dtype
        else:  # no user-specified dtype, use heuristict to infer one
            current_dtype = series.dtype
            if current_dtype == "object":
                msg = "Cannot infer a dtype that is safe to cast to if the original dtype is 'object'."
                raise TypeError(msg)
            if current_dtype == "string":
                target_dtype = "object"
            elif current_dtype == "int64":
                target_dtype = "float64"
            elif current_dtype == "Int64":
                target_dtype = "Float64"
            elif current_dtype == "float64":
                target_dtype = "int64"
            # not sture about this logic. There is a larget hierarchy that I could tap into.

        series_mask = get_column(error_mask, column)
        series.loc[series_mask] = series.loc[series_mask].astype(target_dtype)

        return series
