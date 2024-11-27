from __future__ import annotations

from typing import TYPE_CHECKING

from tab_err.error_type import ErrorType
from tab_err.utils import get_column

if TYPE_CHECKING:
    import pandas as pd


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
            supported_dtypes = ["object", "string", "int64", "Int64", "float64", "Float64"]

            if self.config.mistype_dtype not in supported_dtypes:
                msg = f"Unsupported user-specified dtype {self.config.mistype_dtype}. Supported dtypes as {supported_dtypes}."
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
            elif current_dtype == "Float64":
                target_dtype = "Int64"
            elif current_dtype == "bool":
                target_dtype = "int64"
            # PJ: not sure about this logic, there might be a better way to do this.

        series = series.astype("object")
        series_mask = get_column(error_mask, column)
        series.loc[series_mask] = series.loc[series_mask].astype(target_dtype)

        return series