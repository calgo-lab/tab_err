from __future__ import annotations

from typing import TYPE_CHECKING

from tab_err._utils import get_column

from ._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


class Mistype(ErrorType):
    """Insert incorrectly typed values into a column. Note that the dtype of the column is changed by this operation.

    - String / Object is the dead end of typing
    In an effort to keep the code relatively simple, we cast the corrupted column to an Object Dtype.
    """

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        # all dtypes are supported
        pass

    def _get_valid_columns(self: Mistype, data: pd.DataFrame) -> list[str | int]:
        """Returns all column names of columns with dtypes other than object. This is necessary for the high level API."""
        return [col_name for col_name in data.columns.tolist() if data[col_name].dtype != "object"]

    def _apply(self: Mistype, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the Mistype ErrorType to a column of data. Note that the dtype of the column is changed by this operation.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            TypeError: If the type supplied by the user in the config is not supported, a TypeError will be thrown.
            TypeError: If no type is supplied by the user in the config, and the series' datatype is 'object', a TypeError will be thrown.

        Returns:
            pd.Series: The data column, 'column', after Mistype errors at the locations specified by 'error_mask' are introduced.
        """
        series = get_column(data, column).copy()
        supported_dtypes = ["object", "string", "int64", "Int64", "float64", "Float64"]

        if self.config.mistype_dtype is not None:
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
            else:
                msg = f"The type: {current_dtype} is unsupported. The type must be one of: {*supported_dtypes, 'bool'}."
                raise ValueError(msg)
            # NOTE(PJ): not sure about this logic, there might be a better way to do this.

        series = series.astype("object")
        series_mask = get_column(error_mask, column)
        series.loc[series_mask] = series.loc[series_mask].astype(target_dtype)

        return series
