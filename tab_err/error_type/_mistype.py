from __future__ import annotations

import narwhals as nw

from tab_err._utils import get_column, get_column_str

from ._error_type import ErrorType


class Mistype(ErrorType):
    """Insert incorrectly typed values into a column. Note that the dtype of the column is changed by this operation.

    - String / Object is the dead end of typing
    In an effort to keep the code relatively simple, we cast the corrupted column to an Object dtype.
    """

    @staticmethod
    def _check_type(data: nw.DataFrame, column: int | str) -> None:
        # all dtypes are supported
        pass

    def _get_valid_columns(self: Mistype, data: nw.DataFrame) -> list[str | int]:
        """Returns all column names of columns with dtypes other than object. This is necessary for the high level API."""
        return [col_name for col_name in data.columns if data[col_name].dtype != nw.Object]

    def _apply(self: Mistype, data: nw.DataFrame, error_mask: nw.DataFrame, column: int | str) -> nw.Series:
        """Applies the Mistype ErrorType to a column of data. Note that the dtype of the column is changed by this operation.

        Args:
            data (nw.DataFrame): DataFrame containing the column to add errors to.
            error_mask (nw.DataFrame): A DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            TypeError: If the type supplied by the user in the config is not supported, a TypeError will be thrown.
            TypeError: If no type is supplied by the user in the config, and the series' datatype is 'object', a TypeError will be thrown.

        Returns:
            nw.Series: The data column, 'column', after Mistype errors at the locations specified by 'error_mask' are introduced.
        """
        col_name = get_column_str(data, column)
        series = get_column(data, column)
        series_mask = get_column(error_mask, column)

        supported_dtypes = ["object", "string", "int64", "Int64", "float64", "Float64"]

        if self.config.mistype_dtype is not None:
            if self.config.mistype_dtype not in supported_dtypes:
                msg = f"Unsupported user-specified dtype {self.config.mistype_dtype}. Supported dtypes are {supported_dtypes}."
                raise TypeError(msg)

            target_dtype = self.config.mistype_dtype
        else:  # no user-specified dtype, use heuristic to infer one
            current_dtype = series.dtype
            if current_dtype == nw.Object:
                msg = "Cannot infer a dtype that is safe to cast to if the original dtype is 'object'."
                raise TypeError(msg)
            if current_dtype == nw.String:
                target_dtype = "object"
            elif current_dtype == nw.Int64:
                target_dtype = "float64"
            elif current_dtype in {nw.Float64, nw.Boolean}:
                target_dtype = "int64"
            elif current_dtype.is_integer():
                target_dtype = "float64"
            elif current_dtype.is_numeric():
                target_dtype = "int64"
            else:
                msg = f"The type: {current_dtype} is unsupported. The type must be one of: {*supported_dtypes, 'bool'}."
                raise ValueError(msg)

        # Get numpy arrays
        data_arr = series.to_numpy().copy()
        mask_arr = series_mask.to_numpy()

        # Convert to object array to allow mixed types
        result_arr = data_arr.astype(object)

        # Apply type conversion where mask is True
        for i in range(len(result_arr)):
            if mask_arr[i]:
                val = result_arr[i]
                try:
                    if target_dtype in ("int64", "Int64"):
                        result_arr[i] = int(val) if val is not None else val
                    elif target_dtype in ("float64", "Float64"):
                        result_arr[i] = float(val) if val is not None else val
                    elif target_dtype in ("object", "string"):
                        result_arr[i] = str(val) if val is not None else val
                except (ValueError, TypeError):
                    # Keep original value if conversion fails
                    pass

        return nw.new_series(col_name, result_arr.tolist(), backend=nw.get_native_namespace(data))
