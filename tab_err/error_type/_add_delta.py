from __future__ import annotations

import warnings

import narwhals as nw
import numpy as np

from tab_err._utils import get_column, get_column_str, is_datetime_dtype, is_numeric_dtype, select_numeric_or_datetime_columns

from ._error_type import ErrorType


class AddDelta(ErrorType):
    """Adds a delta to values in a column."""

    @staticmethod
    def _check_type(data: nw.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not (is_numeric_dtype(series) or is_datetime_dtype(series)):
            msg = f"Column {column} with dtype: {series.dtype} does not contain numeric or datetime64 values. Cannot apply AddDelta."
            raise TypeError(msg)

    def _get_valid_columns(self: AddDelta, data: nw.DataFrame) -> list[str | int]:
        """Returns all column names with numeric dtype elements."""
        return select_numeric_or_datetime_columns(data)

    def _apply(self: AddDelta, data: nw.DataFrame, error_mask: nw.DataFrame, column: int | str) -> nw.Series:
        """Applies the AddDelta ErrorType to a column of data.

        Args:
            data (nw.DataFrame): DataFrame containing the column to add errors to.
            error_mask (nw.DataFrame): A DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            ValueError: If the add_delta_value is None, a ValueError will be thrown.

        Returns:
            nw.Series: The data column, 'column', after AddDelta errors at the locations specified by 'error_mask' are introduced.
        """
        col_name = get_column_str(data, column)
        series = get_column(data, column)
        series_mask = get_column(error_mask, column)
        was_datetime = False

        # Get numpy arrays
        data_arr = series.to_numpy().copy()
        mask_arr = series_mask.to_numpy()

        if is_datetime_dtype(series):
            # Convert datetime to seconds since epoch
            data_arr = data_arr.astype("datetime64[ns]").astype("int64") // 10**9
            was_datetime = True

        # Ensure float for calculations
        data_arr = data_arr.astype(np.float64)

        if self.config.add_delta_value is None:
            msg = f"self.config.add_delta_value is none, sampling a random delta value uniformly from the range of column: {column}."
            warnings.warn(msg, stacklevel=2)
            mean_val = np.nanmean(data_arr)
            std_val = np.nanstd(data_arr)
            random_choice = self._random_generator.choice(data_arr[~np.isnan(data_arr)])
            self.config.add_delta_value = (random_choice - mean_val) / std_val if std_val != 0 else 0

        # Apply delta where mask is True
        data_arr[mask_arr] += self.config.add_delta_value

        if was_datetime:
            # Convert back to datetime (from seconds)
            data_arr = (data_arr * 10**9).astype("int64").astype("datetime64[ns]")

        return nw.new_series(col_name, data_arr.tolist(), backend=nw.get_native_namespace(data))
