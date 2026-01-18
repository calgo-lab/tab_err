from __future__ import annotations

import warnings

import narwhals as nw

from tab_err._utils import get_column, get_column_str, is_string_dtype, select_string_columns

from ._error_type import ErrorType


class Replace(ErrorType):
    """Replace a part of strings within a column."""

    @staticmethod
    def _check_type(data: nw.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not is_string_dtype(series):
            msg = f"Column {column} does not contain values of the string dtype. Cannot Permutate values."
            raise TypeError(msg)

    def _get_valid_columns(self: Replace, data: nw.DataFrame) -> list[str | int]:
        """Returns column names with string dtype elements."""
        return select_string_columns(data)

    def _apply(self: Replace, data: nw.DataFrame, error_mask: nw.DataFrame, column: int | str) -> nw.Series:
        """Applies the Replace ErrorType to a column of data.

        Args:
            data (nw.DataFrame): DataFrame containing the column to add errors to.
            error_mask (nw.DataFrame): A DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Returns:
            nw.Series: The data column, 'column', after Replace errors at the locations specified by 'error_mask' are introduced.
        """
        col_name = get_column_str(data, column)
        series = get_column(data, column)
        series_mask = get_column(error_mask, column)

        # Get numpy arrays
        data_arr = series.to_numpy().copy()
        mask_arr = series_mask.to_numpy()

        if self.config.replace_what is None:
            msg = "The 'replace_what' parameter is not configured, defaulting to a random character from the given series. Replacements are not guaranteed."
            warnings.warn(msg, stacklevel=2)
            # Get valid values (non-None strings)
            valid_values = [v for v in data_arr if v is not None and isinstance(v, str) and len(v) > 0]
            if valid_values:
                random_row = self._random_generator.choice(len(valid_values))
                random_val = valid_values[random_row]
                self.config.replace_what = self._random_generator.choice(list(random_val))
            else:
                self.config.replace_what = ""

        # Apply replace where mask is True
        for i in range(len(data_arr)):
            if mask_arr[i]:
                val = data_arr[i]
                if val is not None and isinstance(val, str):
                    data_arr[i] = val.replace(self.config.replace_what, self.config.replace_with)

        return nw.new_series(col_name, data_arr.tolist(), backend=nw.get_native_namespace(data))
