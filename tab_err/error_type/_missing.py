from __future__ import annotations

from typing import TYPE_CHECKING, Union, cast

from tab_err._utils import get_column, is_string_dtype, new_series_like, select_string_columns

from ._error_type import ErrorType

if TYPE_CHECKING:
    import narwhals as nw

class MissingValue(ErrorType):
    """Insert missing values into a column.

    Missing value handling varies across DataFrame libraries. This implementation
    inserts None/null values which will be handled appropriately by the underlying library.
    """

    @staticmethod
    def _check_type(data: nw.DataFrame, column: int | str) -> None:
        # all dtypes are supported
        pass

    def _get_valid_columns(self: MissingValue, data: nw.DataFrame) -> list[str | int]:
        """If the config missing value is None, returns all columns. Otherwise, only the columns with string type."""
        if self.config.missing_value is None:
            return cast("list[Union[str, int]]", list(data.columns))
        return select_string_columns(data)

    def _apply(self: MissingValue, data: nw.DataFrame, error_mask: nw.DataFrame, column: int | str) -> nw.Series:
        """Applies the MissingValue ErrorType to a column of data.

        Args:
            data (nw.DataFrame): DataFrame containing the column to add errors to.
            error_mask (nw.DataFrame): A DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Returns:
            nw.Series: The data column, 'column', after MissingValue errors at the locations specified by 'error_mask' are introduced.
        """
        series = get_column(data, column)
        series_mask = get_column(error_mask, column)

        # Get numpy arrays for manipulation
        data_arr = series.to_numpy().copy()
        mask_arr = series_mask.to_numpy()

        # Set values to missing_value or None where mask is True
        if is_string_dtype(series) and self.config.missing_value is None:
            # For string columns, convert to object array to allow None
            data_arr = data_arr.astype(object)
            data_arr[mask_arr] = None
        else:
            # For other types, use the configured missing value or None
            missing_val = self.config.missing_value
            if missing_val is None:
                # Convert to object to allow None
                data_arr = data_arr.astype(object)
            data_arr[mask_arr] = missing_val

        # Create new series with the modified data
        return new_series_like(data, column, data_arr)
