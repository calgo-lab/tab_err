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

    def _get_valid_columns(self:AddDelta, data: pd.DataFrame, preserve_dtypes = True) -> list[str | int]:
        """Returns all column names since all dtypes are supported."""
        return data.columns.tolist()

    def _apply(self: AddDelta, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the AddDelta ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            ValueError: If the add_delta_value is None, a ValueError will be thrown.

        Returns:
            pd.Series: The data column, 'column', after AddDelta errors at the locations specified by 'error_mask' are introduced.
        """
        # cast to object because our operation potentially changes the type of a column.
        series = get_column(data, column).copy().astype("object")
        series_mask = get_column(error_mask, column)

        if self.config.add_delta_value is None:
            msg = "No add_delta_value has been configured. Please add it to the ErrorTypeConfig."
            raise ValueError(msg)

        series.loc[series_mask] = series.loc[series_mask].apply(lambda x: x + self.config.add_delta_value)
        return series
