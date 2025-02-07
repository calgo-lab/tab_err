from __future__ import annotations

from typing import TYPE_CHECKING

from pandas.api.types import is_numeric_dtype

from tab_err._utils import get_column

from ._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


class WrongUnit(ErrorType):
    """Simulate a column containing values that are scaled because they are not stored in the same unit."""

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not is_numeric_dtype(series):
            msg = f"Column {column} does not contain scalars. Cannot apply a wrong unit."
            raise TypeError(msg)

    def _apply(self: WrongUnit, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the WrongUnit ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            ValueError: If wrong_unit_scaling is not defined in config, a ValueError will be thrown.

        Returns:
            pd.Series: The data column, 'column', after Replace errors at the locations specified by 'error_mask' are introduced.
        """
        if self.config.wrong_unit_scaling is None:
            msg = f"Cannot apply wrong unit to column {column} because no scaling function wrong_unit_scaling was defined in the ErrorTypeConfig."
            raise ValueError(msg)

        series = get_column(data, column).astype("object").copy()
        series_mask = get_column(error_mask, column)

        series.loc[series_mask] = series.loc[series_mask].apply(self.config.wrong_unit_scaling)
        return series
