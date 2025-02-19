from __future__ import annotations

from typing import TYPE_CHECKING

from tab_err._utils import get_column

from ._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


class Extraneous(ErrorType):
    """Adds Extraneous strings around the values in a column."""

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        # all data types are fine
        pass

    def _get_valid_columns(self:Extraneous, data: pd.DataFrame) -> list[str | int]:
        """Returns all column names with string dtype elements. Necessary for high level API."""
        return data.select_dtypes(include=["string", "object"]).columns.to_list()

    def _apply(self: Extraneous, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the Extraneous ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            ValueError: If extraneous_value_template is not configured, a ValueError will be thrown.
            ValueError: If extraneous_value_template does not contain the placeholder value, a ValueError will be thrown.

        Returns:
            pd.Series: The data column, 'column', after Extraneous errors at the locations specified by 'error_mask' are introduced.
        """
        series = get_column(data, column).copy()
        series_mask = get_column(error_mask, column)

        if self.config.extraneous_value_template is None:
            msg = "No extraneous_value_template has been configured. Please add it to the ErrorTypeConfig."
            raise ValueError(msg)
        if "{value}" not in self.config.extraneous_value_template:
            msg = f"The extraneous template {self.config.extraneous_value_template} does not contain the placeholder "
            msg += "{value}. Please add it for a valid format."
            raise ValueError(msg)

        series.loc[series_mask] = series.loc[series_mask].apply(lambda x: self.config.extraneous_value_template.format(value=x))
        return series
