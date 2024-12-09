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

    def _apply(self: Extraneous, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        # cast to object because our operation potentially changes the type of a column.
        series = get_column(data, column).copy().astype("object")
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
