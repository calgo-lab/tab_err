from __future__ import annotations

import random

import pandas as pd

from tab_err._utils import get_column

from ._error_type import ErrorType


class CategorySwap(ErrorType):
    """Simulate incorrect labels in a column that contains categorical values."""

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        """Checks that the data type is Categorical and the number of categories is at least two in the column to be modified.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            TypeError: If the column does not contain Categorical dtype values, a TypeError is thrown.
            ValueError: If there are less than two categories in the column, a ValueError will be thrown.
        """
        series = get_column(data, column)

        if not isinstance(series.dtype, pd.CategoricalDtype):
            msg = f"Column {column} does not contain values of the Categorical dtype. Cannot insert Mislables.\n"
            msg += "Try casting the column to CategoricalDtype using df[column].astype('category')."
            raise TypeError(msg)

        if len(series.cat.categories) <= 1:
            msg = f"Column {column} contains {len(series.cat.categories)} categories. Require at least 2 categories to insert mislabels."
            raise ValueError(msg)

    def _get_valid_columns(self: CategorySwap, data: pd.DataFrame) -> list[str | int]:
        """Checks which columns are categorical and returns the indices of those with two or more categories."""
        valid_columns = []
        for col_name in data.columns:
            series = get_column(data, col_name)

            if isinstance(series.dtype, pd.CategoricalDtype) and len(series.cat.categories) > 1:
                valid_columns.append(col_name)

        return valid_columns

    def _apply(self: CategorySwap, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the CategorySwap ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            ValueError: If the value for parameter 'config.mislabel_weighing' is invalid (not 'uniform' or 'frequency'), a ValueError will be thrown.

        Returns:
            pd.Series: The data column, 'column', after CategorySwap errors at the locations specified by 'error_mask' are introduced.
        """
        series = get_column(data, column).copy()

        if self.config.mislabel_weighing == "uniform":

            def sample_label(old_label: pd.Any) -> pd.Any:
                choices = [x for x in series.cat.categories.to_numpy() if x != old_label]
                return random.choice(choices)

        elif self.config.mislabel_weighing == "frequency":

            def sample_label(old_label: pd.Any) -> pd.Any:
                se_sample = series.loc[series != old_label]
                return se_sample.sample(1, replace=True).to_numpy()[0]
        else:
            msg = "Invalid value for parameter 'config.mislabel_weighing'. Allowed values are: 'uniform', 'frequency'."
            raise ValueError(msg)

        series_mask = get_column(error_mask, column)
        series.loc[series_mask] = series.loc[series_mask].apply(sample_label)
        return series
