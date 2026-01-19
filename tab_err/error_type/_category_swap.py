from __future__ import annotations

import random

import narwhals as nw

from tab_err._utils import get_column, new_series_like

from ._error_type import ErrorType


class CategorySwap(ErrorType):
    """Simulate incorrect labels in a column that contains categorical values."""

    @staticmethod
    def _check_type(data: nw.DataFrame, column: int | str) -> None:
        """Checks that the data type is Categorical and the number of categories is at least two in the column to be modified.

        Args:
            data (nw.DataFrame): DataFrame containing the column to add errors to.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            TypeError: If the column does not contain Categorical dtype values, a TypeError is thrown.
            ValueError: If there are less than two categories in the column, a ValueError will be thrown.
        """
        series = get_column(data, column)

        if series.dtype != nw.Categorical:
            msg = f"Column {column} does not contain values of the Categorical dtype. Cannot insert Mislabels.\n"
            msg += "Try casting the column to Categorical dtype."
            raise TypeError(msg)

        # Get unique values to check number of categories
        unique_vals = series.unique()
        if len(unique_vals) <= 1:
            msg = f"Column {column} contains {len(unique_vals)} categories. Require at least 2 categories to insert mislabels."
            raise ValueError(msg)

    def _get_valid_columns(self: CategorySwap, data: nw.DataFrame) -> list[str | int]:
        """Checks which columns are categorical and returns the indices of those with two or more categories."""
        valid_columns: list[str | int] = []
        for col_name in data.columns:
            series = get_column(data, col_name)

            if series.dtype == nw.Categorical:
                unique_vals = series.unique()
                if len(unique_vals) > 1:
                    valid_columns.append(col_name)

        return valid_columns

    def _apply(self: CategorySwap, data: nw.DataFrame, error_mask: nw.DataFrame, column: int | str) -> nw.Series:
        """Applies the CategorySwap ErrorType to a column of data.

        Args:
            data (nw.DataFrame): DataFrame containing the column to add errors to.
            error_mask (nw.DataFrame): A DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            ValueError: If the value for parameter 'config.mislabel_weighing' is invalid (not 'uniform' or 'frequency'), a ValueError will be thrown.

        Returns:
            nw.Series: The data column, 'column', after CategorySwap errors at the locations specified by 'error_mask' are introduced.
        """
        series = get_column(data, column)
        series_mask = get_column(error_mask, column)

        # Get numpy arrays
        data_arr = series.to_numpy().copy()
        mask_arr = series_mask.to_numpy()

        # Get categories
        categories = series.unique().to_numpy()

        if self.config.mislabel_weighing == "uniform":

            def sample_label(old_label: str) -> str:
                choices = [x for x in categories if x != old_label]
                return random.choice(choices)

        elif self.config.mislabel_weighing == "frequency":
            # Calculate frequency weights
            value_counts: dict[str, int] = {}
            for val in data_arr:
                if val not in value_counts:
                    value_counts[val] = 0
                value_counts[val] += 1

            def sample_label(old_label: str) -> str:
                choices = [x for x in categories if x != old_label]
                weights: list[float] = [float(value_counts.get(x, 1)) for x in choices]
                total = float(sum(weights))
                weights = [w / total for w in weights]
                return random.choices(choices, weights=weights, k=1)[0]
        else:
            msg = "Invalid value for parameter 'config.mislabel_weighing'. Allowed values are: 'uniform', 'frequency'."
            raise ValueError(msg)

        # Apply mislabeling where mask is True
        for i in range(len(data_arr)):
            if mask_arr[i]:
                data_arr[i] = sample_label(data_arr[i])

        return new_series_like(data, column, data_arr)
