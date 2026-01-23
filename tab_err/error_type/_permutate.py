from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import narwhals as nw
    import numpy as np

from tab_err._utils import get_column, is_string_dtype, new_series_like, select_string_columns

from ._error_type import ErrorType


def _generate_shuffle_pattern(format_len: int) -> list[int]:
    """Generates a list of integers that indicates the positions of each value in a formatted string."""
    initial_pattern = list(range(format_len + 1))  # list that indicates the positions of each value
    new_pattern = initial_pattern

    while initial_pattern == new_pattern:  # Ensure the sample is different from original
        new_pattern = random.sample(initial_pattern, len(initial_pattern))

    return new_pattern


def _check_column_format_consistency(separator_counts: list[int], column: int | str) -> None:
    """Checks that each string in the column has the same number of separators, throws a ValueError if not."""
    if len(set(separator_counts)) > 1:
        msg = f"Column '{column}' cannot be permutated using a fixed permutation_automation_pattern: A fixed permutation_automation_pattern requires "
        msg += "all values to be formatted in the same way."
        raise ValueError(msg)


class Permutate(ErrorType):
    """Permutates the parts of a compound value in a column."""

    @staticmethod
    def _check_type(data: nw.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not is_string_dtype(series):
            msg = f"Column {column} does not contain values of the string dtype. Cannot Permutate values."
            raise TypeError(msg)

    def _get_valid_columns(self: Permutate, data: nw.DataFrame) -> list[str | int]:
        """Returns column names with string dtype elements."""
        return select_string_columns(data)

    def _random_pattern_function(self: Permutate, old_string: str) -> str:
        """Generates a random permutation of `old_string` elements split on the `permutation_separator`."""
        old_list = old_string.split(self.config.permutation_separator)
        new = old_list
        while new == old_list:  # Ensure the sample is different from original
            new = random.sample(old_list, len(old_list))

        return self.config.permutation_separator.join(new)

    def _fixed_pattern_function(self: Permutate, old_string: str, new_pattern: list[int]) -> str:
        """Generates a permutation of `old_string` elements split on the `permutation_separator` given a permutation specified by `new_pattern`."""
        string_as_part_lists = old_string.split(self.config.permutation_separator)
        new_string_as_part_list = [string_as_part_lists[index] for index in new_pattern]

        return self.config.permutation_separator.join(new_string_as_part_list)

    def _pattern_permutation(self, permutation_pattern: list[int], data_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
        """Permutates values of a series data_arr as specified in permutation_pattern."""
        for i in range(len(data_arr)):
            if mask_arr[i]:
                val = data_arr[i]
                if val is not None and isinstance(val, str):
                    data_arr[i] = self._fixed_pattern_function(val, permutation_pattern)
        return data_arr

    def _random_permutation(self, data_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
        """Permutates values of a series data_arr using a new, random pattern for each value."""
        for i in range(len(data_arr)):
            if mask_arr[i]:
                val = data_arr[i]
                if val is not None and isinstance(val, str):
                    data_arr[i] = self._random_pattern_function(val)

        return data_arr

    def _apply(self: Permutate, data: nw.DataFrame, error_mask: nw.DataFrame, column: int | str) -> nw.Series:
        """Applies the `Permutate` `ErrorType` to a column of data.

        Args:
            data (nw.DataFrame): DataFrame containing the column to add errors to.
            error_mask (nw.DataFrame): A DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            ValueError: If the column contains values not supported by the separator, a ValueError will be thrown.
            ValueError: If a fixed_permutation_pattern is selected and all values are not formatted the same way, a ValueError will be thrown.

        Returns:
            nw.Series: The data column, 'column', after Permutate errors at the locations specified by 'error_mask' are introduced.
        """
        series = get_column(data, column)
        series_mask = get_column(error_mask, column)

        # Get numpy arrays
        data_arr = series.to_numpy().copy()
        mask_arr = series_mask.to_numpy()

        # Get separator counts for non-null values
        separator_counts = [
            val.count(self.config.permutation_separator)
            for val in data_arr
            if val is not None and isinstance(val, str)
        ]

        for i, count in enumerate(separator_counts):
            if count == 0:
                msg = f'Cannot permutate values, because column {column} contains value "{data_arr[i]}" that is not separated by the separator '
                msg += f'"{self.config.permutation_separator}". To use another separator, define it in the ErrorTypeConfig.'
                raise ValueError(msg)

        if self.config.permutation_pattern is not None:  # Permutation of each entry from pattern.
            _check_column_format_consistency(separator_counts, column)
            data_arr = self._pattern_permutation(self.config.permutation_pattern, data_arr, mask_arr)

        elif self.config.permutation_automation_pattern == "fixed":  # Fixed permutation -- random once, applied to all.
            _check_column_format_consistency(separator_counts, column)
            rnd_permutatin_pattern = _generate_shuffle_pattern(separator_counts[0])

            data_arr = self._pattern_permutation(rnd_permutatin_pattern, data_arr, mask_arr)

        else:  # Random permutation -- random for each entry.
            data_arr = self._random_permutation(data_arr, mask_arr)

        return new_series_like(data, column, data_arr)
