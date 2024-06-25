from __future__ import annotations

import random
from typing import TYPE_CHECKING, Callable

from pandas.api.types import is_string_dtype

from error_generation.error_type import ErrorType
from error_generation.utils import get_column

if TYPE_CHECKING:
    import pandas as pd


def fixed_shuffle_pattern(format_len: int, permutation_separator: str) -> Callable:
    """Returns a function that shuffles the values in a column following a fixed pattern."""
    initial_pattern = list(range(format_len + 1))  # list that indicates the positions of each value
    new_pattern = initial_pattern

    while initial_pattern == new_pattern:
        new_pattern = random.sample(initial_pattern, len(initial_pattern))

    def shuffle_pattern(old_string: str) -> str:
        old_list = old_string.split(permutation_separator)
        new = ["" for _ in range(len(old_list))]
        for i, n in zip(initial_pattern, new_pattern):
            new[n] = old_list[i]
        return permutation_separator.join(new)

    return shuffle_pattern


class Permutate(ErrorType):
    """Permutates the values in a column."""

    @staticmethod
    def _check_type(table: pd.DataFrame, column: int | str) -> None:
        series = get_column(table, column)

        if not is_string_dtype(series):
            msg = f"Column {column} does not contain values of the string dtype. Cannot Permutate values."
            raise TypeError(msg)

    def _apply(self: Permutate, table: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        series = get_column(table, column).copy()
        series_mask = get_column(error_mask, column)

        separator_counts = [x.count(self.config.permutation_separator) for x in series.dropna()]
        for i, count in enumerate(separator_counts):
            if count == 0:
                msg = f'Cannot permutate values, because column {column} contains value "{series[i]}" that is not separated by the separator '
                msg += f'"{self.config.permutation_separator}". To use another separator, define it in the ErrorTypeconfig.'
                raise ValueError(msg)

        if self.config.permutation_pattern == "fixed":
            if len(set(separator_counts)) > 1:
                msg = f"Column {column} cannot be permutated using a fixed permutation_pattern: A fixed permutation_pattern requires all values "
                msg += "to be formatted in the same way."
                raise ValueError(msg)
            shuffle_pattern = fixed_shuffle_pattern(separator_counts[0], self.config.permutation_separator)

        if self.config.permutation_pattern == "random":

            def shuffle_pattern(old_string: str) -> str:
                old_list = old_string.split(self.config.permutation_separator)
                new = old_list
                while new == old_list:
                    new = random.sample(old_list, len(old_list))
                return self.config.permutation_separator.join(new)

        series.loc[series_mask] = series.loc[series_mask].apply(shuffle_pattern)
        return series
