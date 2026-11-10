from __future__ import annotations

import random
from typing import TYPE_CHECKING

from pandas.api.types import is_string_dtype

from tab_err._utils import get_column

from ._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


def _validate_repeat_bounds(min_repeats: int, max_repeats: int) -> None:
    if not isinstance(min_repeats, int) or not isinstance(max_repeats, int):
        msg = "min_repeats and max_repeats must be integers."
        raise TypeError(msg)

    if min_repeats < 1 or max_repeats < 1:
        msg = "min_repeats and max_repeats must be at least 1."
        raise ValueError(msg)

    if min_repeats > max_repeats:
        msg = "min_repeats must not be greater than max_repeats."
        raise ValueError(msg)


class StickyKey(ErrorType):
    """Repeats a randomly selected character in strings.

    The number of additional copies is selected uniformly from the inclusive range configured by
    ``sticky_key_min_repeats`` and ``sticky_key_max_repeats``.
    """

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not is_string_dtype(series):
            msg = f"Column {column} does not contain values of the string dtype. Cannot apply StickyKey."
            raise TypeError(msg)

    def _get_valid_columns(self: StickyKey, data: pd.DataFrame) -> list[str | int]:
        """Returns column names with string dtype elements."""
        return data.select_dtypes(include=["string", "object"]).columns.to_list()

    def _apply(self: StickyKey, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies StickyKey to the selected cells in a column."""
        min_repeats = self.config.sticky_key_min_repeats
        max_repeats = self.config.sticky_key_max_repeats
        _validate_repeat_bounds(min_repeats, max_repeats)

        series = get_column(data, column).copy()
        series_mask = get_column(error_mask, column)
        series.loc[series_mask] = series.loc[series_mask].apply(lambda value: sticky_key(value, min_repeats, max_repeats))
        return series


def sticky_key(input_text: str, min_repeats: int = 1, max_repeats: int = 5) -> str:
    """Repeats one randomly selected character in a string.

    ``min_repeats`` and ``max_repeats`` specify the inclusive range for the number of additional
    copies. Empty strings are returned unchanged.

    Args:
        input_text: The string to modify.
        min_repeats: Minimum number of additional character copies. Defaults to 1.
        max_repeats: Maximum number of additional character copies. Defaults to 5.

    Returns:
        The string with the selected character repeated next to its original position.

    Raises:
        TypeError: If either repeat bound is not an integer.
        ValueError: If a repeat bound is less than 1 or the minimum exceeds the maximum.
    """
    _validate_repeat_bounds(min_repeats, max_repeats)

    if input_text == "":
        return input_text

    char_position = random.randrange(len(input_text))
    repeats = random.randint(min_repeats, max_repeats)
    character = input_text[char_position]
    return input_text[:char_position] + character * (repeats + 1) + input_text[char_position + 1 :]
