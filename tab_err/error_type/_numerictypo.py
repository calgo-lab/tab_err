from __future__ import annotations

import random
from typing import TYPE_CHECKING

from pandas.api.types import is_numeric_dtype

from tab_err._utils import get_column

from ._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


class NumericTypo(ErrorType):
    """Inserts realistic numeric typos into a column containing numeric values.

    NumericTypo imitates a typist who misses the correct key. For a given keyboard-layout and key, Typo maps
    all keys that physically border the given key on the given layout. It assumes that all bordering keys are equally
    likely to be hit by the typist. It considers the number keys of a number pad, including comma.

    NumericTypo will always insert at least one typo into an affected cell.
    """

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not is_numeric_dtype(series):
            msg = f"Column {column} is not of a numeric dtype. Cannot apply numeric typos."
            raise TypeError(msg)

    def _get_valid_columns(self: NumericTypo, data: pd.DataFrame) -> list[str | int]:
        """Returns column names with string dtype elements."""
        return data.select_dtypes(include=["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64"]).columns.to_list()

    def _apply(self: NumericTypo, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the NumericTypo ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.
        typo_error_period: specifies how frequent typo corruptions are - see class description for details.

        Returns:
            pd.Series: The data column, 'column', after NumericTypo errors at the locations specified by 'error_mask' are introduced. Series are converted to dtype float.
        """
        series = get_column(data, column).copy().astype(str)
        series_mask = get_column(error_mask, column)

        def butterfn(x: str) -> str:
            return numerictypo(x, self.config.numerictypo_keyboard_layout)

        series.loc[series_mask] = series.loc[series_mask].apply(butterfn)

        series = series.astype(float)

        return series


def numerictypo(value: str, layout: str = "numpad") -> str:
    """Inserts realistic numerictypos into string representations of numeric values.

    Typo imitates a typist who misses the correct key. For a given keyboard-layout and key, Typo maps
    all keys that physically border the given key on the given layout. It assumes that all bordering keys are equally
    likely to be hit by the typist.

    Args:
        value (str): the string value to be corrupted
        layout (str): the keyboard layout to be used for the corruption. Currently, only "numpad" is supported for numeric typos.

    Returns:
        str: The corrupted string value.
    """
    if layout == "numpad":
        neighbors = {
            "1": "024",
            "2": "0135",
            "3": ".26",
            "4": "157",
            "5": "2468",
            "6": "359",
            "7": "48",
            "8": "759",
            "9": "68",
            "0": ".12",
            ".": "03",
        }
    else:
        message = f"Unsupported keyboard layout {layout}."
        raise ValueError(message)

    if value == "":  # return random char if empty string
        return random.choice(list(neighbors.keys()))


    char_position = random.choice(list(range(len(value))))
    char = value[char_position]

    already_contains_comma = "." in value

    new_choice = True
    new_char = ""
    while new_choice:
        new_char = random.choice(neighbors.get(char, [char]))
        if already_contains_comma and new_char == ".":
            # If the numeric value already contains a comma and comma was chosen as replacement,
            # redraw to get another replacement
            pass
        else:
            new_choice = False # A valid replacement was determined

    new_value = "".join([x if i != char_position else new_char for i, x in enumerate(value)])

    return new_value
