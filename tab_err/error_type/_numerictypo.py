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
        """Returns column names with numeric dtype elements."""
        return data.select_dtypes(include=["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64"]).columns.to_list()

    def _apply(self: NumericTypo, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the NumericTypo ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.
        typo_error_period: specifies how frequent typo corruptions are - see class description for details.

        Returns:
            pd.Series: The data column, 'column', after NumericTypo errors at the locations specified by 'error_mask' are introduced.
        """
        series = get_column(data, column).copy().astype(str)
        original_type = get_column(data, column).dtype

        series_mask = get_column(error_mask, column)

        def butterfn(x: str) -> str:
            return numerictypo(x, self.config.numerictypo_keyboard_layout)

        series.loc[series_mask] = series.loc[series_mask].apply(butterfn)

        return series.astype(original_type)


def numerictypo(value: str, layout: str = "numpad") -> str:
    """Inserts realistic numeric typos into string representations of numeric values.

    Typo imitates a typist who misses the correct key. For a given keyboard-layout and key, Typo maps
    all keys that physically border the given key on the given layout. It assumes that all bordering keys are equally
    likely to be hit by the typist.

    Args:
        value (str): the string value to be corrupted
        layout (str): the keyboard layout to be used for the corruption.
                      Currently, "numpad" and "number_pad" are supported for numeric typos, defaults to "numpad"

    Returns:
        str: The corrupted string value.
    """
    if layout == "numpad":
        neighbors = {
            "1": "024",
            "2": "0135",
            "3": "26",
            "4": "157",
            "5": "2468",
            "6": "359",
            "7": "48",
            "8": "759",
            "9": "68",
            "0": "12",
        }
    elif layout == "number_bar":
        neighbors = {
            "1": "2",
            "2": "13",
            "3": "24",
            "4": "35",
            "5": "46",
            "6": "57",
            "7": "68",
            "8": "79",
            "9": "80",
            "0": "9",
        }
    else:
        message = f'Unsupported keyboard layout "{layout}".'
        raise ValueError(message)

    if value == "":  # return random char if empty string
        return random.choice(list(neighbors.keys()))

    char_position = random.choice(list(range(len(value))))

    new_choice = True
    new_char = ""
    while new_choice:
        new_choice = False  # A valid replacement was determined

    return "".join([x if i != char_position else new_char for i, x in enumerate(value)])
