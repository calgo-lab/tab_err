from __future__ import annotations

import random
from typing import TYPE_CHECKING

from pandas.api.types import is_string_dtype

from tab_err._utils import get_column

from ._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


class Typo(ErrorType):
    """Inserts realistic typos into a column containing strings.

    Typo imitates a typist who misses the correct key. For a given keyboard-layout and key, Typo maps
    all keys that physically border the given key on the given layout. It assumes that all bordering keys are equally
    likely to be hit by the typist.

    Typo assumes that words are separated by whitespaces. Applied to a cell, the period with which Typo
    will corrupt words in that cell is controlled by the parameter `typo_error_period`. By default, Typo will insert
    a typo into every 10th word. Typo will always insert at least one typo into an affected cell.
    """

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not is_string_dtype(series):
            msg = f"Column {column} does not contain values of the string dtype. Cannot apply Typos."
            raise TypeError(msg)

    def _get_valid_columns(self: Typo, data: pd.DataFrame) -> list[str | int]:
        """Returns column names with string dtype elements."""
        return data.select_dtypes(include=["string", "object"]).columns.to_list()

    def _apply(self: Typo, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the Typo ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.
        typo_error_period: specifies how frequent typo corruptions are - see class description for details.

        Returns:
            pd.Series: The data column, 'column', after Typo errors at the locations specified by 'error_mask' are introduced.
        """
        series = get_column(data, column).copy()
        series_mask = get_column(error_mask, column)

        def butterfn(x: str) -> str:
            return typo(x, self.config.typo_error_period, self.config.typo_keyboard_layout)

        series.loc[series_mask] = series.loc[series_mask].apply(butterfn)
        return series


def typo(input_text: str, typo_error_period: int = 10, layout: str = "ansi-qwerty") -> str:
    """Inserts realistic typos into a string.

    Typo imitates a typist who misses the correct key. For a given keyboard-layout and key, Typo maps
    all keys that physically border the given key on the given layout. It assumes that all bordering keys are equally
    likely to be hit by the typist.

    Typo assumes that words are separated by whitespaces. It will corrupt words in the input text with a period
    controlled by the parameter `typo_error_period`. By default, Typo will insert a typo into every 10th word.
    Typo will always insert at least one typo into the input text.

    Args:
        input_text (str): the string to be corrupted
        typo_error_period (int, optional): specifies how frequent typo corruptions are - see class description for details. Defaults to '10'.
        layout (str): the keyboard layout to be used for the corruption. Currently, only "ansi-qwerty" is supported. Defaults to 'ansi-qwerty'

    Returns:
        str: The corrupted string.
    """
    if layout == "ansi-qwerty":
        neighbors = {
            "q": "12wa",
            "w": "q23esa",
            "e": "34rdsw",
            "r": "e45tfd",
            "t": "56ygfr",
            "y": "t67uhg",
            "u": "y78ijh",
            "i": "u89okj",
            "o": "i90plk",
            "p": "o0-[;l",
            "a": "qwsz",
            "s": "awedxz",
            "d": "serfcx",
            "f": "drtgvc",
            "g": "ftyhbv",
            "h": "gyujnb",
            "j": "huikmn",
            "k": "jiol,m",
            "l": "kop;.,",
            "z": "asx",
            "x": "sdcz",
            "c": "dfvx",
            "v": "cfgb",
            "b": "vghn",
            "n": "bhjm",
            "m": "njk,",
            "1": "2q`",
            "2": "13wq",
            "3": "24ew",
            "4": "35re",
            "5": "46tr",
            "6": "57yt",
            "7": "68uy",
            "8": "79iu",
            "9": "80oi",
            "0": "9-po",
            "-": "0=[p",
            "=": "-][",
            "[": "-=]';p",
            "]": "[=\\'",
            ";": "lp['/.",
            "'": ";[]/",
            ",": "mkl.",
            ".": ",l;/",
            "/": ".;'",
            "\\": "]",
        }
    else:
        message = f"Unsupported keyboard layout {layout}."
        raise ValueError(message)

    if typo_error_period < 1:
        message = "typo_error_period smaller than 1 is invalid, as multiple errors per word are not supported."
        raise ValueError(message)

    splits = input_text.split(" ")

    # draw only from splits that have a content
    valid_positions = [i for i, w in enumerate(splits) if len(w) > 0]
    n_draws = max(len(valid_positions) // typo_error_period, 1)
    positions = random.sample(valid_positions, n_draws)

    for p in positions:
        word = splits[p]  # select the to-be-corrupted word
        char_position = random.choice(list(range(len(word))))
        char = word[char_position]
        is_upper = char.isupper()

        new_char = random.choice(neighbors.get(char.lower(), [char.lower()]))

        new_char = new_char.upper() if is_upper else new_char
        new_word = "".join([x if i != char_position else new_char for i, x in enumerate(word)])
        splits[p] = new_word

    return " ".join(splits)
