from __future__ import annotations

import random
from typing import TYPE_CHECKING

from pandas.api.types import is_string_dtype

from tab_err.error_type import ErrorType
from tab_err.utils import get_column

if TYPE_CHECKING:
    import pandas as pd


class Butterfinger(ErrorType):
    """Inserts realistic typos into a column containing strings.

    Butterfinger imitates a typist who misses the correct key. For a given keyboard-layout and key, Butterfinger maps
    all keys that physically border the given key on the given layout. It assumes that all bordering keys are equally
    likely to be hit by the typist.

    Butterfinger assumes that words are separated by whitespaces. Applied to a cell, the period with which Butterfinger
    will corrupt words in that cell is controlled by the parameter `error_period`. By default, Butterfinger will insert
    a typo into every 10th word. Butterfinger will always insert at least one typo into an affected cell.
    """

    @staticmethod
    def _check_type(table: pd.DataFrame, column: int | str) -> None:
        series = get_column(table, column)

        if not is_string_dtype(series):
            msg = f"Column {column} does not contain values of the string dtype. Cannot apply Butterfingers."
            raise TypeError(msg)

    def _apply(self: Butterfinger, table: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Apply butterfinger.

        table: the pandas DataFrame to-be-corrupted
        error_mask: binary mask the marks the error positions
        column: column into which errors shall be inserted
        error_period: specifies how frequent butterfinger corruptions are - see class description for details.
        """
        series = get_column(table, column).copy()
        series_mask = get_column(error_mask, column)

        def butterfn(x: str) -> str:
            return butterfinger(x, self.config.error_period, self.config.keyboard_layout)

        series.loc[series_mask] = series.loc[series_mask].apply(butterfn)
        return series


def butterfinger(input_text: str, error_period: int = 10, layout: str = "ansi-qwerty") -> str:
    """Inserts realistic typos into a string.

    Butterfinger imitates a typist who misses the correct key. For a given keyboard-layout and key, Butterfinger maps
    all keys that physically border the given key on the given layout. It assumes that all bordering keys are equally
    likely to be hit by the typist.

    Butterfinger assumes that words are separated by whitespaces. It will corrupt words in the input text with a period
    controlled by the parameter `error_period`. By default, Butterfinger will insert a typo into every 10th word.
    Butterfinger will always insert at least one typo into the input text.

    Args:
        input_text: the string to be corrupted
        error_period: specifies how frequent butterfinger corruptions are - see class description for details.
        layout: the keyboard layout to be used for the corruption. Currently, only "ansi-qwerty" is supported.

    Returns:
        the corrupted string
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

    if error_period < 1:
        message = "error_period smaller than 1 is invalid, as multiple errors per word are not supported."
        raise ValueError(message)

    splits = input_text.split(" ")

    # draw only from splits that have a content
    valid_positions = [i for i, w in enumerate(splits) if len(w) > 0]
    n_draws = max(len(valid_positions) // error_period, 1)
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
