from __future__ import annotations

import random
from typing import TYPE_CHECKING

from pandas.api.types import is_string_dtype

from tab_err._utils import get_column

from ._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


class Mojibake(ErrorType):
    """Inserts mojibake into a column containing strings."""

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not is_string_dtype(series):
            msg = f"Column {column} does not contain values of the string dtype. Cannot insert Mojibake."
            raise TypeError(msg)

    def _get_valid_columns(self: Mojibake, data: pd.DataFrame) -> list[str | int]:
        """Returns all column names with string dtype elements."""
        return data.select_dtypes(include=["string", "object"]).columns.to_list()

    def _apply(self: Mojibake, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the Mojibake ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Returns:
            pd.Series: The data column, 'column', after Mojibake errors at the locations specified by 'error_mask' are introduced.
        """
        # Top 10 most used encodings on the internet
        # https://w3techs.com/technologies/overview/character_encoding
        top10 = {"utf_8", "iso-8859-1", "windows-1252", "windows-1251", "shift_jis", "euc_jp", "gb2312", "euc_kr", "windows-1250", "iso-8859-2"}

        # Some encodings are compatible with other encodings, which I remove here.
        encodings: dict[str, set[str]] = {
            "utf_8": top10 - {"utf_8"},
            "iso-8859-1": top10 - {"iso-8859-1", "windows-1252", "windows-1250", "iso-8859-2"},
            "windows-1252": top10 - {"windows-1252", "windows-1250", "iso-8859-1", "iso-8859-2"},
            "windows-1251": top10 - {"windows-1251"},
            "shift_jis": top10 - {"shift_jis"},
            "euc_jp": top10 - {"euc_jp"},
            "gb2312": top10 - {"gb2312"},
            "euc_kr": top10 - {"euc_kr"},
            "windows-1250": top10 - {"windows-1250", "iso-8859-1", "iso-8859-2", "windows-1252", "windows-1251"},
            "iso-8859-2": top10 - {"iso-8859-2", "windows-1250", "iso-8859-1", "windows-1252"},
        }

        series = get_column(data, column).copy()
        encoding_sender = self.config.encoding_sender
        encoding_receiver = self.config.encoding_receiver

        # TODO(nich): Check validity of supplied combo
        # TODO(nich): Choose valid combo if only one is None - don't ignore user choice
        if encoding_sender is None or encoding_receiver is None:
            encoding_sender = random.choice(list(top10))
            encoding_receiver = random.choice(list(encodings[encoding_sender]))

        series_mask = get_column(error_mask, column)
        series.loc[series_mask] = (
            series.loc[series_mask].apply(lambda x: x.encode(encoding_sender, errors="ignore")).apply(lambda x: x.decode(encoding_receiver, errors="ignore"))
        )
        return series
