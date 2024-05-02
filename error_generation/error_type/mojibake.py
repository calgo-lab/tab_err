import random

import pandas as pd
from pandas.api.types import is_string_dtype

from error_generation.error_type import ErrorType
from error_generation.utils import Column, get_column


class Mojibake(ErrorType):
    """Inserts mojibake into a column containing strings."""

    @staticmethod
    def _check_type(table: pd.DataFrame, column: Column) -> None:
        series = get_column(table, column)

        if not is_string_dtype(series):
            msg = f"Column {column} does not contain values of the string dtype. Cannot insert Mojibake."
            raise TypeError(msg)

    @staticmethod
    def _apply(table: pd.DataFrame, error_mask: pd.DataFrame, column: Column) -> pd.Series:
        # Top 10 most used encodings on the internet
        # https://w3techs.com/technologies/overview/character_encoding
        encodings: list[str] = ["utf_8", "iso-8859-1", "windows-1252", "windows-1251", "shift_jis", "euc_jp", "gb2312", "euc_kr", "windows-1250", "iso-8859-2"]

        series = get_column(table, column).copy()
        encoding_sender, encoding_receiver = random.sample(encodings, 2)

        series_mask = get_column(error_mask, column)
        series.iloc[series_mask].apply(lambda x: x.encode(encoding_sender))
        series.iloc[series_mask].apply(lambda x: x.decode(encoding_receiver))
        return series
