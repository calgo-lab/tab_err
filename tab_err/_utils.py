from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def set_column(table: pd.DataFrame, column: int | str, series: pd.Series) -> None:
    """Replaces a column in the given DataFrame with the given Series.

    Mutates table and changes the dtype of the original table to that of the series,
    which, depending on the error type, might change.
    """
    col = table.columns[column] if isinstance(column, int) else column
    table[col] = table[col].astype(series.dtype)
    table[col] = series


def get_column_str(table: pd.DataFrame, column: int | str) -> str:
    """Return column's name of the given DataFrame, where column can be defined as name or index."""
    if isinstance(column, int):
        col = table.columns[column]
    elif isinstance(column, str):
        col = column
    else:
        msg = f"Column must be an int or str, not {type(column)}"
        raise TypeError(msg)

    return col


def get_column(table: pd.DataFrame, column: int | str) -> pd.Series:
    """Selects a column from the given DataFrame and returns it as a Series."""
    return table[get_column_str(table, column)]
