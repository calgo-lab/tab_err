from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class ErrorMechanismConfig:
    """Parameters that describe the error distribution, which we call the mechanism.

    Args:
        error_rate: The rate at which the error occurs.
        error_type: The type of error that is generated.
        condition_to_column: The column that determines whether the error is generated.
    """

    error_rate: float
    condition_to_column: int | str | None = None


@dataclass
class ErrorTypeConfig:
    """Parameters that describe the error type.

    Arguments that are specific to the error type. Most error types do not share the same arguments, which
    is why there are many attributes of this dataclass that are mostly default values.

    Args:
        encoding_sender: When creating Mojibake, used to encode strings to bytes.
        encoding_receiver: When creating Mojibake, used to decode bytes back to strings.
        keyboard_layout: When using Butterfinger, the keyboard layout used by the typer.
        error_period: When using Butterfinger, the period at which the error occurs.
    """

    encoding_sender: str | None = None
    encoding_receiver: str | None = None

    keyboard_layout: str = "ansi-qwerty"
    error_period: int = 10


def get_column(table: pd.DataFrame, column: int | str) -> pd.Series:
    """Selects a column from a dataframe and returns it as a series."""
    if isinstance(column, int):
        series = table.iloc[:, column]
    elif isinstance(column, str):
        series = table.loc[:, column]
    else:
        msg = f"Column must be an int or str, not {type(column)}"
        raise TypeError(msg)
    return series


def set_column(table: pd.DataFrame, column: int | str, series: pd.Series) -> pd.Series:
    """Replaces a column in a dataframe with a series. Mutates table."""
    if isinstance(column, int):
        table.iloc[:, column] = series
    elif isinstance(column, str):
        table.loc[:, column] = series
    return table
