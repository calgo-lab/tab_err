from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class Column:
    """Describe a column in a Dataframe. We support selection by both index and column names."""

    name: str | None = field(default=None)
    index: int | None = field(default=None)

    def __post_init__(self: Column) -> None:
        """Ensures that either column name, or an index is set."""
        if self.name is None and self.index is None:
            msg = "Specify either column name or index."
            raise ValueError(msg)


def get_column(table: pd.DataFrame, column: Column) -> pd.Series:
    """Selects a column from a dataframe and returns it as a series."""
    try:
        return table.loc[column.name]
    except KeyError:  # Assume it's integer index
        return table.iloc[column.index]
    except IndexError:
        msg = f"Invalid column: {column}"
        raise ValueError(msg) from None


def set_column(table: pd.DataFrame, column: Column, series: pd.Series) -> pd.Series:
    """Replaces a column in a dataframe with a series. Mutates table."""
    try:
        table.loc[column.name] = series
    except KeyError:  # Assume it's integer index
        table.iloc[column.index] = series
    except IndexError:
        msg = f"Invalid column: {column}"
        raise ValueError(msg) from None
    return table
