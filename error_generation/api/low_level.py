import math
import random
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Column:

    """Describe a column in a Dataframe."""

    name: str = field(default=None)
    index: int = field(default=None)

    def __post_init__(self):
        if self.name is None and self.index is None:
            msg = "Specify either column name or index."
            raise ValueError(msg)


class Mechanism:
    pass


class ErrorType:
    pass


def error_function(x):
    return x


def create_errors(
    table: pd.DataFrame, column: Column, error_rate: float, mechanism: Mechanism, error_type: ErrorType, condition_to_column: Column | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        series = table.loc[column.name]
    except KeyError:  # Assume it's integer index
        series = table.iloc[column.index]
    except IndexError:
        msg = f"Invalid column: {column}"
        raise ValueError(msg) from None
    n_rows = len(series)

    # TODO: this should be its own function
    print(f"And use {mechanism} and {error_type} and {condition_to_column} to infer " "error positions.")
    n_errors = math.floor(n_rows * error_rate)

    error_rows = random.sample(n_rows, n_errors)
    series.apply()
    error_series = series.iloc[error_rows].apply(error_function)
    mask = [0 if (i not in error_rows) else 1 for i in range(n_rows)]
    return error_series, pd.Series(mask)
