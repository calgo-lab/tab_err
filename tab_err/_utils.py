from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def set_column(data: pd.DataFrame, column: int | str, series: pd.Series) -> None:
    """Replaces a column in the given DataFrame with the given Series.

    Mutates data and changes the dtype of the original data to that of the series,
    which, depending on the error type, might change.
    """
    col = data.columns[column] if isinstance(column, int) else column
    data[col] = data[col].astype(series.dtype)
    data[col] = series


def get_column_str(data: pd.DataFrame, column: int | str) -> str:
    """Return column's name of the given DataFrame, where column can be defined as name or index."""
    if isinstance(column, int):
        col = data.columns[column]
    elif isinstance(column, str):
        col = column
    else:
        msg = f"Column must be an int or str, not {type(column)}"
        raise TypeError(msg)

    return col


def get_column(data: pd.DataFrame, column: int | str) -> pd.Series:
    """Selects a column from the given DataFrame and returns it as a Series."""
    return data[get_column_str(data, column)]


def seed_randomness(seed: int | None) -> np.random.Generator:
    if seed is not None:
        random.seed(seed)
        random_generator = np.random.default_rng(seed=seed)

    else:
        random_generator = np.random.default_rng()

    return random_generator


def check_error_rate(error_rate: float) -> None:
    """Check that the error rate falls in the valid range, raise a ValueError otherwise."""
    if error_rate < 0.0 or error_rate > 1.0:
        msg = f"The error rate is {error_rate} but must be between 0 and 1"
        raise ValueError(msg)


def check_data_emptiness(data: pd.DataFrame) -> None:
    """Check that the dataset is not empty, raise a ValueError otherwise."""
    if data.empty:
        msg = "The dataframe is empty, cannot introduce errors."
        raise ValueError(msg)
