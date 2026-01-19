from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from narwhals.typing import IntoDType

import random
import warnings
from typing import Any

import narwhals as nw
import numpy as np


def set_column(data: nw.DataFrame, column: int | str, series: nw.Series) -> nw.DataFrame:
    """Replaces a column in the given DataFrame with the given Series.

    Returns a new DataFrame with the column replaced.
    """
    col_name = get_column_str(data, column)
    return data.with_columns(series.alias(col_name))


def get_column_str(data: nw.DataFrame, column: int | str) -> str:
    """Return column's name of the given DataFrame, where column can be defined as name or index."""
    if isinstance(column, int):
        col = data.columns[column]
    elif isinstance(column, str):
        col = column
    else:
        msg = f"Column must be an int or str, not {type(column)}"
        raise TypeError(msg)

    return col


def get_column(data: nw.DataFrame, column: int | str) -> nw.Series:
    """Selects a column from the given DataFrame and returns it as a Series."""
    return data[get_column_str(data, column)]


def seed_randomness_and_get_generator(seed: int | None) -> np.random.Generator:
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


def check_data_emptiness(data: nw.DataFrame) -> None:
    """Check that the dataset is not empty, raise a ValueError otherwise."""
    if data.is_empty():
        msg = "The dataframe is empty, cannot introduce errors."
        raise ValueError(msg)


def is_string_dtype(series: nw.Series) -> bool:
    """Check if a series has a string dtype."""
    return series.dtype in {nw.String, nw.Object}


def is_numeric_dtype(series: nw.Series) -> bool:
    """Check if a series has a numeric dtype."""
    return series.dtype.is_numeric()


def is_integer_dtype(series: nw.Series) -> bool:
    """Check if a series has an integer dtype."""
    return series.dtype.is_integer()


def is_datetime_dtype(series: nw.Series) -> bool:
    """Check if a series has a datetime dtype."""
    return series.dtype == nw.Datetime


def select_string_columns(data: nw.DataFrame) -> list[str | int]:
    """Select columns with string dtype."""
    return [col for col in data.columns if is_string_dtype(data[col])]


def select_numeric_columns(data: nw.DataFrame) -> list[str | int]:
    """Select columns with numeric dtype."""
    return [col for col in data.columns if is_numeric_dtype(data[col])]


def select_datetime_columns(data: nw.DataFrame) -> list[str | int]:
    """Select columns with datetime dtype."""
    return [col for col in data.columns if is_datetime_dtype(data[col])]


def select_numeric_or_datetime_columns(data: nw.DataFrame) -> list[str | int]:
    """Select columns with numeric or datetime dtype."""
    return [col for col in data.columns if is_numeric_dtype(data[col]) or is_datetime_dtype(data[col])]


def create_empty_boolean_mask(data: nw.DataFrame) -> nw.DataFrame:
    """Create an empty boolean mask DataFrame with the same shape as data."""
    n_rows = len(data)
    mask_values = [False] * n_rows
    return nw.from_dict(
        dict.fromkeys(data.columns, mask_values),
        backend=nw.get_native_namespace(data),
    )


def cast_series_like(series: nw.Series, like: nw.Series, column: int | str) -> nw.Series:
    """Cast series to the dtype of 'like' when possible, otherwise keep original."""
    if series.dtype == like.dtype:
        return series
    dtype: IntoDType = like.dtype

    try:
        return series.cast(dtype)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to cast column {column} to dtype {like.dtype}: {exc}. Keeping inferred dtype."
        warnings.warn(msg, stacklevel=2)
        return series


def _values_to_list(values: Sequence[Any] | np.ndarray) -> list[Any]:
    """Normalize values into a list for nw.new_series."""
    if isinstance(values, np.ndarray):
        return values.tolist()
    return list(values)


def new_series_like(data: nw.DataFrame, column: int | str, values: Sequence[Any] | np.ndarray) -> nw.Series:
    """Create a new series for 'column' and cast it back to the original dtype."""
    col_name = get_column_str(data, column)
    original = get_column(data, column)
    series = nw.new_series(col_name, _values_to_list(values), backend=nw.get_native_namespace(data))
    return cast_series_like(series, original, column)
