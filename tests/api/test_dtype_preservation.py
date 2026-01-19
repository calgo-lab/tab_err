"""Tests to verify dtype preservation after error application."""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

from tab_err import error_mechanism, error_type
from tab_err.api import low_level


def _nw_dtype(df: IntoDataFrame, column: str) -> object:
    return nw.from_native(df, eager_only=True)[column].dtype


def _assert_dtype_preserved(original: IntoDataFrame, mutated: IntoDataFrame, column: str) -> None:
    assert _nw_dtype(original, column) == _nw_dtype(mutated, column)


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_categorical_dtype_preserved(backend: str) -> None:
    """Ensure categorical dtype is preserved after error injection."""
    raw = {"cat": ["a", "b", "a", "c"], "num": [1, 2, 3, 4]}
    if backend == "pandas":
        df = pd.DataFrame(raw)
        df["cat"] = pd.Categorical(df["cat"])
    else:
        df = pl.DataFrame(raw).with_columns(pl.col("cat").cast(pl.Categorical))

    mutated, _mask = low_level.create_errors(
        df,
        "cat",
        error_rate=0.5,
        error_mechanism=error_mechanism.ECAR(seed=0),
        error_type=error_type.CategorySwap(seed=0),
    )

    _assert_dtype_preserved(df, mutated, "cat")


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_datetime_dtype_preserved(backend: str) -> None:
    """Ensure datetime dtype is preserved after error injection."""
    ts = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]")
    raw = {"ts": ts, "num": [1, 2, 3]}
    df = pd.DataFrame(raw) if backend == "pandas" else pl.DataFrame(raw)

    mutated, _mask = low_level.create_errors(
        df,
        "ts",
        error_rate=0.5,
        error_mechanism=error_mechanism.ECAR(seed=0),
        error_type=error_type.AddDelta(config={"add_delta_value": 3600}, seed=0),
    )

    _assert_dtype_preserved(df, mutated, "ts")


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_nullable_int_dtype_preserved(backend: str) -> None:
    """Ensure nullable integer dtype is preserved after error injection."""
    raw = {"x": [1, 2, 3, 4]}
    df = (
        pd.DataFrame({"x": pd.Series(raw["x"], dtype="Int64")})
        if backend == "pandas"
        else pl.DataFrame({"x": pl.Series(raw["x"], dtype=pl.Int64)})
    )

    mutated, _mask = low_level.create_errors(
        df,
        "x",
        error_rate=0.5,
        error_mechanism=error_mechanism.ECAR(seed=0),
        error_type=error_type.AddDelta(config={"add_delta_value": 1}, seed=0),
    )

    _assert_dtype_preserved(df, mutated, "x")
