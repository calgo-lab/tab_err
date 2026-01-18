from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame
    from numpy.random import Generator


def _create_raw_test_data(rng: Generator) -> dict[str, dict]:
    """Create raw data dicts that can be converted to any backend.

    Args:
        rng: NumPy random generator for reproducible data generation.

    Returns:
        Dictionary mapping dataset names to column data dictionaries.
    """
    return {
        "data_10rows_3columns": {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": rng.random(10).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10).tolist(),
        },
        "data_4rows_5columns": {
            "A": rng.integers(0, 100, 4).tolist(),
            "B": rng.random(4).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 4).tolist(),
            "D": rng.integers(0, 100, 4).tolist(),
            "E": rng.random(4).tolist(),
        },
        "data_100rows_3columns": {
            "A": rng.integers(0, 100, 100).tolist(),
            "B": rng.random(100).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 100).tolist(),
        },
        "data_10rows_3columns_with_datetime": {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": pd.date_range(start="2025-03-04", periods=10, freq="2h").tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10).tolist(),
        },
    }


def assert_dataframes_equal(
    df1: IntoDataFrame,
    df2: IntoDataFrame,
    rtol: float = 1e-7,
    atol: float = 1e-10,
) -> None:
    """Compare DataFrames by converting to pandas and using pd.testing.assert_frame_equal.

    Args:
        df1: First DataFrame (any narwhals-compatible DataFrame).
        df2: Second DataFrame (any narwhals-compatible DataFrame).
        rtol: Relative tolerance for numeric comparisons.
        atol: Absolute tolerance for numeric comparisons.

    Raises:
        AssertionError: If DataFrames are not equal.
    """
    # Use narwhals to wrap and then get native back for conversion
    df1_native = nw.from_native(df1, eager_only=True).to_pandas()
    df2_native = nw.from_native(df2, eager_only=True).to_pandas()

    pd.testing.assert_frame_equal(df1_native, df2_native, check_dtype=False, rtol=rtol, atol=atol)


@pytest.fixture(params=["pandas", "polars"])
def test_data(request: pytest.FixtureRequest) -> dict[str, IntoDataFrame]:
    """Fixture to provide test data before each test runs.

    Parametrized to run tests with both pandas and polars backends.
    """
    rng = np.random.default_rng(42)
    raw_data = _create_raw_test_data(rng)

    if request.param == "pandas":
        return {k: pd.DataFrame(v) for k, v in raw_data.items()}
    return {k: pl.DataFrame(v) for k, v in raw_data.items()}
