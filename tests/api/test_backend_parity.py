"""Tests to verify pandas and polars produce identical outputs with the same seed."""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from tab_err import ErrorModel, error_mechanism, error_type
from tab_err.api import MidLevelConfig, low_level, mid_level
from tab_err.api.high_level import create_errors as high_level_create_errors
from tests.conftest import assert_dataframes_equal

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame


def _mask_sum(df: IntoDataFrame) -> int:
    """Sum all boolean values in a mask DataFrame."""
    df_nw = nw.from_native(df, eager_only=True)
    total = 0
    for col in df_nw.columns:
        total += int(df_nw[col].to_numpy().sum())
    return total


def _column_all(df: IntoDataFrame, column: str) -> bool:
    """Check if all values in a column are True."""
    df_nw = nw.from_native(df, eager_only=True)
    return bool(df_nw[column].to_numpy().all())


class TestLowLevelBackendParity:
    """Tests for low-level API backend parity."""

    @pytest.mark.parametrize("error_rate", [0.0, 0.1, 0.5, 1.0])
    @pytest.mark.parametrize("seed", [42, 123, 999])
    def test_ecar_add_delta_parity(self, error_rate: float, seed: int) -> None:
        """Test ECAR + AddDelta produces identical results for pandas and polars."""
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": rng.random(10).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10).tolist(),
        }
        pd_df, pl_df = pd.DataFrame(raw), pl.DataFrame(raw)

        pd_result, pd_mask = low_level.create_errors(pd_df, "A", error_rate, error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed))
        pl_result, pl_mask = low_level.create_errors(pl_df, "A", error_rate, error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed))

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)

    @pytest.mark.parametrize(
        ("mechanism_cls", "error_type_cls", "column"),
        [
            (error_mechanism.ECAR, error_type.AddDelta, "A"),  # numeric column
            (error_mechanism.ECAR, error_type.Outlier, "A"),  # numeric column
            (error_mechanism.ECAR, error_type.Typo, "C"),  # string column
            (error_mechanism.ECAR, error_type.MissingValue, "A"),  # any column
            (error_mechanism.ECAR, error_type.Replace, "C"),  # string column
            (error_mechanism.ENAR, error_type.AddDelta, "A"),  # numeric column
        ],
    )
    def test_error_type_mechanism_combinations(
        self,
        mechanism_cls: type,
        error_type_cls: type,
        column: str,
    ) -> None:
        """Test various error type and mechanism combinations produce identical results."""
        seed = 42
        error_rate = 0.3
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": rng.random(10).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10).tolist(),
        }
        pd_df, pl_df = pd.DataFrame(raw), pl.DataFrame(raw)

        pd_result, pd_mask = low_level.create_errors(pd_df, column, error_rate, mechanism_cls(seed=seed), error_type_cls(seed=seed))
        pl_result, pl_mask = low_level.create_errors(pl_df, column, error_rate, mechanism_cls(seed=seed), error_type_cls(seed=seed))

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)

    def test_category_swap_parity(self) -> None:
        """Test CategorySwap produces identical results for pandas and polars with Categorical dtype."""
        seed = 42
        error_rate = 0.3
        rng = np.random.default_rng(42)
        choices = rng.choice(["X", "Y", "Z"], 10).tolist()

        # Create pandas DataFrame with categorical dtype
        pd_df = pd.DataFrame(
            {
                "A": rng.integers(0, 100, 10).tolist(),
                "B": rng.random(10).tolist(),
                "C": pd.Categorical(choices),
            }
        )

        # Create polars DataFrame with categorical dtype
        pl_df = pl.DataFrame(
            {
                "A": rng.integers(0, 100, 10).tolist(),
                "B": rng.random(10).tolist(),
                "C": pl.Series(choices).cast(pl.Categorical),
            }
        )

        _pd_result, pd_mask = low_level.create_errors(pd_df, "C", error_rate, error_mechanism.ECAR(seed=seed), error_type.CategorySwap(seed=seed))
        _pl_result, pl_mask = low_level.create_errors(pl_df, "C", error_rate, error_mechanism.ECAR(seed=seed), error_type.CategorySwap(seed=seed))

        assert_dataframes_equal(pd_mask, pl_mask)

    def test_ear_mechanism_parity(self) -> None:
        """Test EAR conditioned on another column produces identical results."""
        seed = 42
        error_rate = 0.3
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": rng.random(10).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10).tolist(),
        }
        pd_df, pl_df = pd.DataFrame(raw), pl.DataFrame(raw)

        pd_result, pd_mask = low_level.create_errors(
            pd_df, "A", error_rate, error_mechanism.EAR(condition_to_column="B", seed=seed), error_type.AddDelta(seed=seed)
        )
        pl_result, pl_mask = low_level.create_errors(
            pl_df, "A", error_rate, error_mechanism.EAR(condition_to_column="B", seed=seed), error_type.AddDelta(seed=seed)
        )

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)


class TestMidLevelBackendParity:
    """Tests for mid-level API backend parity."""

    def test_basic_config_parity(self) -> None:
        """Test single error model per column produces identical results."""
        seed = 42
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": rng.random(10).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10).tolist(),
        }
        pd_df, pl_df = pd.DataFrame(raw), pl.DataFrame(raw)

        config = MidLevelConfig(
            columns={
                "A": [ErrorModel(error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed), error_rate=0.3)],
            }
        )

        pd_result, pd_mask = mid_level.create_errors(pd_df, config)
        pl_result, pl_mask = mid_level.create_errors(pl_df, config)

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)

    def test_multiple_models_per_column_parity(self) -> None:
        """Test multiple error models on same column produces identical results."""
        seed = 42
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": rng.random(10).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10).tolist(),
        }
        pd_df, pl_df = pd.DataFrame(raw), pl.DataFrame(raw)

        config = MidLevelConfig(
            columns={
                "A": [
                    ErrorModel(error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed), error_rate=0.2),
                    ErrorModel(error_mechanism.ECAR(seed=seed + 1), error_type.Outlier(seed=seed + 1), error_rate=0.2),
                ],
            }
        )

        pd_result, pd_mask = mid_level.create_errors(pd_df, config)
        pl_result, pl_mask = mid_level.create_errors(pl_df, config)

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)


class TestHighLevelBackendParity:
    """Tests for high-level API backend parity."""

    @pytest.mark.parametrize("error_rate", [0.1, 0.5])
    @pytest.mark.parametrize("n_models", [1, 2])
    def test_automatic_error_generation_parity(self, error_rate: float, n_models: int) -> None:
        """Test automatic error generation produces identical results."""
        seed = 42
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 100).tolist(),
            "B": rng.random(100).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 100).tolist(),
        }
        pd_df, pl_df = pd.DataFrame(raw), pl.DataFrame(raw)

        pd_result, pd_mask = high_level_create_errors(pd_df, error_rate, n_error_models_per_column=n_models, seed=seed)
        pl_result, pl_mask = high_level_create_errors(pl_df, error_rate, n_error_models_per_column=n_models, seed=seed)

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)

    def test_with_error_type_filters_parity(self) -> None:
        """Test include/exclude error type filters produce identical results."""
        seed = 42
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 100).tolist(),
            "B": rng.random(100).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 100).tolist(),
        }
        pd_df, pl_df = pd.DataFrame(raw), pl.DataFrame(raw)

        # Test with include filter
        pd_result, pd_mask = high_level_create_errors(
            pd_df,
            error_rate=0.3,
            error_types_to_include=[error_type.AddDelta(seed=seed), error_type.MissingValue(seed=seed)],
            seed=seed,
        )
        pl_result, pl_mask = high_level_create_errors(
            pl_df,
            error_rate=0.3,
            error_types_to_include=[error_type.AddDelta(seed=seed), error_type.MissingValue(seed=seed)],
            seed=seed,
        )

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)

    def test_with_mechanism_filters_parity(self) -> None:
        """Test include/exclude mechanism filters produce identical results."""
        seed = 42
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 100).tolist(),
            "B": rng.random(100).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 100).tolist(),
        }
        pd_df, pl_df = pd.DataFrame(raw), pl.DataFrame(raw)

        # Test with mechanism exclude filter
        pd_result, pd_mask = high_level_create_errors(
            pd_df,
            error_rate=0.3,
            error_mechanisms_to_exclude=[error_mechanism.EAR(condition_to_column=None)],
            seed=seed,
        )
        pl_result, pl_mask = high_level_create_errors(
            pl_df,
            error_rate=0.3,
            error_mechanisms_to_exclude=[error_mechanism.EAR(condition_to_column=None)],
            seed=seed,
        )

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)

    def test_datetime_column_parity(self) -> None:
        """Test datetime column handling produces identical results."""
        seed = 42
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": pd.date_range(start="2025-03-04", periods=10, freq="2h").tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10).tolist(),
        }
        pd_df = pd.DataFrame(raw)
        pl_df = pl.DataFrame(raw)

        pd_result, pd_mask = high_level_create_errors(pd_df, error_rate=0.3, seed=seed)
        pl_result, pl_mask = high_level_create_errors(pl_df, error_rate=0.3, seed=seed)

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)


class TestReturnTypePreservation:
    """Tests verifying input type is preserved in output."""

    def test_pandas_input_returns_pandas(self) -> None:
        """Verify pandas input returns pandas output."""
        seed = 42
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": rng.random(10).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10).tolist(),
        }
        pd_df = pd.DataFrame(raw)

        # Low-level API
        result, mask = low_level.create_errors(pd_df, "A", 0.3, error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed))
        assert isinstance(result, pd.DataFrame)
        assert isinstance(mask, pd.DataFrame)

        # Mid-level API
        config = MidLevelConfig(columns={"A": [ErrorModel(error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed), error_rate=0.3)]})
        result, mask = mid_level.create_errors(pd_df, config)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(mask, pd.DataFrame)

        # High-level API
        result, mask = high_level_create_errors(pd_df, error_rate=0.3, seed=seed)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(mask, pd.DataFrame)

    def test_polars_input_returns_polars(self) -> None:
        """Verify polars input returns polars output."""
        seed = 42
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": rng.random(10).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10).tolist(),
        }
        pl_df = pl.DataFrame(raw)

        # Low-level API
        result, mask = low_level.create_errors(pl_df, "A", 0.3, error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed))
        assert isinstance(result, pl.DataFrame)
        assert isinstance(mask, pl.DataFrame)

        # Mid-level API
        config = MidLevelConfig(columns={"A": [ErrorModel(error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed), error_rate=0.3)]})
        result, mask = mid_level.create_errors(pl_df, config)
        assert isinstance(result, pl.DataFrame)
        assert isinstance(mask, pl.DataFrame)

        # High-level API
        result, mask = high_level_create_errors(pl_df, error_rate=0.3, seed=seed)
        assert isinstance(result, pl.DataFrame)
        assert isinstance(mask, pl.DataFrame)


class TestEdgeCases:
    """Tests for edge cases in backend parity."""

    def test_zero_error_rate_parity(self) -> None:
        """Test 0% error rate produces identical results."""
        seed = 42
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": rng.random(10).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10).tolist(),
        }
        pd_df, pl_df = pd.DataFrame(raw), pl.DataFrame(raw)

        pd_result, pd_mask = low_level.create_errors(pd_df, "A", 0.0, error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed))
        pl_result, pl_mask = low_level.create_errors(pl_df, "A", 0.0, error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed))

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)

        # Verify no errors were introduced
        assert _mask_sum(pd_mask) == 0
        assert _mask_sum(pl_mask) == 0

    def test_full_error_rate_parity(self) -> None:
        """Test 100% error rate produces identical results."""
        seed = 42
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": rng.random(10).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10).tolist(),
        }
        pd_df, pl_df = pd.DataFrame(raw), pl.DataFrame(raw)

        pd_result, pd_mask = low_level.create_errors(pd_df, "A", 1.0, error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed))
        pl_result, pl_mask = low_level.create_errors(pl_df, "A", 1.0, error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed))

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)

        # Verify all rows in column A have errors
        assert _column_all(pd_mask, "A")
        assert _column_all(pl_mask, "A")

    def test_large_dataset_parity(self) -> None:
        """Test 10,000 rows produces identical results."""
        seed = 42
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 10000).tolist(),
            "B": rng.random(10000).tolist(),
            "C": rng.choice(["X", "Y", "Z"], 10000).tolist(),
        }
        pd_df, pl_df = pd.DataFrame(raw), pl.DataFrame(raw)

        pd_result, pd_mask = low_level.create_errors(pd_df, "A", 0.3, error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed))
        pl_result, pl_mask = low_level.create_errors(pl_df, "A", 0.3, error_mechanism.ECAR(seed=seed), error_type.AddDelta(seed=seed))

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)

    def test_empty_string_handling_parity(self) -> None:
        """Test empty strings in string columns produces identical results."""
        seed = 42
        rng = np.random.default_rng(42)
        raw = {
            "A": rng.integers(0, 100, 10).tolist(),
            "B": rng.random(10).tolist(),
            "C": ["", "X", "", "Y", "Z", "", "X", "Y", "", "Z"],
        }
        pd_df, pl_df = pd.DataFrame(raw), pl.DataFrame(raw)

        pd_result, pd_mask = low_level.create_errors(pd_df, "C", 0.3, error_mechanism.ECAR(seed=seed), error_type.Typo(seed=seed))
        pl_result, pl_mask = low_level.create_errors(pl_df, "C", 0.3, error_mechanism.ECAR(seed=seed), error_type.Typo(seed=seed))

        assert_dataframes_equal(pd_result, pl_result)
        assert_dataframes_equal(pd_mask, pl_mask)
