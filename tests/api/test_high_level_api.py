from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
import numpy as np
import pytest

from tab_err.api.high_level import create_errors
from tests.conftest import assert_dataframes_equal

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame


def _get_shape(df: IntoDataFrame) -> tuple[int, int]:
    """Get shape of a DataFrame using narwhals."""
    return nw.from_native(df, eager_only=True).shape


def _get_numpy_mean(df: IntoDataFrame) -> float:
    """Get numpy array mean from a DataFrame using narwhals."""
    df_nw = nw.from_native(df, eager_only=True)
    all_values = []
    for col in df_nw.columns:
        all_values.extend(df_nw[col].to_numpy().tolist())
    return float(np.mean(all_values))


def _check_all_bool_dtypes(df: IntoDataFrame) -> bool:
    """Check if all columns in a DataFrame have boolean dtype."""
    df_nw = nw.from_native(df, eager_only=True)
    return all(df_nw[col].dtype == nw.Boolean for col in df_nw.columns)


def _check_same_type(df1: IntoDataFrame, df2: IntoDataFrame) -> bool:
    """Check if two DataFrames are of the same type."""
    return type(df1) is type(df2)


class TestHighLevelAPI:
    """Tests the high-level API."""

    def test_create_errors_basic(self, test_data: dict[str, IntoDataFrame]) -> None:
        """Test that create_errors returns two DataFrames with expected properties."""
        seed = 42
        error_rate = 0.5
        modified_data_4rows_5columns, data_4rows_5columns_error_mask = create_errors(test_data["data_4rows_5columns"], error_rate, seed=seed)
        modified_data_10rows_3columns, data_10rows_3columns_error_mask = create_errors(test_data["data_10rows_3columns"], error_rate, seed=seed)

        # Check that the output type matches the input type
        assert _check_same_type(modified_data_4rows_5columns, test_data["data_4rows_5columns"])
        assert _check_same_type(data_4rows_5columns_error_mask, test_data["data_4rows_5columns"])
        assert _check_same_type(modified_data_10rows_3columns, test_data["data_10rows_3columns"])
        assert _check_same_type(data_10rows_3columns_error_mask, test_data["data_10rows_3columns"])

        # Check Shapes
        assert _get_shape(modified_data_4rows_5columns) == _get_shape(test_data["data_4rows_5columns"])
        assert _get_shape(data_4rows_5columns_error_mask) == _get_shape(test_data["data_4rows_5columns"])
        assert _get_shape(modified_data_10rows_3columns) == _get_shape(test_data["data_10rows_3columns"])
        assert _get_shape(data_10rows_3columns_error_mask) == _get_shape(test_data["data_10rows_3columns"])

        # Assert the error masks contain only boolean values
        assert _check_all_bool_dtypes(data_4rows_5columns_error_mask)
        assert _check_all_bool_dtypes(data_10rows_3columns_error_mask)

        # Assert that the error masks have the correct proportion of True to False
        assert pytest.approx(error_rate) == _get_numpy_mean(data_4rows_5columns_error_mask)
        assert pytest.approx(error_rate) == _get_numpy_mean(data_10rows_3columns_error_mask)

    def test_create_errors_seed(self, test_data: dict[str, IntoDataFrame]) -> None:
        """Test that create_errors returns the same dataframe when a seed is used."""
        seed = 42
        error_rate = 0.5

        modified_data_1, error_mask_1 = create_errors(test_data["data_10rows_3columns"], error_rate=error_rate, seed=seed)
        modified_data_2, error_mask_2 = create_errors(test_data["data_10rows_3columns"], error_rate=error_rate, seed=seed)

        # Ensure same seed yields same dataframes
        assert_dataframes_equal(modified_data_1, modified_data_2)
        assert_dataframes_equal(error_mask_1, error_mask_2)

    def test_create_errors_error_rates(self, test_data: dict[str, IntoDataFrame]) -> None:
        """Test that create_errors returns two DataFrames with expected properties."""
        seed = 42
        for i in range(11):
            error_rate = 0.1 * float(i)
            _, data_100rows_3columns_error_mask = create_errors(test_data["data_100rows_3columns"], error_rate, seed=seed)
            _, data_10rows_3columns_error_mask = create_errors(test_data["data_10rows_3columns"], error_rate, seed=seed)
            _, data_10rows_3columns_with_datetime_error_mask = create_errors(test_data["data_10rows_3columns_with_datetime"], error_rate, seed=seed)

            # Assert that the error masks have the correct proportion of True to False
            assert pytest.approx(error_rate) == _get_numpy_mean(data_100rows_3columns_error_mask)
            assert pytest.approx(error_rate) == _get_numpy_mean(data_10rows_3columns_error_mask)
            assert pytest.approx(error_rate) == _get_numpy_mean(data_10rows_3columns_with_datetime_error_mask)

    def test_create_errors_more_models(self, test_data: dict[str, IntoDataFrame]) -> None:
        """Test that when more error models are introduced, the create_errors method has expected DataFrame return."""
        error_rate = 1.0
        seed = 42
        n_error_models = 2

        modified_data_4rows_5columns, data_4rows_5columns_error_mask = create_errors(
            test_data["data_4rows_5columns"], error_rate, n_error_models_per_column=n_error_models, seed=seed
        )
        modified_data_10rows_3columns, data_10rows_3columns_error_mask = create_errors(
            test_data["data_10rows_3columns"], error_rate, n_error_models_per_column=n_error_models, seed=seed
        )
        modified_data_100rows_3columns, data_100rows_3columns_error_mask = create_errors(
            test_data["data_100rows_3columns"], error_rate, n_error_models_per_column=n_error_models, seed=seed
        )

        # Check that the output type matches the input type
        assert _check_same_type(modified_data_4rows_5columns, test_data["data_4rows_5columns"])
        assert _check_same_type(data_4rows_5columns_error_mask, test_data["data_4rows_5columns"])
        assert _check_same_type(modified_data_10rows_3columns, test_data["data_10rows_3columns"])
        assert _check_same_type(data_10rows_3columns_error_mask, test_data["data_10rows_3columns"])
        assert _check_same_type(modified_data_100rows_3columns, test_data["data_100rows_3columns"])
        assert _check_same_type(data_100rows_3columns_error_mask, test_data["data_100rows_3columns"])

        # Check Shapes
        assert _get_shape(modified_data_4rows_5columns) == _get_shape(test_data["data_4rows_5columns"])
        assert _get_shape(data_4rows_5columns_error_mask) == _get_shape(test_data["data_4rows_5columns"])
        assert _get_shape(modified_data_10rows_3columns) == _get_shape(test_data["data_10rows_3columns"])
        assert _get_shape(data_10rows_3columns_error_mask) == _get_shape(test_data["data_10rows_3columns"])
        assert _get_shape(modified_data_100rows_3columns) == _get_shape(test_data["data_100rows_3columns"])
        assert _get_shape(data_100rows_3columns_error_mask) == _get_shape(test_data["data_100rows_3columns"])

        # Assert the error masks contain only boolean values
        assert _check_all_bool_dtypes(data_4rows_5columns_error_mask)
        assert _check_all_bool_dtypes(data_10rows_3columns_error_mask)
        assert _check_all_bool_dtypes(data_100rows_3columns_error_mask)

        # Assert that the error masks have the correct proportion of True to False
        assert pytest.approx(error_rate) == _get_numpy_mean(data_4rows_5columns_error_mask)
        assert pytest.approx(error_rate) == _get_numpy_mean(data_10rows_3columns_error_mask)
        assert pytest.approx(error_rate) == _get_numpy_mean(data_100rows_3columns_error_mask)
