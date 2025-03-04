import numpy as np
import pandas as pd
import pytest

from tab_err.api.high_level import create_errors


@pytest.fixture
def test_data() -> dict[str, pd.DataFrame]:
    """Fixture to provide test data before each test runs."""
    rng = np.random.default_rng(42)
    return {
        "data_10rows_3columns": pd.DataFrame({"A": rng.integers(0, 100, 10), "B": rng.random(10), "C": rng.choice(["X", "Y", "Z"], 10)}),
        "data_4rows_5columns": pd.DataFrame(
            {"A": rng.integers(0, 100, 4), "B": rng.random(4), "C": rng.choice(["X", "Y", "Z"], 4), "D": rng.integers(0, 100, 4), "E": rng.random(4)}
        ),
        "data_100rows_3columns": pd.DataFrame({"A": rng.integers(0, 100, 100), "B": rng.random(100), "C": rng.choice(["X", "Y", "Z"], 100)}),
        "data_10rows_3columns_with_datetime": pd.DataFrame(
            {"A": rng.integers(0, 100, 10), "B": pd.date_range(start="2025-03-04", periods=10, freq="2h"), "C": rng.choice(["X", "Y", "Z"], 10)}
        ),
    }


def test_create_errors_basic(test_data: dict[str, pd.DataFrame]) -> None:
    """Test that create_errors returns two DataFrames with expected properties."""
    seed = 45
    error_rate = 0.5
    modified_data_4rows_5columns, data_4rows_5columns_error_mask = create_errors(test_data["data_4rows_5columns"], error_rate, seed=seed)
    modified_data_10rows_3columns, data_10rows_3columns_error_mask = create_errors(test_data["data_10rows_3columns"], error_rate, seed=seed)

    # Check that they are still dataframes
    assert isinstance(modified_data_4rows_5columns, pd.DataFrame)
    assert isinstance(data_4rows_5columns_error_mask, pd.DataFrame)
    assert isinstance(modified_data_10rows_3columns, pd.DataFrame)
    assert isinstance(data_10rows_3columns_error_mask, pd.DataFrame)

    # Check Shapes
    assert modified_data_4rows_5columns.shape == test_data["data_4rows_5columns"].shape
    assert data_4rows_5columns_error_mask.shape == test_data["data_4rows_5columns"].shape
    assert modified_data_10rows_3columns.shape == test_data["data_10rows_3columns"].shape
    assert data_10rows_3columns_error_mask.shape == test_data["data_10rows_3columns"].shape

    # Assert the error masks contain only boolean values
    assert data_4rows_5columns_error_mask.dtypes.apply(lambda dt: np.issubdtype(dt, np.bool_)).all()
    assert data_10rows_3columns_error_mask.dtypes.apply(lambda dt: np.issubdtype(dt, np.bool_)).all()

    # Assert that the correct number of cells in the dataframes are actually different
    assert pytest.approx(error_rate) == modified_data_4rows_5columns.ne(test_data["data_4rows_5columns"]).to_numpy().mean()
    assert pytest.approx(error_rate) == modified_data_10rows_3columns.ne(test_data["data_10rows_3columns"]).to_numpy().mean()

    # Assert that the error masks have the correct proportion of True to False
    assert pytest.approx(error_rate) == data_4rows_5columns_error_mask.to_numpy().mean()
    assert pytest.approx(error_rate) == data_10rows_3columns_error_mask.to_numpy().mean()


def test_create_errors_seed(test_data: dict[str, pd.DataFrame]) -> None:
    """Test that create_errors returns the same dataframe when a seed is used."""
    error_rate = 0.5
    seed = 45

    modified_data_1, error_mask_1 = create_errors(test_data["data_10rows_3columns"], error_rate=error_rate, seed=seed)
    modified_data_2, error_mask_2 = create_errors(test_data["data_10rows_3columns"], error_rate=error_rate, seed=seed)

    # Ensure same seed yields same dataframes
    pd.testing.assert_frame_equal(modified_data_1, modified_data_2)
    pd.testing.assert_frame_equal(error_mask_1, error_mask_2)


def test_create_errors_error_rates(test_data: dict[str, pd.DataFrame]) -> None:
    """Test that create_errors returns two DataFrames with expected properties."""
    seed = 46
    for i in range(11):
        error_rate = 0.1 * float(i)
        modified_data_100rows_3columns, data_100rows_3columns_error_mask = create_errors(test_data["data_100rows_3columns"], error_rate, seed=seed)
        modified_data_10rows_3columns, data_10rows_3columns_error_mask = create_errors(test_data["data_10rows_3columns"], error_rate, seed=seed)
        modified_data_10rows_3columns_with_datetime, data_10rows_3columns_with_datetime_error_mask = create_errors(
            test_data["data_10rows_3columns_with_datetime"], error_rate, seed=seed
        )

        # Assert that the proportion of different values is correct
        assert pytest.approx(error_rate) == modified_data_100rows_3columns.ne(test_data["data_100rows_3columns"]).to_numpy().mean()
        assert pytest.approx(error_rate) == modified_data_10rows_3columns.ne(test_data["data_10rows_3columns"]).to_numpy().mean()
        assert pytest.approx(error_rate) == modified_data_10rows_3columns_with_datetime.ne(test_data["data_10rows_3columns_with_datetime"]).to_numpy().mean()

        # Assert that the error masks have the correct proportion of True to False
        assert pytest.approx(error_rate) == data_100rows_3columns_error_mask.to_numpy().mean()
        assert pytest.approx(error_rate) == data_10rows_3columns_error_mask.to_numpy().mean()
        assert pytest.approx(error_rate) == data_10rows_3columns_with_datetime_error_mask.to_numpy().mean()


def test_create_errors_more_models(test_data: dict[str, pd.DataFrame]) -> None:
    """Test that when more error models are introduced, the create_errors method has expected DataFrame return."""
    error_rate = 1.0
    seed = 47
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

    # Check that they are still dataframes
    assert isinstance(modified_data_4rows_5columns, pd.DataFrame)
    assert isinstance(data_4rows_5columns_error_mask, pd.DataFrame)
    assert isinstance(modified_data_10rows_3columns, pd.DataFrame)
    assert isinstance(data_10rows_3columns_error_mask, pd.DataFrame)
    assert isinstance(modified_data_100rows_3columns, pd.DataFrame)
    assert isinstance(data_100rows_3columns_error_mask, pd.DataFrame)

    # Check Shapes
    assert modified_data_4rows_5columns.shape == test_data["data_4rows_5columns"].shape
    assert data_4rows_5columns_error_mask.shape == test_data["data_4rows_5columns"].shape
    assert modified_data_10rows_3columns.shape == test_data["data_10rows_3columns"].shape
    assert data_10rows_3columns_error_mask.shape == test_data["data_10rows_3columns"].shape
    assert modified_data_100rows_3columns.shape == test_data["data_100rows_3columns"].shape
    assert data_100rows_3columns_error_mask.shape == test_data["data_100rows_3columns"].shape

    # Assert the error masks contain only boolean values
    assert data_4rows_5columns_error_mask.dtypes.apply(lambda dt: np.issubdtype(dt, np.bool_)).all()
    assert data_10rows_3columns_error_mask.dtypes.apply(lambda dt: np.issubdtype(dt, np.bool_)).all()
    assert data_100rows_3columns_error_mask.dtypes.apply(lambda dt: np.issubdtype(dt, np.bool_)).all()

    # Assert that the correct number of cells in the dataframes are actually different
    assert pytest.approx(error_rate) == modified_data_4rows_5columns.ne(test_data["data_4rows_5columns"]).to_numpy().mean()
    assert pytest.approx(error_rate) == modified_data_10rows_3columns.ne(test_data["data_10rows_3columns"]).to_numpy().mean()
    assert pytest.approx(error_rate) == modified_data_100rows_3columns.ne(test_data["data_100rows_3columns"]).to_numpy().mean()

    # Assert that the error masks have the correct proportion of True to False
    assert pytest.approx(error_rate) == data_4rows_5columns_error_mask.to_numpy().mean()
    assert pytest.approx(error_rate) == data_10rows_3columns_error_mask.to_numpy().mean()
    assert pytest.approx(error_rate) == data_100rows_3columns_error_mask.to_numpy().mean()
