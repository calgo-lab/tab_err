import numpy as np
import pandas as pd
import pytest

from tab_err import error_mechanism, error_type
from tab_err.api.low_level import create_errors


@pytest.fixture
def test_data() -> dict[str, pd.DataFrame]:
    """Fixture to provide test data before each test runs."""
    rng = np.random.default_rng(42)
    return {
        "data_10rows_3columns": pd.DataFrame({"A": rng.integers(0, 100, 10), "B": rng.random(10), "C": rng.choice(["X", "Y", "Z"], 10)}),
        "data_100rows_3columns": pd.DataFrame({"A": rng.integers(0, 100, 100), "B": rng.random(100), "C": rng.choice(["X", "Y", "Z"], 100)}),
    }


def test_create_errors_error_rates(test_data: dict[str, pd.DataFrame]) -> None:
    """Test that create_errors returns two DataFrames with expected properties."""
    for i in range(11):
        error_rate = 0.1 * float(i)
        modified_data_100rows_3columns, data_100rows_3columns_error_mask = create_errors(
            test_data["data_100rows_3columns"], "A", error_rate, error_mechanism.ECAR(), error_type.AddDelta()
        )
        modified_data_10rows_3columns, data_10rows_3columns_error_mask = create_errors(
            test_data["data_10rows_3columns"], "A", error_rate, error_mechanism.ECAR(), error_type.AddDelta()
        )

        # Assert that the proportion of different values is correct
        assert pytest.approx(error_rate / 3.0) == modified_data_10rows_3columns.ne(data_10rows_3columns_error_mask).to_numpy().mean()
        assert pytest.approx(error_rate / 3.0) == modified_data_100rows_3columns.ne(data_100rows_3columns_error_mask).to_numpy().mean()

        # Assert that the error masks have the correct proportion of True to False -- Note only one column is errored
        assert pytest.approx(error_rate / 3.0) == data_100rows_3columns_error_mask.to_numpy().mean()
        assert pytest.approx(error_rate / 3.0) == data_10rows_3columns_error_mask.to_numpy().mean()
