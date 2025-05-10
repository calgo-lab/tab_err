import pytest
import numpy as np
import pandas as pd

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

