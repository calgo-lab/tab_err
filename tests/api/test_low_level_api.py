from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw
import numpy as np
import pytest

from tab_err import error_mechanism, error_type
from tab_err.api.low_level import create_errors

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame


def _get_numpy_mean(df: IntoDataFrame) -> float:
    """Get numpy array mean from a DataFrame using narwhals."""
    df_nw = nw.from_native(df, eager_only=True)
    all_values = []
    for col in df_nw.columns:
        all_values.extend(df_nw[col].to_numpy().tolist())
    return float(np.mean(all_values))


class TestLowLevelAPI:
    """Tests the low-level API."""

    def test_create_errors_error_rates(self, test_data: dict[str, IntoDataFrame]) -> None:
        """Test that create_errors returns two DataFrames with expected properties."""
        for i in range(11):
            error_rate = 0.1 * float(i)
            _, data_100rows_3columns_error_mask = create_errors(
                test_data["data_100rows_3columns"], "A", error_rate, error_mechanism.ECAR(), error_type.AddDelta()
            )
            _, data_10rows_3columns_error_mask = create_errors(
                test_data["data_10rows_3columns"], "A", error_rate, error_mechanism.ECAR(), error_type.AddDelta()
            )

            # Assert that the error masks have the correct proportion of True to False - Note only one column is errored
            assert pytest.approx(error_rate / 3.0) == _get_numpy_mean(data_100rows_3columns_error_mask)
            assert pytest.approx(error_rate / 3.0) == _get_numpy_mean(data_10rows_3columns_error_mask)
