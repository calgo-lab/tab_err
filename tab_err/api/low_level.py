from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

from tab_err._utils import check_data_emptiness, check_error_rate, set_column

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from tab_err import ErrorMechanism, ErrorType


def create_errors(
    data: IntoDataFrame, column: str | int, error_rate: float, error_mechanism: ErrorMechanism, error_type: ErrorType
) -> tuple[IntoDataFrame, IntoDataFrame]:
    """Creates errors in a given column of a DataFrame.

    Args:
        data (IntoDataFrame): The DataFrame to create errors in. Supports pandas, Polars, and other narwhals-compatible backends.
        column (str | int): The column to create errors in.
        error_rate (float): The rate at which errors will be created.
        error_mechanism (ErrorMechanism): The mechanism, controls the error distribution.
        error_type (ErrorType): The type of the error that will be distributed.

    Returns:
        tuple[IntoDataFrame, IntoDataFrame]:
            - The first element is a copy of 'data' with errors.
            - The second element is the associated error mask.
            Both are returned in the same format as the input data.
    """
    # Wrap native DataFrame to narwhals
    data_nw = nw.from_native(data, eager_only=True)

    check_error_rate(error_rate)
    check_data_emptiness(data_nw)

    # Clone the data to avoid modifying the original
    data_copy = data_nw.clone()

    error_mask = error_mechanism.sample(data_copy, column, error_rate, error_mask=None)
    series = error_type.apply(data_copy, error_mask, column)
    data_copy = set_column(data_copy, column, series)

    # Return in the original format
    return nw.to_native(data_copy), nw.to_native(error_mask)
