from __future__ import annotations

from typing import TYPE_CHECKING

from error_generation.utils import set_column

if TYPE_CHECKING:
    import pandas as pd

    from error_generation.error_mechanism import ErrorMechanism
    from error_generation.error_type import ErrorType
    from error_generation.utils import Column


def create_errors(
    table: pd.DataFrame,
    column: Column,
    error_rate: float,
    mechanism: ErrorMechanism,
    error_type: ErrorType,
    condition_to_column: Column | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    error_mask = mechanism.sample(table, error_rate, condition_to_column, seed=None)
    series = error_type.apply(table, error_mask, column)
    set_column(table, column, series)
    return table
