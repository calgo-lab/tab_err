from __future__ import annotations

import pandas as pd

from tab_err.utils import MidLevelConfig, set_column


def create_errors(table: pd.DataFrame, config: MidLevelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates errors in a given DataFrame, following a user-defined configuration.

    Args:
        table: The pandas DataFrame to create errors in.
        config: The configuration for the error generation process.

    Returns:
        A tuple of a copy of the table with errors, and the error mask.
    """
    table_dirty = table.copy()
    error_mask = pd.DataFrame(data=False, index=table.index, columns=table.columns)

    for column in config.columns:
        for error_model in config.columns[column]:
            error_mechanism = error_model.error_mechanism
            error_type = error_model.error_type
            error_rate = error_model.error_rate

            old_error_mask = error_mask.copy()
            error_mask = error_mechanism.sample(table, column, error_rate, error_mask)

            series = error_type.apply(table_dirty, old_error_mask != error_mask, column)
            set_column(table_dirty, column, series)
    return table_dirty, error_mask
