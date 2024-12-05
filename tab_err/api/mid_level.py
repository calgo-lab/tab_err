from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import pandas as pd

from tab_err._utils import set_column

if TYPE_CHECKING:
    from tab_err._error_model import ErrorModel


@dataclasses.dataclass
class MidLevelConfig:
    """Configuration of the mid_level API.

    The mid_level API applies N pairs of (error_mechanism, error_type) to a data. In consequence, the user
    is required to specify up to N pairs of error_mechanism, error_type per column when calling the mid_level
    API.
    """

    columns: dict[int | str, list[ErrorModel]]


def create_errors(data: pd.DataFrame, config: MidLevelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates errors in a given DataFrame, following a user-defined configuration.

    Args:
        data: The pandas DataFrame to create errors in.
        config: The configuration for the error generation process.

    Returns:
        A tuple of a copy of the data with errors, and the error mask.
    """
    data_dirty = data.copy()
    error_mask = pd.DataFrame(data=False, index=data.index, columns=data.columns)

    for column in config.columns:
        for error_model in config.columns[column]:
            error_mechanism = error_model.error_mechanism
            error_type = error_model.error_type
            error_rate = error_model.error_rate

            old_error_mask = error_mask.copy()
            error_mask = error_mechanism.sample(data, column, error_rate, error_mask)

            series = error_type.apply(data_dirty, old_error_mask != error_mask, column)
            set_column(data_dirty, column, series)

    return data_dirty, error_mask
