from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from tab_err.api import low_level

if TYPE_CHECKING:
    import pandas as pd

    from .error_mechanism import ErrorMechanism
    from .error_type import ErrorType


@dataclasses.dataclass
class ErrorModel:
    """Combines an error mechanism and error type and defines how many percent of the column should be perturbed.

    Attributes:
        error_mechanism: Instance of an `ErrorMechanism` that will be applied.
        error_type: Instance of an `ErrorType` that will be applied.
        error_rate: Defines how many percent should be perturbed.
    """

    error_mechanism: ErrorMechanism
    error_type: ErrorType
    error_rate: float

    def apply_to(self: ErrorModel, table: pd.DataFrame, column: str | int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Applies the defined ErrorModel to the given column of a pandas DataFrame.

        Args:
            table: The pandas DataFrame to create errors in.
            column: The column to create errors in.

        Returns:
            A tuple of a copy of the table with errors, and the error mask.
        """
        table_with_errors, error_mask = low_level.create_errors(
            table=table, column=column, error_rate=self.error_rate, error_mechanism=self.error_mechanism, error_type=self.error_type
        )

        return table_with_errors, error_mask
