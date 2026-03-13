from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from tab_err.api import low_level
from tab_err.error_mechanism import EAR

if TYPE_CHECKING:
    import pandas as pd

    from tab_err import ErrorMechanism, ErrorType


@dataclasses.dataclass
class ErrorModel:
    """Combines an error mechanism and error type and defines how many percent of the column should be perturbed.

    Attributes:
        error_mechanism (ErrorMechanism): Instance of an `ErrorMechanism` that will be applied.
        error_type (ErrorType): Instance of an `ErrorType` that will be applied.
        error_rate (float): Defines how many percent should be perturbed.
    """

    error_mechanism: ErrorMechanism
    error_type: ErrorType
    error_rate: float

    def __repr__(self) -> str:
        """Unambiguous representation, evaluating the mechanism's repr but only the type's name."""
        if self.error_mechanism.__class__ == EAR:
            return (
                f"{self.__class__.__name__}("
                f"error_mechanism={self.error_mechanism.__class__.__name__}(condition_to_column='{self.error_mechanism.condition_to_column}'), "
                f"error_type={self.error_type.__class__.__name__}, "
                f"error_rate={self.error_rate})"
            )
        else:
            return (
                f"{self.__class__.__name__}("
                f"error_mechanism={self.error_mechanism.__class__.__name__}, "
                f"error_type={self.error_type.__class__.__name__}, "
                f"error_rate={self.error_rate})"
            )

    def __str__(self) -> str:
        """Readable representation for end-users."""
        # Assumes error_rate is a float like 0.05. Displays as 5.0%.
        if self.error_mechanism.__class__ == EAR:
            return f"ErrorModel: {self.error_rate:.1%} '{self.error_type.__class__.__name__}' errors via {self.error_mechanism.__class__.__name__} conditioning on column '{self.error_mechanism.condition_to_column}'"
        else:
            return f"ErrorModel: {self.error_rate:.1%} '{self.error_type.__class__.__name__}' errors via {self.error_mechanism.__class__.__name__}"

    def apply(self: ErrorModel, data: pd.DataFrame, column: str | int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Applies the defined ErrorModel to the given column of a pandas DataFrame.

        Args:
            data (pd.DataFrame): The pandas DataFrame to create errors in.
            column (str | int): The column to create errors in.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                - The first element is a copy of 'data' with errors.
                - The second element is the associated error mask.
        """
        data_with_errors, error_mask = low_level.create_errors(
            data=data, column=column, error_rate=self.error_rate, error_mechanism=self.error_mechanism, error_type=self.error_type
        )

        return data_with_errors, error_mask
