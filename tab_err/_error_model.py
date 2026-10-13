from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from tab_err.api import low_level
from tab_err.error_mechanism._error_mechanism import ErrorMechanism
from tab_err.error_type._error_type import ErrorType

if TYPE_CHECKING:
    import pandas as pd


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

    def to_dict(self: ErrorModel) -> dict[str, Any]:
        """Serializes the ErrorModel to a dict."""
        return {
            "error_mechanism": self.error_mechanism.to_dict(),
            "error_type": self.error_type.to_dict(),
            "error_rate": self.error_rate,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ErrorModel:
        """Deserializes an ErrorModel from a dict."""
        return ErrorModel(
            error_mechanism=ErrorMechanism.from_dict(data["error_mechanism"]),
            error_type=ErrorType.from_dict(data["error_type"]),
            error_rate=data["error_rate"],
        )
