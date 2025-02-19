from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from tab_err._utils import seed_randomness

from ._config import ErrorTypeConfig

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class ErrorType(ABC):
    """Error Type Abstract Base Class."""

    def __init__(self: ErrorType, config: ErrorTypeConfig | dict | None = None, seed: int | None = None) -> None:
        """Initialization method of ErrorType class.

        Args:
            config (ErrorTypeConfig | dict | None, optional): Config denoting special values to be used in each ErrorType implementation. Defaults to None.
            seed (int | None, optional): Random seed for random number generator. Defaults to None.

        Raises:
            TypeError: If the type of seed is not int or None, a TypeError is raised.
            TypeError: If the type of config is not ErrorTypeConfig, dict, or None, a TypeError is raised.
        """
        if not (isinstance(seed, int) or seed is None):
            msg = f"'seed' needs to be int or None but was {type(seed)}."
            raise TypeError(msg)

        if config is None:
            self.config = ErrorTypeConfig()

        elif isinstance(config, dict):
            self.config = ErrorTypeConfig(**config)

        elif isinstance(config, ErrorTypeConfig):
            self.config = config

        else:
            msg = f"config must be of type ErrorTypeConfig or dict but was {type(config)}"
            raise TypeError(msg)

        self._seed = seed
        self._random_generator: np.random.Generator

    def apply(self: ErrorType, data: pd.DataFrame, error_mask: pd.DataFrame, column: str | int) -> pd.Series:
        """Applies an ErrorType to a column of 'data'. Does type and shape checking and creates a random number generator.

        Args:
            data (pd.DataFrame): The Pandas DataFrame containing the column where errors are to be introduced.
            error_mask (pd.DataFrame): The Pandas DataFrame containing the error mask for 'column'.
            column (str | int): The index in the 'data' and 'error_mask' DataFrames where errors are to be introduced.

<<<<<<< HEAD
        Raises:
            ValueError: If the shape of data and the shape of error_mask are not equal, a ValueError is thrown.

=======
>>>>>>> main
        Returns:
            pd.Series: The data column, 'column', after errors of ErrorType at the locations specified by 'error_mask' are introduced.
        """
        self._check_type(data, column)
        print(f"In error_type.apply() the datatype of {column} is {data[column].dtype}. The type being applied is {self}")
        if data.shape != error_mask.shape:
            msg = f"The shape of 'data': {data.shape} was different from the shape of 'error_mask': {error_mask.shape}. They should be the same."
            raise ValueError(msg)

        self._random_generator = seed_randomness(self._seed)
        return self._apply(data, error_mask, column)

    def get_valid_columns(self: ErrorType, data: pd.DataFrame, preserve_dtypes: bool = True) -> list[str | int]:
        """Finds the valid columns to which the error type can be applied. Wrapper around _get_valid_columns."""
        return self._get_valid_columns(data, preserve_dtypes)

    @staticmethod
    @abstractmethod
    def _check_type(data: pd.DataFrame, column: str | int) -> None:
        pass

    # TODO(nich): def _get_valid_columns(data: pd.DataFrame, preserve_dtypes: bool = True) -> list[str | int]:
    # supposed to check for which columns this error type can be applied and returns those.
    @abstractmethod
    def _get_valid_columns(self: ErrorType, data: pd.DataFrame, preserve_dtypes: bool = True) -> list[str | int]:
        """Finds the valid columns to which the error type can be applied."""

    @abstractmethod
    # TODO(nich): def _apply(data: pd.DataFrame, error_mask: pd.DataFrame) -> pd.DataFrame:
    # Assumes 'data' has valid columns. Simply applies error_type to those cells where error_mask is True.
    # Returns changed data
    def _apply(self: ErrorType, data: pd.DataFrame, error_mask: pd.DataFrame, column: str | int) -> pd.Series:
        """Abstract method for the application of an ErrorType to the cells in 'data' where 'error_mask' is True.

        Args:
            data (pd.DataFrame): The Pandas DataFrame containing the column where errors are to be introduced.
            error_mask (pd.DataFrame): The Pandas DataFrame containing the error mask for 'column'.
            column (str | int): The index in the 'data' and 'error_mask' DataFrames where errors are to be introduced.

        Returns:
            pd.Series: The data column, 'column', after errors of ErrorType at the locations specified by 'error_mask' are introduced.
        """

    def to_dict(self: ErrorType) -> dict[str, Any]:
        """Serialized the ErrorType object into a dictionary.

        Returns:
            dict[str, Any]: A dictionary representation of the ErrorType object.
        """
        return {
            "error_type": self.__class__.__name__,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls: type[ErrorType], data: dict[str, Any]) -> ErrorType:
        """Deserialize an ErrorType object from a dictionary.

        Args:
            data (dict[str, Any]): A dictionary representation of the ErrorType object.

        Returns:
            ErrorType: An ErrorType object deserialized from the dictionary.
        """
        error_type_name = data["error_type"]
        error_type_class = globals()[error_type_name]

        return error_type_class(data["config"])
