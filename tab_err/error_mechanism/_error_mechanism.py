from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from tab_err._utils import seed_randomness

if TYPE_CHECKING:
    import numpy as np


class ErrorMechanism(ABC):
    """Error Mechanism Abstract Base Class."""

    def __init__(self: ErrorMechanism, condition_to_column: int | str | None = None, seed: int | None = None) -> None:
        """Initialization method of the Error Mechanism class; defines the general initialization for ErrorMechanism objects.

        Args:
            condition_to_column (int | str | None, optional): For EAR class implementation, determines which column errors are derived from. Defaults to None.
            seed (int | None, optional): Random seed. Defaults to None.

        Attributes:
            condition_to_column (int | str | None, optional): For EAR class implementation, determines which column errors are derived from. Defaults to None.
            _seed (int | None, optional): Random seed. Defaults to None.
            _random_generator (np.random.Generator): The random error generator for choosing entries at which to generate an error.

        Raises:
            TypeError: Raised if the seed is not int or None.
        """
        if not (isinstance(seed, int) or seed is None):
            msg = f"'seed' needs to be int or None but was {type(seed)}."
            raise TypeError(msg)

        self.condition_to_column = condition_to_column

        self._seed = seed
        self._random_generator: np.random.Generator

    def sample(
        self: ErrorMechanism,
        data: pd.DataFrame,
        column: str | int,
        error_rate: float,
        error_mask: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Returns an error mask for locations to introduce errors in a pandas DataFrame.

        Description:
            Does error checking for the abstract method '_sample'.
            Assigns the _random_generator attribute.
            Calls subclass _sample method.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to
            column (str | int): The column of 'data' to create an error mask for
            error_rate (float): Percentage of rows to be affected by errors in range [0,1].
            error_mask (pd.DataFrame | None, optional): An existing error mask to add more errors to in the case of the mid-/high-level APIs. Defaults to None.

        Raises:
            ValueError: If error rate is out of the [0,1] interval, a ValueError is thrown
            TypeError: If the 'data' argument is not a pandas dataframe or the data is empty, a TypeError is thrown
            ValueError: If required and there are not 2 columns in the 'data' argument, a ValueError is thrown.

        Returns:
            pd.DataFrame: Updated dataframe with the generated error mask
        """
        if error_rate < 0 or error_rate > 1:
            error_rate_msg = "'error_rate' need to be float: 0 <= error_rate <= 1."
            raise ValueError(error_rate_msg)

        if not isinstance(data, pd.DataFrame) or data.empty:
            data_msg = "'data' needs to be a non-empty DataFrame."
            raise TypeError(data_msg)

        # At least two columns are necessary if we condition to another
        if self.condition_to_column is not None and len(data.columns) < 2:  # noqa: PLR2004
            msg = "'data' need at least 2 columns if 'condition_to_column' is given."
            raise ValueError(msg)

        # When using the mid_level or high_level API, error mechanisms sample on top of
        # an existing error_mask. To avoid inserting errors into cells that another error_mechanism
        # already inserted errors into, we have error mechanisms sample only from cells that
        # do not contain errors.
        if error_mask is None:  # initialize empty error_mask
            error_mask = pd.DataFrame(data=False, index=data.index, columns=data.columns)

        self._random_generator = seed_randomness(self._seed)
        return self._sample(data, column, error_rate, error_mask)

    @abstractmethod
    def _sample(self: ErrorMechanism, data: pd.DataFrame, column: str | int, error_rate: float, error_mask: pd.DataFrame) -> pd.DataFrame:
        """Abstract method for the creation of an error mask over a given Pandas DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to
            column (str | int): The column of `data` to create an error mask for
            error_rate (float): Proportion of rows to be affected by errors; in range [0,1]
            error_mask (pd.DataFrame): A Pandas `DataFrame` with the same index & columns as `data` that will be modified and returned

        Returns:
            pd.DataFrame: A Pandas `DataFrame` with `True` values at entries where an error should be introduced, `False` otherwise
        """
