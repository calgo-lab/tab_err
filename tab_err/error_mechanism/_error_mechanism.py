from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from tab_err._utils import seed_randomness

if TYPE_CHECKING:
    import numpy as np


class ErrorMechanism(ABC):
    def __init__(self: ErrorMechanism, condition_to_column: int | str | None = None, seed: int | None = None) -> None:
        if not (isinstance(seed, int) or seed is None):
            msg = "'seed' needs to be int or None."
            raise TypeError(msg)

        self.condition_to_column = condition_to_column

        self._seed = seed
        self._random_generator: np.random.Generator

    def sample(
        self: ErrorMechanism,
        data: pd.DataFrame,
        column: int | str,
        error_rate: float,
        error_mask: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # TODO(seja): Docs
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
        # TODO(seja): Docs
        pass
