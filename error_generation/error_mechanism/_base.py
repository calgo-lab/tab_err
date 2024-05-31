from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pandas._typing import Dtype


class ErrorMechanism(ABC):
    def __init__(self: ErrorMechanism, error_rate: float, condition_to_column: int | str | None = None) -> None:
        self.error_rate = error_rate
        self.condition_to_column = condition_to_column

    def sample(
        self: ErrorMechanism,
        data: pd.DataFrame,
        seed: int | None = None,
    ) -> pd.DataFrame:
        if self.error_rate < 0 or self.error_rate > 1:
            error_rate_msg = "'error_rate' need to be float: 0 <= error_rate <= 1."
            raise ValueError(error_rate_msg)

        if not (isinstance(seed, int) or seed is None):
            msg = "'seed' need to be int or None."
            raise TypeError(msg)

        data_msg = "'data' needs to be a non-empty DataFrame."
        if not isinstance(data, pd.DataFrame):
            raise TypeError(data_msg)

        if data.empty:
            raise ValueError(data_msg)

        # At least two columns are necessary if we condition to another
        if self.condition_to_column is not None and len(data.columns) < 2:  # noqa: PLR2004
            msg = "'data' need at least 2 columns if 'condition_to_column' is given."
            raise ValueError(msg)

        return self._sample(data=data, error_rate=self.error_rate, condition_to_column=self.condition_to_column, seed=seed)

    @staticmethod
    @abstractmethod
    def _sample(data: pd.DataFrame, error_rate: float, condition_to_column: Dtype | None = None, seed: int | None = None) -> pd.DataFrame:
        pass
