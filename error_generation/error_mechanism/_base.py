from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas._typing import Dtype


class NotInstantiableError(Exception):
    def __init__(self: NotInstantiableError) -> None:
        super().__init__("This class is not meant to be instantiated.")


class ErrorMechanism(ABC):
    def __init__(self: ErrorMechanism) -> None:
        raise NotInstantiableError

    @classmethod
    def sample(
        cls: type[ErrorMechanism],
        data: pd.DataFrame,
        error_rate: float,
        condition_to_column: Dtype | None = None,
        seed: int | None = None,
    ) -> np.array:
        if not isinstance(error_rate, float):
            raise TypeError  # FIXME

        if error_rate <= 0 or error_rate >= 1:
            raise ValueError  # FIXME

        if not (isinstance(seed, int) or seed is None):
            raise TypeError  # FIXME

        if not isinstance(data, pd.DataFrame):
            raise TypeError  # FIXME

        if data.empty:
            raise ValueError  # FIXME

        # At least two columns are necessary if we condition to another
        if condition_to_column is not None and len(data.columns) < 2:  # noqa: PLR2004
            raise ValueError  # FIXME

        return cls._sample(data=data, error_rate=error_rate, condition_to_column=condition_to_column, seed=seed)

    @staticmethod
    @abstractmethod
    def _sample(data: pd.DataFrame, error_rate: float, condition_to_column: Dtype | None = None, seed: int | None = None) -> pd.DataFrame:
        pass
