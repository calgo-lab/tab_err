from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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
    ) -> pd.DataFrame:
        error_rate_msg = "'error_rate' need to be float: 0 <= error_rate <= 1."
        if error_rate < 0 or error_rate > 1:
            raise ValueError(error_rate_msg)

        if not (isinstance(seed, int) or seed is None):
            msg = "'seed' need to be int or None."
            raise TypeError(msg)

        data_msg = "'data' need to be non-empty DataFrame."
        if not isinstance(data, pd.DataFrame):
            raise TypeError(data_msg)

        if data.empty:
            raise ValueError(data_msg)

        # At least two columns are necessary if we condition to another
        if condition_to_column is not None and len(data.columns) < 2:  # noqa: PLR2004
            msg = "'data' need at least 2 columns if 'condition_to_column' is given."
            raise ValueError(msg)

        return cls._sample(data=data, error_rate=error_rate, condition_to_column=condition_to_column, seed=seed)

    @staticmethod
    @abstractmethod
    def _sample(data: pd.DataFrame, error_rate: float, condition_to_column: Dtype | None = None, seed: int | None = None) -> pd.DataFrame:
        pass
