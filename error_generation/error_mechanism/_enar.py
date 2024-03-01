from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._base import ErrorMechanism

if TYPE_CHECKING:
    from pandas._typing import Dtype


class ENAR(ErrorMechanism):
    def _sample(data: pd.DataFrame, error_rate: float, condition_to_column: Dtype | None = None, seed: int | None = None) -> np.array:
        # TODO: warning when condition_to_column is used
        return np.array("ENAR")
