from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ._base import ErrorMechanism

if TYPE_CHECKING:
    from pandas._typing import Dtype


class ECAR(ErrorMechanism):
    @staticmethod
    def _sample(data: pd.DataFrame, error_rate: float, condition_to_column: Dtype | None = None, seed: int | None = None) -> pd.DataFrame:
        # TODO: warning when condition_to_column is used

        how_many_error_cells = int(data.size * error_rate)
        error_mask = pd.DataFrame(data=False, index=data.index, columns=data.columns)

        # randomly choose cells as errors
        error_indices = np.random.default_rng(seed=seed).choice(data.size, how_many_error_cells, replace=False)
        error_mask.to_numpy().ravel()[error_indices] = True
        return error_mask
