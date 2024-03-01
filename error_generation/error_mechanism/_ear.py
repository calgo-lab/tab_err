from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pandas._typing import Dtype
from ._base import ErrorMechanism


class EAR(ErrorMechanism):
    @staticmethod
    def _sample(data: pd.DataFrame, error_rate: float, condition_to_column: Dtype | None = None, seed: int | None = None) -> pd.DataFrame:
        #  TODO pro spalte blockweise fehler erzeugen.
        how_many_error_cells = int(data.size * error_rate)
        error_mask = pd.DataFrame(data=False, index=data.index, columns=data.columns)

        # TODO: warning, wenn zu vielel Fehler fuer die uebrigen spalten erzeugt werden sollen

        # TODO: anzahl zellen auf zeilen umrechnen in abhaengigkeit von der anzahl spalten
        lower_error_index = np.random.default_rng(seed=seed).integers(0, len(data) - how_many_error_cells)
        error_index_range = range(lower_error_index, lower_error_index + how_many_error_cells)

        # exclude condition_to_column
        error_indices = np.random.default_rng(seed=seed).choice(data.size - len(data), how_many_error_cells, replace=False)

        data[condition_to_column].sort_values().iloc[error_index_range].index

        return error_mask
