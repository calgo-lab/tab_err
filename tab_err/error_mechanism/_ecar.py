from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from tab_err._utils import get_column

from ._error_mechanism import ErrorMechanism

if TYPE_CHECKING:
    import pandas as pd


class ECAR(ErrorMechanism):
    # TODO(seja): Docs
    def _sample(self: ECAR, data: pd.DataFrame, column: str | int, error_rate: float, error_mask: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG002
        se_mask = get_column(error_mask, column)
        se_mask_error_free = se_mask[~se_mask]

        if self.condition_to_column is not None:
            warnings.warn("'condition_to_column' is set but will be ignored by ECAR.", stacklevel=1)

        n_errors = int(se_mask.size * error_rate)

        if len(se_mask_error_free) < n_errors:
            msg = f"The error rate of {error_rate} requires {n_errors} error-free cells. "
            msg += f"However, only {len(se_mask_error_free)} error-free cells are available."
            raise ValueError(msg)

        # randomly choose error-cells
        error_indices = self._random_generator.choice(se_mask_error_free.index, n_errors, replace=False)
        se_mask[error_indices] = True
        return error_mask
