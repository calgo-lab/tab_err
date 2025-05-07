from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from tab_err._utils import check_error_rate, get_column

from ._error_mechanism import ErrorMechanism

if TYPE_CHECKING:
    import pandas as pd


class ECAR(ErrorMechanism):
    """ErrorMechanism subclass implementing the 'Erroneous Completely At Random' error mechanism.

    Description:
        Errors are assumed to be completely independent of the data distribution
    """

    def _sample(
        self: ECAR,
        data: pd.DataFrame,  # noqa: ARG002
        column: str | int,
        error_rate: float,
        error_mask: pd.DataFrame,
    ) -> pd.DataFrame:
        """Creates an error mask according to the 'Erroneous Completely At Random' error mechanism.

        Description:
            Sells are chosen uniform randomly by a NumPy random number generator

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to
            column (str | int): The column of 'data' to create an error mask for
            error_rate (float): Proportion of rows to be affected by errors; in range [0,1]
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned

        Raises:
            ValueError: If there are insufficient entries to add errors to with respect to the error rate, a ValueError will be returned

        Returns:
            pd.DataFrame: A Pandas DataFrame with True values at entries where an error should be introduced, False otherwise
        """
        check_error_rate(error_rate)
        se_mask = get_column(error_mask, column)
        se_mask_error_free = se_mask[~se_mask]

        if self.condition_to_column is not None:
            warnings.warn("'condition_to_column' is set but will be ignored by ECAR.", stacklevel=1)

        n_errors = int(se_mask.size * error_rate)

        if len(se_mask_error_free) < n_errors:
            msg = f"The error rate of {error_rate} requires {n_errors} error-free cells. "
            msg += f"However, only {len(se_mask_error_free)} error-free cells are available."
            raise ValueError(msg)

        # Uniform randomly choose error-cells
        error_indices = self._random_generator.choice(se_mask_error_free.index, n_errors, replace=False)
        se_mask[error_indices] = True
        return error_mask
