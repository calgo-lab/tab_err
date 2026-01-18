from __future__ import annotations

import warnings

import narwhals as nw
import numpy as np

from tab_err._utils import check_error_rate, get_column, get_column_str

from ._error_mechanism import ErrorMechanism


class ECAR(ErrorMechanism):
    """ErrorMechanism subclass implementing the 'Erroneous Completely At Random' error mechanism.

    Description:
        Errors are assumed to be completely independent of the data distribution
    """

    def _sample(
        self: ECAR,
        data: nw.DataFrame,  # noqa: ARG002
        column: str | int,
        error_rate: float,
        error_mask: nw.DataFrame,
    ) -> nw.DataFrame:
        """Creates an error mask according to the 'Erroneous Completely At Random' error mechanism.

        Description:
            Cells are chosen uniform randomly by a NumPy random number generator

        Args:
            data (nw.DataFrame): DataFrame containing the column to add errors to
            column (str | int): The column of 'data' to create an error mask for
            error_rate (float): Proportion of rows to be affected by errors; in range [0,1]
            error_mask (nw.DataFrame): A DataFrame with the same index & columns as 'data' that will be modified and returned

        Raises:
            ValueError: If there are insufficient entries to add errors to with respect to the error rate, a ValueError will be returned

        Returns:
            nw.DataFrame: A DataFrame with True values at entries where an error should be introduced, False otherwise
        """
        check_error_rate(error_rate)
        col_name = get_column_str(error_mask, column)
        se_mask = get_column(error_mask, column)

        # Get indices where mask is False (error-free cells)
        mask_arr = se_mask.to_numpy()
        error_free_indices = np.where(~mask_arr)[0]

        if self.condition_to_column is not None:
            warnings.warn("'condition_to_column' is set but will be ignored by ECAR.", stacklevel=1)

        n_errors = int(len(mask_arr) * error_rate)

        if len(error_free_indices) < n_errors:
            msg = f"The error rate of {error_rate} requires {n_errors} error-free cells. "
            msg += f"However, only {len(error_free_indices)} error-free cells are available."
            raise ValueError(msg)

        # Uniform randomly choose error-cells
        error_indices = self._random_generator.choice(error_free_indices, n_errors, replace=False)

        # Create new mask array with selected indices set to True
        new_mask_arr = mask_arr.copy()
        new_mask_arr[error_indices] = True

        # Update the error_mask DataFrame
        return error_mask.with_columns(nw.new_series(col_name, new_mask_arr.tolist(), backend=nw.get_native_namespace(error_mask)))
