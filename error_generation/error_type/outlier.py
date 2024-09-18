from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pandas.api.types import is_numeric_dtype

from error_generation.error_type import ErrorType
from error_generation.utils import get_column

if TYPE_CHECKING:
    import pandas as pd

class Outlier(ErrorType):
    """Simulate outliers in a column by pushing data points beyond the IQR-based boundaries.

    Data points below the mean are pushed towards lower outliers, and data points above the mean are pushed towards upper outliers.
    The magnitude of the push is controlled by outlier_coefficient. Gaussian noise is added to introduce variability.
    """

    @staticmethod
    def _check_type(table: pd.DataFrame, column: int | str) -> None:
        series = get_column(table, column)

        if not is_numeric_dtype(series):
            msg = f"Column {column} does not contain numeric values. Cannot apply outliers."
            raise TypeError(msg)

    def _apply(self: Outlier, table: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        # Get the column series and mask
        series = get_column(table, column).copy()
        series_mask = get_column(error_mask, column)

        # Calculate mean and IQR
        mean_value = series.mean()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        # Determine boundaries
        upper_boundary = q3 + 1.5 * iqr
        lower_boundary = q1 - 1.5 * iqr

        # Precompute the perturbations based on the mean and boundaries
        perturbation_upper = self.config.outlier_coefficient * (upper_boundary - mean_value)
        perturbation_lower = self.config.outlier_coefficient * (mean_value - lower_boundary)

        # Get masks for the different outlier types depending on the mean
        mask_lower = (series < mean_value) & series_mask
        mask_upper = (series > mean_value) & series_mask
        mask_equal = (series == mean_value) & series_mask

        # Apply perturbations
        series.loc[mask_lower] -= perturbation_lower
        series.loc[mask_upper] += perturbation_upper

        # Random number generator
        rng = np.random.default_rng()
        coin_flip_threshold = 0.5

        # Handle the mean values with a coin flip
        coin_flips = rng.random(mask_equal.sum())
        series.loc[mask_equal] += np.where(coin_flips > coin_flip_threshold, perturbation_upper, -perturbation_lower)

        # Apply Gaussian noise
        noise_std = self.config.outlier_noise_coeff * iqr
        series.loc[series_mask] += rng.normal(loc=0, scale=noise_std, size=series_mask.sum())

        return series
