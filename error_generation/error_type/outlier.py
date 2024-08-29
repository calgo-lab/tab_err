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
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        # Determine boundaries
        upper_boundary = Q3 + 1.5 * IQR
        lower_boundary = Q1 - 1.5 * IQR
        
        # Precompute the perturbations based on the mean and boundaries
        perturbation_upper = self.config.outlier_coefficient * (upper_boundary - mean_value)
        perturbation_lower = self.config.outlier_coefficient * (mean_value - lower_boundary)
        
        # Apply outliers based on value relative to mean
        for idx in series.loc[series_mask].index:
            if series[idx] < mean_value:
                # Push towards lower outlier
                series[idx] -= perturbation_lower
            elif series[idx] > mean_value:
                # Push towards upper outlier
                series[idx] += perturbation_upper
            else:
                # Value is equal to the mean, use a coin flip
                if np.random.rand() > 0.5:
                    series[idx] += perturbation_upper
                else:
                    series[idx] -= perturbation_lower
        
        # Apply Gaussian noise
        noise_std = self.config.outlier_noise_coeff * IQR
        series.loc[series_mask] += np.random.normal(loc=0, scale=noise_std, size=series_mask.sum())
        
        return series
