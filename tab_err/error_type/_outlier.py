from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_integer_dtype, is_numeric_dtype

from tab_err._utils import get_column

from ._error_type import ErrorType


class Outlier(ErrorType):
    """Inserts outliers into a column by pushing data points outside the interquartile range (IQR) boundaries.

    - Data points below the mean are pushed towards lower outliers, while those above the mean are pushed towards upper outliers.
    - The `outlier_coefficient` controls how far values are pushed relative to the IQR. An `outlier_coefficient` of 1.0 means the
    push is equal to half of the IQR, shifting the mean value exactly to the edge of the IQR. Values that deviate more from the
    mean will be pushed beyond the IQR boundary. When `outlier_coefficient` is less than 1.0, values—including the mean—are pushed
    less drastically, potentially keeping them within the IQR.
    - The push is calculated as:
        push = outlier_coefficient * |upper_boundary - mean_value|
    - Values above the mean are pushed towards the upper boundary, and values below the mean are pushed towards the lower boundary.
    If a value equals the mean, a coin flip decides whether it is pushed towards the upper or lower boundary.
    - After this process, Gaussian noise is added to simulate measurement errors and make the outliers appear more realistic. The
    amount of noise can be controlled via the `outlier_noise_coeff` parameter and is scaled with the IQR to ensure it is proportional
    to the data's spread.
    """

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not (is_numeric_dtype(series) or is_datetime64_dtype(series)):
            msg = f"Column {column} with dtype: {series.dtype} does not contain numeric or datetime64 values. Cannot apply outliers."
            raise TypeError(msg)

    def _get_valid_columns(self: Outlier, data: pd.DataFrame) -> list[str | int]:
        """Returns all column names with numeric dtype elements."""
        return data.select_dtypes(include=["number", "datetime64"]).columns.tolist()

    def _apply(self: Outlier, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        """Applies the Outlier ErrorType to a column of data.

        Args:
            data (pd.DataFrame): DataFrame containing the column to add errors to.
            error_mask (pd.DataFrame): A Pandas DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Returns:
            pd.Series: The data column, 'column', after Outlier errors at the locations specified by 'error_mask' are introduced.
        """
        # Get the column series and mask
        series = get_column(data, column).copy()
        series_mask = get_column(error_mask, column)
        was_datetime = False  # Default to false -- changes to code only occur if the series is datetime

        if is_datetime64_dtype(series):  # Convert to int if datetime (ns since UNIX epoch) -- We need to add robustness against intmax/floatmax
            series = series.astype("int64")
            was_datetime = True

        mean_value = series.mean()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        upper_boundary = q3 + 1.5 * iqr
        lower_boundary = q1 - 1.5 * iqr

        # Pre-compute the perturbations
        perturbation_upper = self.config.outlier_coefficient * (upper_boundary - mean_value)
        perturbation_lower = self.config.outlier_coefficient * (mean_value - lower_boundary)

        if is_integer_dtype(series):  # round float to int when series is int
            perturbation_upper = np.ceil(perturbation_upper)
            perturbation_lower = np.floor(perturbation_lower)

        # Get masks for the different outlier types depending on the mean
        mask_lower = (series < mean_value) & series_mask
        mask_upper = (series > mean_value) & series_mask
        mask_equal = (series == mean_value) & series_mask

        # Apply the constant perturbation to the respective mask
        series.loc[mask_lower] -= perturbation_lower
        series.loc[mask_upper] += perturbation_upper

        # Handle the mean values with a coin flip
        coin_flips = self._random_generator.random(mask_equal.sum())
        series.loc[mask_equal] += np.where(coin_flips > self.config.outlier_coin_flip_threshold, perturbation_upper, -perturbation_lower)

        # Apply Gaussian noise to simulate the increase in measurement error of the outliers
        noise_std = self.config.outlier_noise_coeff * iqr

        if is_integer_dtype(series):  # round float to int when series is int
            series.loc[series_mask] += np.rint(self._random_generator.normal(loc=0, scale=noise_std, size=series_mask.sum()))
        else:
            series.loc[series_mask] += self._random_generator.normal(loc=0, scale=noise_std, size=series_mask.sum())

        if was_datetime:  # Handle datetime objects
            series = pd.to_datetime(series)

        return series
