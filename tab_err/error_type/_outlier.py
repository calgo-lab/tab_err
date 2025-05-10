from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_numeric_dtype

from tab_err._utils import get_column

from ._error_type import ErrorType


class Outlier(ErrorType):
    """Inserts outliers into a column by adding/subtracting (k * iqr + noise) to the median of the given column.

    Determines if an outlier is above or below the median by tossing a coin for each row to be errored.
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

        # Set up the necessary values
        median_value = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)
        if iqr == 0:  # To not impute the median +/- noise
            iqr = 1e-9


        # Decide which outliers are above/below the median - at least one is above/below
        coin_tosses = self._random_generator.random(series_mask.sum()) < self.config.outlier_coin_flip_threshold
        if series_mask.sum() > 1:
            if not coin_tosses.any():
                coin_tosses[self._random_generator.integers(0, len(coin_tosses))] = True
            elif coin_tosses.all():
                coin_tosses[self._random_generator.integers(0, len(coin_tosses))] = False

        neg_outliers = series_mask.copy()
        neg_outliers[series_mask] = coin_tosses
        pos_outliers = series_mask & ~neg_outliers

        neg_noise = self._random_generator.normal(
            loc=0, scale=self.config.outlier_noise_coeff*iqr, size=neg_outliers.sum()) if neg_outliers.sum() > 0 else np.array([])
        pos_noise = self._random_generator.normal(
            loc=0, scale=self.config.outlier_noise_coeff*iqr, size=pos_outliers.sum()) if pos_outliers.sum() > 0 else np.array([])

        # Apply outliers
        if neg_noise.size > 0:
            series[neg_outliers] = median_value - (self.config.outlier_coefficient * iqr) - neg_noise
        if pos_noise.size > 0:
            series[pos_outliers] = median_value + (self.config.outlier_coefficient * iqr) + pos_noise

        if was_datetime:  # Handle datetime objects
            series = series.clip(lower=pd.Timestamp.min.value, upper=pd.Timestamp.max.value)
            series = pd.to_datetime(series)

        return series
