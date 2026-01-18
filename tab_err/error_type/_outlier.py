from __future__ import annotations

import narwhals as nw
import numpy as np

from tab_err._utils import get_column, get_column_str, is_datetime_dtype, is_integer_dtype, is_numeric_dtype, select_numeric_or_datetime_columns

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
    def _check_type(data: nw.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not (is_numeric_dtype(series) or is_datetime_dtype(series)):
            msg = f"Column {column} with dtype: {series.dtype} does not contain numeric or datetime64 values. Cannot apply outliers."
            raise TypeError(msg)

    def _get_valid_columns(self: Outlier, data: nw.DataFrame) -> list[str | int]:
        """Returns all column names with numeric dtype elements."""
        return select_numeric_or_datetime_columns(data)

    def _apply(self: Outlier, data: nw.DataFrame, error_mask: nw.DataFrame, column: int | str) -> nw.Series:
        """Applies the Outlier ErrorType to a column of data.

        Args:
            data (nw.DataFrame): DataFrame containing the column to add errors to.
            error_mask (nw.DataFrame): A DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Returns:
            nw.Series: The data column, 'column', after Outlier errors at the locations specified by 'error_mask' are introduced.
        """
        col_name = get_column_str(data, column)
        series = get_column(data, column)
        series_mask = get_column(error_mask, column)
        was_datetime = False
        original_dtype = series.dtype

        # Get numpy arrays
        data_arr = series.to_numpy().copy()
        mask_arr = series_mask.to_numpy()

        if is_datetime_dtype(series):
            # Convert datetime to int64 (nanoseconds since epoch)
            data_arr = data_arr.astype("datetime64[ns]").astype("int64")
            was_datetime = True

        # Ensure float for calculations
        data_arr = data_arr.astype(np.float64)

        mean_value = np.nanmean(data_arr)
        q1 = np.nanquantile(data_arr, 0.25)
        q3 = np.nanquantile(data_arr, 0.75)
        iqr = q3 - q1

        upper_boundary = q3 + 1.5 * iqr
        lower_boundary = q1 - 1.5 * iqr

        # Pre-compute the perturbations
        perturbation_upper = self.config.outlier_coefficient * (upper_boundary - mean_value)
        perturbation_lower = self.config.outlier_coefficient * (mean_value - lower_boundary)

        is_integer = is_integer_dtype(series) and not was_datetime
        if is_integer:
            perturbation_upper = np.ceil(perturbation_upper)
            perturbation_lower = np.floor(perturbation_lower)

        # Get masks for the different outlier types depending on the mean
        mask_lower = (data_arr < mean_value) & mask_arr
        mask_upper = (data_arr > mean_value) & mask_arr
        mask_equal = (data_arr == mean_value) & mask_arr

        # Apply the constant perturbation to the respective mask
        data_arr[mask_lower] -= perturbation_lower
        data_arr[mask_upper] += perturbation_upper

        # Handle the mean values with a coin flip
        n_equal = np.sum(mask_equal)
        if n_equal > 0:
            coin_flips = self._random_generator.random(n_equal)
            perturbations = np.where(coin_flips > self.config.outlier_coin_flip_threshold, perturbation_upper, -perturbation_lower)
            data_arr[mask_equal] += perturbations

        # Apply Gaussian noise to simulate the increase in measurement error of the outliers
        noise_std = self.config.outlier_noise_coeff * iqr
        n_errors = np.sum(mask_arr)

        if is_integer:
            data_arr[mask_arr] += np.rint(self._random_generator.normal(loc=0, scale=noise_std, size=n_errors))
        else:
            data_arr[mask_arr] += self._random_generator.normal(loc=0, scale=noise_std, size=n_errors)

        if was_datetime:
            # Convert back to datetime
            data_arr = data_arr.astype("int64").astype("datetime64[ns]")
        elif is_integer:
            data_arr = data_arr.astype(np.int64)

        return nw.new_series(col_name, data_arr.tolist(), backend=nw.get_native_namespace(data))
