from __future__ import annotations

import string
import warnings
from typing import TYPE_CHECKING

from tab_err._utils import get_column, new_series_like, select_string_columns

from ._error_type import ErrorType

if TYPE_CHECKING:
    import narwhals as nw

class Extraneous(ErrorType):
    """Adds Extraneous strings around the values in a column."""

    def _generate_value_template_string(self: Extraneous, min_n: int = 0, max_n: int = 2) -> str:
        """Generates the value template string. Prepends and appends a random number of characters to the {value} string."""
        n1 = self._random_generator.integers(min_n + 1, max_n + 1)  # Random number of characters - guaranteed one
        n2 = self._random_generator.integers(min_n, max_n + 1)  # Random number of characters
        valid_chars = [char for char in string.punctuation if char not in "{}"]  # Exclude { and }
        prepend = "".join(self._random_generator.choice(valid_chars, size=n1))
        append = "".join(self._random_generator.choice(valid_chars, size=n2))
        return prepend + r"{value}" + append

    @staticmethod
    def _check_type(data: nw.DataFrame, column: int | str) -> None:
        # all data types are fine
        pass

    def _get_valid_columns(self: Extraneous, data: nw.DataFrame) -> list[str | int]:
        """Returns all column names with string dtype elements. Necessary for high level API."""
        return select_string_columns(data)

    def _apply(self: Extraneous, data: nw.DataFrame, error_mask: nw.DataFrame, column: int | str) -> nw.Series:
        """Applies the Extraneous ErrorType to a column of data.

        Args:
            data (nw.DataFrame): DataFrame containing the column to add errors to.
            error_mask (nw.DataFrame): A DataFrame with the same index & columns as 'data' that will be modified and returned.
            column (int | str): The column of 'data' to create an error mask for.

        Raises:
            ValueError: If extraneous_value_template does not contain the placeholder value, a ValueError will be thrown.

        Returns:
            nw.Series: The data column, 'column', after Extraneous errors at the locations specified by 'error_mask' are introduced.
        """
        series = get_column(data, column)
        series_mask = get_column(error_mask, column)

        # Get numpy arrays
        data_arr = series.to_numpy().copy()
        mask_arr = series_mask.to_numpy()

        if self.config.extraneous_value_template is None:
            msg = "self.config.extraneous_value_template is not set. Choosing a random string augmentation."
            warnings.warn(msg, stacklevel=2)
            self.config.extraneous_value_template = self._generate_value_template_string()

        if "{value}" not in self.config.extraneous_value_template:
            msg = f"The extraneous template {self.config.extraneous_value_template} does not contain the placeholder "
            msg += "{value}. Please add it for a valid format."
            raise ValueError(msg)

        # Apply extraneous template where mask is True
        for i in range(len(data_arr)):
            if mask_arr[i]:
                val = data_arr[i]
                data_arr[i] = self.config.extraneous_value_template.format(value=val)

        return new_series_like(data, column, data_arr)
