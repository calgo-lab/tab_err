from __future__ import annotations

import random

import pandas as pd

from tab_err._utils import get_column

from ._error_type import ErrorType


class CategorySwap(ErrorType):
    """Simulate incorrect labels in a column that contains categorical values."""

    @staticmethod
    def _check_type(data: pd.DataFrame, column: int | str) -> None:
        series = get_column(data, column)

        if not isinstance(series.dtype, pd.CategoricalDtype):
            msg = f"Column {column} does not contain values of the Categorical dtype. Cannot insert Mislables.\n"
            msg += "Try casting the column to CategoricalDtype using df[column].astype('category')."
            raise TypeError(msg)

        if len(series.cat.categories) <= 1:
            msg = f"Column {column} contains {len(series.cat.categories)} categories. Require at least 2 categories to insert mislabels.."
            raise ValueError(msg)

    def _apply(self: CategorySwap, data: pd.DataFrame, error_mask: pd.DataFrame, column: int | str) -> pd.Series:
        series = get_column(data, column).copy()

        if self.config.mislabel_weighing == "uniform":

            def sample_label(old_label: pd.Any) -> pd.Any:
                choices = [x for x in series.cat.categories.to_numpy() if x != old_label]
                return random.choice(choices)

        elif self.config.mislabel_weighing == "frequency":

            def sample_label(old_label: pd.Any) -> pd.Any:
                se_sample = series.loc[series != old_label]
                return se_sample.sample(1, replace=True).to_numpy()[0]
        else:
            msg = "Invalid value for parameter 'config.mislabel_weighing'. Allowed values are: 'uniform', 'frequency'."
            raise ValueError(msg)

        series_mask = get_column(error_mask, column)
        series.loc[series_mask] = series.loc[series_mask].apply(sample_label)
        return series
