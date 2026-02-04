from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import narwhals as nw

from tab_err._utils import check_data_emptiness, check_error_rate, create_empty_boolean_mask, set_column

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from tab_err._error_model import ErrorModel


@dataclasses.dataclass
class MidLevelConfig:
    """Configuration of the mid_level API.

    The mid_level API applies N pairs of (error_mechanism, error_type) to data. Consequently, the user
    is required to specify up to N pairs of error_mechanism, error_type per column when calling the mid_level
    API.

    Attributes:
        columns (dict[int | str, list[ErrorModel]]): A dictionary mapping from columns to a list of `ErrorModel`s that should be applied
    """

    columns: dict[int | str, list[ErrorModel]]

    def to_dict(self: MidLevelConfig) -> dict[str, Any]:
        """Serializes the MidLevelConfig to a dict."""
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> MidLevelConfig:
        """Deserializes the MidLevelConfig from a dict."""
        return MidLevelConfig(**data)


def create_errors(data: IntoDataFrame, config: MidLevelConfig | dict) -> tuple[IntoDataFrame, IntoDataFrame]:
    """Creates errors in a given DataFrame, following a user-defined configuration.

    Args:
        data (IntoDataFrame): The DataFrame to create errors in. Supports pandas, Polars, and other narwhals-compatible backends.
        config (MidLevelConfig | dict): The configuration for the error generation process.

    Returns:
        tuple[IntoDataFrame, IntoDataFrame]:
            - The first element is a copy of 'data' with errors.
            - The second element is the associated error mask.
            Both are returned in the same format as the input data.

    Raises:
        TypeError: If `config` has incorrect type.
    """
    # Wrap native DataFrame to narwhals
    data_nw = nw.from_native(data, eager_only=True)

    check_data_emptiness(data_nw)
    if isinstance(config, dict):
        _config = MidLevelConfig(config)

    elif isinstance(config, MidLevelConfig):
        _config = config

    else:
        msg = f"The type of 'config' must be either MidLevelConfig or dict but was {type(config)}."
        raise TypeError(msg)

    data_dirty = data_nw.clone()
    error_mask = create_empty_boolean_mask(data_nw)

    for column in _config.columns:
        for error_model in _config.columns[column]:
            check_error_rate(error_model.error_rate)

            error_mechanism = error_model.error_mechanism
            error_type = error_model.error_type
            error_rate = error_model.error_rate

            old_error_mask = error_mask.clone()
            error_mask = error_mechanism.sample(data_nw, column, error_rate, error_mask)

            # Compute new error positions (where mask changed from False to True)
            old_mask_col = old_error_mask[column if isinstance(column, str) else data_nw.columns[column]]
            new_mask_col = error_mask[column if isinstance(column, str) else data_nw.columns[column]]

            # Create a mask for just the new errors
            new_errors_arr = new_mask_col.to_numpy() & ~old_mask_col.to_numpy()
            col_name = column if isinstance(column, str) else data_nw.columns[column]
            new_errors_mask = old_error_mask.with_columns(nw.new_series(col_name, new_errors_arr.tolist(), backend=nw.get_native_namespace(old_error_mask)))

            series = error_type.apply(data_dirty, new_errors_mask, column)
            data_dirty = set_column(data_dirty, column, series)

    # Return in the original format
    return nw.to_native(data_dirty), nw.to_native(error_mask)
