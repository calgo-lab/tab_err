from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import pandas as pd

    from error_generation.error_mechanism import ErrorMechanism
    from error_generation.error_type import ErrorType


@dataclasses.dataclass
class ErrorModel:
    """An ErrorModel, which consists of an ErrorMechanis, an ErrorType, and an error rate."""

    error_mechanism: ErrorMechanism
    error_type: ErrorType
    error_rate: float


@dataclasses.dataclass
class MidLevelConfig:
    """Configuration of the mid_level API.

    The mid_level API applies N pairs of (error_mechanism, error_type) to a table. In consequence, the user
    is required to specify up to N pairs of error_mechanism, error_type per column when calling the mid_level
    API.
    """

    columns: dict[int | str, list[ErrorModel]]


@dataclasses.dataclass
class ErrorTypeConfig:
    """Parameters that describe the error type.

    Arguments that are specific to the error type. Most error types do not share the same arguments, which
    is why there are many attributes of this dataclass that are mostly default values.

    Args:
        encoding_sender: When creating Mojibake, used to encode strings to bytes.
        encoding_receiver: When creating Mojibake, used to decode bytes back to strings.
        keyboard_layout: When using Butterfinger, the keyboard layout used by the typer.
        error_period: When using Butterfinger, the period at which the error occurs.
        na_value: Token used to indicate missing values in Pandas.
        mislabel_weighing: Weight of the distribution that mislables are drawn from. Either "uniform", "frequency" or "custom".
        mistype_dtype: dtype of the column that is incorrectly typed. One of "object", "string", "int64", "Int64", "float64", "Float64".
        wrong_unit_scaling: Function that scales a value from one unit to another.
        permutation_separator: A Char that separates structured text, e.g. ' ' in an address or '-' in a date.
        permutation_automation_pattern: Permutations either all follow the same pattern (fixed) or not (random).
        permutation_pattern: Manually specify the pattern which the permutations follow. Overwrite automation patterns if set.
        extraneous_value_template: Template string used to add extraneous data to the value. The position of the value is indicated by the template string
        '{value}'.
        replace_what: String that the Replace Error Type replaces with replace_with.
        replace_with: String that the Replace Error Type uses to replace replace_what with. Defaults to "".
        add_delta_value: Value that is added to the value by the AddDelta Error Type.
    """

    encoding_sender: str | None = None
    encoding_receiver: str | None = None

    keyboard_layout: str = "ansi-qwerty"
    error_period: int = 10

    na_value: str | None = None

    mislabel_weighing: str = "uniform"
    mislabel_weights: dict[Any, float] | None = None

    mistype_dtype: str | None = None

    wrong_unit_scaling: Callable | None = None

    permutation_separator: str = " "
    permutation_automation_pattern: str = "random"
    permutation_pattern: list[int] | None = None

    extraneous_value_template: str | None = None

    replace_what: str | None = None
    replace_with: str = ""

    add_delta_value: Any | None = None

    def to_dict(self: ErrorTypeConfig) -> dict[str, Any]:
        """Serializes the ErrorTypeConfig to a dict."""
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ErrorTypeConfig:
        """Deserializes the ErrorTypeConfig from a dict."""
        return ErrorTypeConfig(**data)


def set_column(table: pd.DataFrame, column: int | str, series: pd.Series) -> pd.Series:
    """Replaces a column in a dataframe with a series.

    Mutates table and changes the dtype of the original table to that of the series,
    which, depending on the error type, might change.
    """
    col = table.columns[column] if isinstance(column, int) else column
    table[col] = table[col].astype(series.dtype)
    table[col] = series
    return table


def get_column_str(table: pd.DataFrame, column: int | str) -> str:
    """Return the a column's name from the available names of columns of a dataframe."""
    if isinstance(column, int):
        col = table.columns[column]
    elif isinstance(column, str):
        col = column
    else:
        msg = f"Column must be an int or str, not {type(column)}"
        raise TypeError(msg)
    return col


def get_column(table: pd.DataFrame, column: int | str) -> pd.Series:
    """Selects a column from a dataframe and returns it as a series."""
    return table[get_column_str(table, column)]
