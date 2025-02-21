from __future__ import annotations

import dataclasses
from typing import Any, Callable


@dataclasses.dataclass
class ErrorTypeConfig:
    """Parameters that describe the error type.

    Arguments that are specific to the error type. Most error types do not share the same arguments, which
    is why there are many attributes of this dataclass that are mostly default values.

    Attributes:
        encoding_sender (str | None, optional): When creating Mojibake, used to encode strings to bytes. Defaults to None.

        encoding_receiver (str | None, optional): When creating Mojibake, used to decode bytes back to strings. Defaults to None.

        typo_keyboard_layout (str): When using Typo, the keyboard layout used by the typer. Defaults to "ansi-qwerty".

        typo_error_period (int): When using Typo, the period at which the error occurs. Defaults to 10.

        missing_value (str | None, optional): Token used to indicate missing values in Pandas. Defaults to None.

        mislabel_weighing (str): Weight of the distribution that mislables are drawn from. Either "uniform" or "frequency". Defaults to "uniform"

        mislabel_weights (dict[Any, float] | None): _description_. Defaults to None.

        mistype_dtype (str | None): dtype of the column that is incorrectly typed. One of "object", "string", "int64", "Int64", "float64", "Float64".
            Defaults to None.

        wrong_unit_scaling (Callable | None): Function that scales a value from one unit to another. Defaults to None.

        permutation_separator (str): A Char that separates structured text, e.g. ' ' in an address or '-' in a date. Defaults to " ".

        permutation_automation_pattern (str): Permutations either all follow the same pattern (fixed) or not (random). Defaults to "random"

        permutation_pattern (list[int] | None): Manually specify the pattern which the permutations follow. Overwrite automation patterns if set.
            Defaults to None.

        extraneous_value_template (str | None): Template string used to add extraneous data to the value. The position of the value is indicated by the template
            string '{value}'. Defaults to None.

        replace_what (str | None): String that the Replace Error Type replaces with replace_with. Defaults to None.

        replace_with (str): String that the Replace Error Type uses to replace replace_what with. Defaults to "".

        add_delta_value (Any | None): Value that is added to the value by the AddDelta Error Type. Defaults to None.

        outlier_coin_flip_threshold (float): Coin flip determines the direction (positive, negative) of the outlier. Defaults to 0.5.

        outlier_coefficient (float): Coefficient that determines the magnitude of the outliers for the Outlier Error Type. Defaults to 1.0.

        outlier_noise_coeff (float): Coefficient that influences the standard deviation of the noise added to the outliers for the Outlier Error Type.
            Defaults to 0.1.
    """

    encoding_sender: str | None = None
    encoding_receiver: str | None = None

    typo_keyboard_layout: str = "ansi-qwerty"
    typo_error_period: int = 10

    missing_value: str | None = None

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

    add_delta_value: float | int | None = None

    outlier_coin_flip_threshold: float = 0.5
    outlier_coefficient: float = 1.0
    outlier_noise_coeff: float = 0.1

    def to_dict(self: ErrorTypeConfig) -> dict[str, Any]:
        """Serializes the ErrorTypeConfig to a dict."""
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ErrorTypeConfig:
        """Deserializes the ErrorTypeConfig from a dict."""
        return ErrorTypeConfig(**data)
