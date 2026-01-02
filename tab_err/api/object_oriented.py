from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

from tab_err.api.high_level import create_errors, create_errors_with_config

if TYPE_CHECKING:
    from typing import Self

    import pandas as pd

    from tab_err import ErrorMechanism, ErrorType
    from tab_err.api import MidLevelConfig


class ErrorInjector:
    """Object-oriented wrapper around the high-level error creation API.

    This class allows:
    - Reproducible error injection
    - Access to the configuration used
    - Serialization of the configuration for later reuse
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        error_rate: float,
        n_error_models_per_column: int = 1,
        error_types_to_include: list[ErrorType] | None = None,
        error_types_to_exclude: list[ErrorType] | None = None,
        error_mechanisms_to_include: list[ErrorMechanism] | None = None,
        error_mechanisms_to_exclude: list[ErrorMechanism] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the ErrorInjector object."""
        self._error_rate = error_rate
        self._n_error_models_per_column = n_error_models_per_column
        self._error_types_to_include = error_types_to_include
        self._error_types_to_exclude = error_types_to_exclude
        self._error_mechanisms_to_include = error_mechanisms_to_include
        self._error_mechanisms_to_exclude = error_mechanisms_to_exclude
        self._seed = seed

        self._last_config: MidLevelConfig | None = None

    @property
    def config(self) -> MidLevelConfig:
        """Return the configuration used in the most recent error creation."""
        if self._last_config is None:
            msg = "No configuration available. Call apply_with_config() first."
            raise RuntimeError(msg)
        return self._last_config

    def apply(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply errors to a DataFrame without retaining the configuration."""
        dirty_data, error_mask = create_errors(
            data=data,
            error_rate=self._error_rate,
            n_error_models_per_column=self._n_error_models_per_column,
            error_types_to_include=self._error_types_to_include,
            error_types_to_exclude=self._error_types_to_exclude,
            error_mechanisms_to_include=self._error_mechanisms_to_include,
            error_mechanisms_to_exclude=self._error_mechanisms_to_exclude,
            seed=self._seed,
        )
        return dirty_data, error_mask

    def apply_with_config(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply errors to a DataFrame and store the configuration used."""
        dirty_data, error_mask, config = create_errors_with_config(
            data=data,
            error_rate=self._error_rate,
            n_error_models_per_column=self._n_error_models_per_column,
            error_types_to_include=self._error_types_to_include,
            error_types_to_exclude=self._error_types_to_exclude,
            error_mechanisms_to_include=self._error_mechanisms_to_include,
            error_mechanisms_to_exclude=self._error_mechanisms_to_exclude,
            seed=self._seed,
        )
        self._last_config = config
        return dirty_data, error_mask

    def save_config(self, path: str | Path) -> None:
        """Serialize the last-used configuration to disk."""
        with Path(path).open("wb") as f:
            pickle.dump(self.config, f)

    @classmethod
    def from_config(cls, config: MidLevelConfig) -> Self:
        """Create an injector that reuses an existing configuration."""
        injector = cls(error_rate=0.0)
        injector._last_config = config
        return injector
