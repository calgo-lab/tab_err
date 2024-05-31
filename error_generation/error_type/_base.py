from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from error_generation.utils import ErrorTypeConfig

if TYPE_CHECKING:
    import pandas as pd


class ErrorType(ABC):
    def __init__(self: ErrorType, config: ErrorTypeConfig | dict | None = None) -> None:
        if config is None:
            self.config = ErrorTypeConfig()
        elif isinstance(config, dict):
            self.config = ErrorTypeConfig(**config)
        elif isinstance(config, ErrorTypeConfig):
            self.config = config
        else:
            msg = "config must be of type ErrorTypeConfig or dict"
            raise TypeError(msg)

    # TODO (seja): def apply(cls: type[ErrorType], table: pd.DataFrame, error_mask: pd.DataFrame, preserve_dtypes: bool = True)
    # -> tuple[pd.DataFrame, pd.DataFrame]:
    # 1. prüft parameters, sodass table.shape == error_mask.shape
    # 2. kopiert 'table'
    # 3. ruft '_get_valid_columns' auf um mögliche Spalten zu bekommen
    # 4. ruft '_apply' mit 'table[valid_columns]' auf um geänderte 'table' zu bekommen
    # 5. gibt gänderte 'table' und maske zurück, die anzeigt welche Zellen verändert wurden
    def apply(self: ErrorType, table: pd.DataFrame, error_mask: pd.DataFrame, column: str | int) -> pd.Series:
        self._check_type(table, column)
        return self._apply(table, error_mask, column)

    @staticmethod
    @abstractmethod
    # TODO (seja): def _get_valid_columns(table: pd.DataFrame, preserve_dtypes: bool = True) -> list[Dtype]:
    # Prüft auf welche columns dieser Fehler angewendet werden kann und gibt die entsprechenden Namen zurück.
    def _check_type(table: pd.DataFrame, column: str | int) -> None:
        pass

    @abstractmethod
    # TODO (seja): def _apply(table: pd.DataFrame, error_mask: pd.DataFrame) -> pd.DataFrame:
    # erwartet, dass 'table' ausschließlich valide columns hat. Wendet fehler stumpf auf alle Zellen an, wenn 'error_mask' True ist
    # Gibt geänderte 'table' zurück.
    def _apply(self: ErrorType, table: pd.DataFrame, error_mask: pd.DataFrame, column: str | int) -> pd.Series:
        pass
