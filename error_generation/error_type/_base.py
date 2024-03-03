from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from error_generation.api import Column


class NotInstantiableError(Exception):
    def __init__(self: ErrorType) -> None:
        super().__init__("This class is not meant to be instantiated.")


class ErrorType(ABC):
    def __init__(self: ErrorType) -> None:
        raise NotInstantiableError

    @classmethod
    def apply(cls: type[ErrorType], table: pd.DataFrame, error_mask: pd.DataFrame, column: Column) -> pd.Series:
        cls._check_type(table, column)
        return cls._apply(table, error_mask, column)

    @staticmethod
    @abstractmethod
    def _check_type(table: pd.DataFrame, column: Column) -> None:
        pass

    @staticmethod
    @abstractmethod
    def _apply(table: pd.DataFrame, error_mask: pd.DataFrame, column: Column) -> pd.Series:
        pass
