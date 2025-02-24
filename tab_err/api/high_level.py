from __future__ import annotations

import warnings

import pandas as pd

from tab_err import ErrorMechanism, ErrorType, error_mechanism, error_type
from tab_err._error_model import ErrorModel
from tab_err._utils import check_data_emptiness, check_error_rate
from tab_err.api import MidLevelConfig, mid_level


def _are_same_class(obj1: object, obj2: object) -> bool:
    """Checks if two objects are of the same class.

    Args:
        obj1 (object): The first object to compare.
        obj2 (object): The second object to compare.

    Returns:
        bool: True if both objects are of the same class, False otherwise.
    """
    return isinstance(obj1, obj2.__class__) and isinstance(obj2, obj1.__class__)


def _are_same_error_mechanism(error_mechanism1: ErrorMechanism, error_mechanism2: ErrorMechanism) -> bool:
    """Checks if two error mechanisms are the same class and have the same class variables.

    If the second error mechanism has a None value for the condition to column, they are deemed to be the same.
    """
    return type(error_mechanism1) is type(error_mechanism2) and ((error_mechanism1.condition_to_column == error_mechanism2.condition_to_column and error_mechanism2.condition_to_column is not None) or (error_mechanism2.condition_to_column is None))


def _build_column_type_dictionary(
    data: pd.DataFrame, error_types_to_include: list[ErrorType] | None = None, error_types_to_exclude: list[ErrorType] | None = None, seed: int | None = None
) -> dict[int | str, list[ErrorType]]:
    """Creates a dictionary mapping from column names to the list of valid error types to apply to that column.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.
        error_types_to_include (list[ErrorType] | None, optional): A list of the error types to be included when building error models. Defaults to None.
        error_types_to_exclude (list[ErrorType] | None, optional): A list of the error types to be excluded when building error models. Defaults to None.
            When both error_types_to_include and error_types_to_exclude are none, the maximum number of default error types will be used.
            At least one must be None or an error will occur.
        seed (int | None, optional): Random seed. Defaults to None.

    Raises:
        ValueError: If error_types_to_exclude is not None and error_types_to_include is not None, a ValueError is thrown.
        ValueError: If error_types_to_exclude is None and error_types_to_include is not None and len(error_types_to_include) == 0, a ValueError is thrown.

    Returns:
        dict[int | str, list[ErrorModel]]: A dictionary mapping from column names to the list of valid error types to apply to that column.
    """
    error_types_applied = [
        error_type.AddDelta(seed=seed),
        error_type.CategorySwap(seed=seed),
        error_type.Extraneous(seed=seed),
        error_type.Mojibake(seed=seed),
        error_type.Outlier(seed=seed),
        error_type.Replace(seed=seed),
        error_type.Typo(seed=seed),
        error_type.WrongUnit(seed=seed),
        error_type.MissingValue(seed=seed),
    ]

    if error_types_to_exclude is not None and error_types_to_include is not None:
        msg = "Possible conflict in error types to be applied. Set at least one of: error_types_to_exclude or error_types_to_exclude to None."
        raise ValueError(msg)

    if error_types_to_exclude is None and error_types_to_include is not None:  # Include specified.
        if not all(issubclass(type(cls), ErrorType) for cls in error_types_to_include):  # Check input
            msg = "One of the elements of error_types_to_exclude is not a subclass of ErrorType."
            raise ValueError(msg)

        error_types_applied = error_types_to_include
    elif error_types_to_exclude is not None and error_types_to_include is None:  # Exclude specified.
        error_types_applied = [
            kept_error_type
            for kept_error_type in error_types_applied
            if not any(_are_same_class(kept_error_type, excluded_error_type) for excluded_error_type in error_types_to_exclude)
        ]
    # else: do nothing because the default behavior uses all error types

    if len(error_types_applied) == 0:
        msg = "The list of error types to be applied cannot have length 0. Use the default or resturcture your input."
        raise ValueError(msg)

    return {
        column: [valid_error_type for valid_error_type in error_types_applied if column in valid_error_type.get_valid_columns(data)] for column in data.columns
    }


def _build_column_mechanism_dictionary(
    data: pd.DataFrame,
    error_mechanisms_to_include: list[ErrorMechanism] | None = None,
    error_mechanisms_to_exclude: list[ErrorMechanism] | None = None,
    seed: int | None = None,
) -> dict[int | str, list[ErrorMechanism]]:
    """Builds a dictionary mapping from column names to the list of valid error mechanisms to apply to that column.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.
        error_mechanisms_to_include (list[ErrorMechanism] | None, optional): The error mechanisms (EAR, ECAR, ENAR) to include from the dictionary.
            Defaults to None.
        error_mechanisms_to_exclude (list[ErrorMechanism] | None, optional): The error mechanisms (EAR, ECAR, ENAR) to exclude from the dictionary.
            Defaults to None.
        seed (int | None, optional): Random seed. Defaults to None.

    Returns:
        dict[int | str, list[ErrorMechanism]]: A dictionary mapping from column names to the list of valid error mechanisms to apply to that column.
    """
    if error_mechanisms_to_exclude is not None and error_mechanisms_to_include is not None:  # Overspecified
        msg = "Possible conflict in error mechanisms to apply. Set at least on of: error_mechanisms_to_exclude or error_mechanisms_to_include to None."
        raise ValueError(msg)

    columns_mechanisms = {}

    if error_mechanisms_to_include is not None and error_mechanisms_to_exclude is None:  # Include specified
        if not all(issubclass(type(cls), ErrorMechanism) for cls in error_mechanisms_to_include):  # Check input
            msg = "One of the elements of error_mechanisms_to_include is not a subclass of ErrorMechanism."
            raise ValueError(msg)

        for column in data.columns:  # Simply check that the conditioning column is different from the current
            columns_mechanisms[column] = [
                kept_error_mechanism for kept_error_mechanism in error_mechanisms_to_include if kept_error_mechanism.condition_to_column != column
            ]

    if error_mechanisms_to_include is None:  # Exclude or none specified (overlapping cases)
        if error_mechanisms_to_exclude is not None and not all(issubclass(type(cls), ErrorMechanism) for cls in error_mechanisms_to_exclude):  # Check input
            msg = "One of the elements of error_mechanisms_to_exclude is not a subclass of ErrorMechanism."
            raise ValueError(msg)

        for column in data.columns:
            column_wise_error_mechs = [error_mechanism.ENAR(seed=seed), error_mechanism.ECAR(seed=seed)] + [
                error_mechanism.EAR(condition_to_column=other_column, seed=seed) for other_column in data.columns if other_column != column
            ]
            # Prune error mechanisms
            if error_mechanisms_to_exclude is not None:
                column_wise_error_mechs = [
                    kept_error_mechanism
                    for kept_error_mechanism in column_wise_error_mechs
                    if not any(
                        _are_same_error_mechanism(kept_error_mechanism, excluded_error_mechanism) for excluded_error_mechanism in error_mechanisms_to_exclude
                    )
                ]
            columns_mechanisms[column] = column_wise_error_mechs

    return columns_mechanisms


def _build_column_number_of_models_dictionary(
    data: pd.DataFrame, column_types: dict[int | str, list[ErrorType]], column_mechanisms: dict[int | str, list[ErrorMechanism]]
) -> dict[int | str, int]:
    """Builds a dictionary mapping from column names to the number of error models to apply to that column.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.
        column_types (dict[int | str, list[ErrorType]]): A dictionary mapping from column names to the list of valid error types to apply to that column.
        column_mechanisms (dict[int | str, list[ErrorMechanism]]): A dictionary mapping from column names to the list of valid error mechanisms to apply.

    Returns:
        dict[int | str, int]: A dictionary mapping from column names to the number of error models to apply to that column.
    """
    return {column: len(column_types[column]) * len(column_mechanisms[column]) for column in data.columns}


def _build_column_error_rate_dictionary(data: pd.DataFrame, error_rate: float, column_num_models: dict[int | str, int]) -> dict[int | str, float]:
    """Builds a dictionary mapping from column names to the error rate to apply to that column, based on the number of error models to be applied.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.
        error_rate (float): The maximum error rate to be introduced in the DataFrame.
        column_num_models (dict[int  |  str, int]): A dictionary mapping from column names to the number of error models to apply to that column.

    Returns:
        dict[int | str, float]: A dictionary mapping from column names to the error rate to apply to that column,
            based on the number of error models to be applied.
    """
    column_error_rates_dictionary = {}

    for column in data.columns:
        if column_num_models[column] > 0:  # Avoid zero division error
            column_error_rate = error_rate / column_num_models[column]
            n_rows = len(data[column])
            column_error_rates_dictionary[column] = column_error_rate

            if column_error_rate * n_rows < 1:  # This value is calculated and rounded to 0 in the sample function of the error mechanism subclasses "n_errors"
                msg = f"With an error rate for the column ({column}) of: {column_error_rate}, {column_num_models[column]} error models, and a length of: {n_rows}, 0 errors will be introduced. Try specifying error_types/ error_mechanisms to leave out"  # noqa: E501
                warnings.warn(msg, stacklevel=2)
        else:
            msg = f"The column {column} has no valid error models. 0 errors will be introduced to this column"
            warnings.warn(msg, stacklevel=2)

    return column_error_rates_dictionary


def create_errors(  # noqa: PLR0913
    data: pd.DataFrame,
    error_rate: float,
    error_types_to_include: list[ErrorType] | None = None,
    error_types_to_exclude: list[ErrorType] | None = None,
    error_mechanisms_to_include: list[ErrorMechanism] | None = None,
    error_mechanisms_to_exclude: list[ErrorMechanism] | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates errors in a given DataFrame, at a rate of *approximately* max_error_rate.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.
        error_rate (float): The maximum error rate to be introduced to each column in the DataFrame.
        error_types_to_include (list[ErrorType] | None, optional): A list of the error types to be included when building error models. Defaults to None.
        error_types_to_exclude (list[ErrorType] | None, optional): A list of the error types to be excluded when building error models. Defaults to None.
            When both error_types_to_include and error_types_to_exclude are none, the maximum number of default error types will be used.
            At least one must be None or an error will occur.
        error_mechanisms_to_include (list[ErrorMechanism] | None = None): A list of the error mechanisms to be included when building error models.
            Defaults to None.
        error_mechanisms_to_exclude (list[ErrorMechanism] | None = None): A list of the error mechanisms to be excluded when building error models.
            Defaults to None.
        seed (int | None, optional): Random seed. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - The first element is a copy of 'data' with errors.
            - The second element is the associated error mask.
    """
    # Input Checking
    check_error_rate(error_rate)
    check_data_emptiness(data)

    # Set Up Data
    data_copy = data.copy()
    error_mask = pd.DataFrame(data=False, index=data.index, columns=data.columns)


    # Build Dictionaries
    col_type = _build_column_type_dictionary(data=data, error_types_to_include=error_types_to_include, error_types_to_exclude=error_types_to_exclude, seed=seed)
    col_mechanisms = _build_column_mechanism_dictionary(
        data=data, error_mechanisms_to_include=error_mechanisms_to_include, error_mechanisms_to_exclude=error_mechanisms_to_exclude, seed=seed
    )
    col_num_models = _build_column_number_of_models_dictionary(data=data, column_types=col_type, column_mechanisms=col_mechanisms)
    col_error_rates = _build_column_error_rate_dictionary(data=data, error_rate=error_rate, column_num_models=col_num_models)


    # NOTE: Could be good to prune models from this MidLevelConfig --> somewhere upstream though so that the per-model-error-rate stays high
    # Build MidLevel Config
    config = MidLevelConfig(
        {
            column: [
                ErrorModel(error_mechanism=mech, error_type=etype, error_rate=col_error_rates[column])
                for mech in col_mechanisms[column]
                for etype in col_type[column]
            ]
            for column in data.columns
            if col_num_models[column] > 0
        }
    )

    # Create Errors & Return
    dirty_data, error_mask = mid_level.create_errors(data_copy, config)
    return dirty_data, error_mask
