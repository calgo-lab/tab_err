from __future__ import annotations

import warnings

import pandas as pd

from tab_err import ErrorMechanism, ErrorType, error_mechanism, error_type
from tab_err._error_model import ErrorModel
from tab_err.api import MidLevelConfig, mid_level


def build_column_type_dictionary(data: pd.DataFrame) -> dict[int | str, list[ErrorType]]:
    """Creates a dictionary mapping from column names to the list of valid error types to apply to that column.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.

    Returns:
        dict[int | str, list[ErrorModel]]: A dictionary mapping from column names to the list of valid error types to apply to that column.
    """
    # What default values should we pass into add delta, extraneous, replace, wrong unit?
    all_error_types =[
        error_type.AddDelta({"add_delta_value": 0.1}),  # Need default value
        error_type.CategorySwap(),
        error_type.Extraneous({"extraneous_value_template": ".{value}"}),  # Need default value
        error_type.Mojibake(),
        error_type.Outlier(),
        # error_type.Permutate(), - # Need default value
        error_type.Replace({"replace_what": "y", "replace_with": "z"}),  # Need default value  -- change default behavior in replace to be randomly sampling a character from the string to be affected.
        error_type.Typo(),
        error_type.WrongUnit({"wrong_unit_scaling": lambda x: 10.0*x}),  # Need default value
        error_type.MissingValue()  # Need to update the code, adding nans for numeric types when the missing value is None
    ]

    return {column:[valid_error_type for valid_error_type in all_error_types if column in valid_error_type.get_valid_columns(data)] for column in data.columns}

def build_column_mechanism_dictionary(data: pd.DataFrame) -> dict[int | str, list[ErrorMechanism]]:
    """Builds a dictionary mapping from column names to the list of valid error mechanisms to apply to that column.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.

    Returns:
        dict[int | str, list[ErrorMechanism]]: A dictionary mapping from column names to the list of valid error mechanisms to apply to that column.
    """
    columns_mechanisms = {}
    base_error_mechanisms = [error_mechanism.ENAR(), error_mechanism.ECAR()]

    for column in data.columns:
        columns_mechanisms[column] = base_error_mechanisms + [error_mechanism.EAR(condition_to_column=other_column)
                                                              for other_column in data.columns if other_column != column]

    return columns_mechanisms

def build_column_number_of_models_dictionary(
    data: pd.DataFrame,
    col_type: dict[int | str, list[ErrorType]],
    col_mech: dict[int | str, list[ErrorMechanism]]
    ) -> dict[int | str, int]:
    """Builds a dictionary mapping from column names to the number of error models to apply to that column.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.
        col_type (dict[int | str, list[ErrorType]]): A dictionary mapping from column names to the list of valid error types to apply to that column.
        col_mech (dict[int | str, list[ErrorMechanism]]): A dictionary mapping from column names to the list of valid error mechanisms to apply to that column.

    Returns:
        dict[int | str, int]: A dictionary mapping from column names to the number of error models to apply to that column.
    """
    return {column: len(col_type[column])*len(col_mech[column]) for column in data.columns}

def build_column_error_rate_dictionary(
    data: pd.DataFrame,
    max_error_rate: float,
    col_num_models: dict[int | str, int]
    ) -> dict[int | str, float]:
    """Builds a dictionary mapping from column names to the error rate to apply to that column, based on the number of error models to be applied.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.
        max_error_rate (float): The maximum error rate to be introduced in the DataFrame.
        col_num_models (dict[int  |  str, int]): A dictionary mapping from column names to the number of error models to apply to that column.

    Returns:
        dict[int | str, float]: A dictionary mapping from column names to the error rate to apply to that column,
            based on the number of error models to be applied.
    """
    column_error_rates_dictionary = {}

    for column in data.columns:
        column_error_rate = max_error_rate/col_num_models[column]
        n_rows = len(data[column])
        column_error_rates_dictionary[column] = column_error_rate

        if column_error_rate*n_rows < 1:  # This value is calculated and rounded to 0 in the sample function of the error mechanism subclasses "n_errors"
            msg = f"With an error rate for the column ({column}) of: {column_error_rate} and a length of: {n_rows}, 0 errors will be introduced."
            warnings.warn(msg, stacklevel=2)

    return column_error_rates_dictionary

def create_errors(data: pd.DataFrame, max_error_rate: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates errors in a given DataFrame, at a rate of *approximately* max_error_rate.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.
        max_error_rate (float): The maximum error rate to be introduced in the DataFrame.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - The first element is a copy of 'data' with errors.
            - The second element is the associated error mask.
    """
    # Input Checking
    if len(data) == 0:
        msg = "Cannot create errors in an empty DataFrame."
        raise ValueError(msg)

    if max_error_rate < 0.0 or max_error_rate > 1.0:
        msg = f"max_error_rate must be between 0 and 1, but was {max_error_rate}."
        raise ValueError(msg)

    # Set Up Data
    data_copy = data.copy()
    error_mask = pd.DataFrame(data=False, index=data.index, columns=data.columns)

    # Build Dictionaries
    col_type = build_column_type_dictionary(data)
    col_mechs = build_column_mechanism_dictionary(data)
    col_num_models = build_column_number_of_models_dictionary(data, col_type, col_mechs)
    col_error_rates = build_column_error_rate_dictionary(data, max_error_rate, col_num_models)

    print("Column-type dict: ", col_type)
    print("Column-mech dict: ", col_mechs)
    print("Column-num_models dict: ", col_num_models)
    print("Column-error_rates dict: ", col_error_rates)

    # NOTE: Could be good to prune models from this
    # Build MidLevel Config
    config = MidLevelConfig(
        {
            key:[
                ErrorModel(
                    error_mechanism=mech,
                    error_type=etype,
                    error_rate=col_error_rates[key]
                ) for mech in col_mechs[key] for etype in col_type[key]
            ]
            for key in data.columns
        }
    )

    # Create Errors & Return
    dirty_data, error_mask = mid_level.create_errors(data_copy, config)
    return dirty_data, error_mask
