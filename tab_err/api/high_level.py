from __future__ import annotations

import ast
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from tab_err import ErrorMechanism, ErrorType, error_mechanism, error_type
from tab_err._error_model import ErrorModel
from tab_err.api import MidLevelConfig, mid_level


def random_subdivision(
    num_intervals: int,
    total_value: float,
    rng: random.Random | None = None
) -> list[float]:
    """Randomly partitions the interval [0, N] into num_breaks segments.

    Args:
        num_intervals (int): The number of segments to partition the interval into.
        total_value (float): The total rate to partition.
        rng (random.Random | None, optional): The random number generator to use. Defaults to None.

    Returns:
        list[float]: a list of num_intervals segment lengths that sum to total_value.
    """
    if rng is None:
        rng = random  # Default to Python's random module

    breakpoints = sorted(rng.betavariate(1.5, 1.0)*total_value for _ in range(num_intervals - 1))
    return [b - a for a, b in zip([0.0, *breakpoints], [*breakpoints, total_value])]  # Compute segment sizes & return


def load_yaml_config(file_path: str) -> dict[str | int, Any]:
    """Loads a yaml configuration file and returns a config dictionary."""
    yaml_path = Path(file_path)

    if not yaml_path.exists():
        msg = f"File {file_path} was not found."
        raise FileNotFoundError(msg)

    try:
        with yaml_path.open("r") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as exc:
        msg = f"Error loading the yaml file: {exc}"
        raise ValueError(msg) from exc


def error_rate_check(error_rate: float, column: str | int) -> None:
    """Ensures the error rates are within the valid range and formatted properly. Throws an error otherwise."""
    if not isinstance(error_rate, float):
        msg = f"Column {column} 'total-error-rate' must be a float."
        raise TypeError(msg)

    if error_rate > 1.0 or error_rate < 0.0:
        msg = f"Column {column} 'total-error-rate' must be between 0 and 1."
        raise ValueError(msg)

    if error_rate == 0.0:
        msg = "total_error_rate is 0.0, this could be due to total-error-rate not being set."
        warnings.warn(msg, category=UserWarning, stacklevel=2)


def num_error_models_check(num_models: int, column: str | int) -> None:
    """Ensures the number of error models are within the valid range and formatted properly. Throws an error otherwise."""
    if num_models < 0:
        msg = f"Column {column} 'num-error-models' must be a non-negative integer."
        raise ValueError(msg)

    if not isinstance(num_models, int):
        msg = f"Column {column} 'num-error-models' must be an integer."
        raise TypeError(msg)

    if num_models == 0:
        msg = "num_error_models is 0, this could be due to num-error-models not being set."
        warnings.warn(msg, category=UserWarning, stacklevel=2)


def parse_error_mechanisms(mech: str, conditions: dict[str, Any], column: str | int) -> ErrorMechanism:
    """Parse error mechanisms from one section of a yaml configuration file."""
    resulting_error_mechanism: ErrorMechanism
    if mech == "ECAR":
        resulting_error_mechanism = error_mechanism.ECAR()
    elif mech == "ENAR":
        resulting_error_mechanism = error_mechanism.ENAR()
    elif mech == "EAR":
        if conditions is None:
            msg = f"Column {column} 'EAR' error mechanism requires a 'conditioning-column' value."
            raise ValueError(msg)

        resulting_error_mechanism = error_mechanism.EAR(condition_to_column=conditions.get("conditioning-column"))
    else:
        msg = f"Column {column} 'error-mechanisms' must be one of ['ECAR', 'ENAR', 'EAR']."
        raise ValueError(msg)

    return resulting_error_mechanism

def parse_error_types(error_type_name: str, params: dict[str, Any] | None, column: str | int) -> ErrorType:  # noqa: C901, PLR0912
    """Parse error types from one section of a yaml configuration file."""
    resulting_error_type: ErrorType
    print(params)
    if error_type_name == "AddDelta":
        """if params is None or params.get("add_delta_value") is None:
            msg = f"Column {column} 'AddDelta' error type requires a parameter 'add_delta_value: Any'."
            msg += "  See error_types config for details."
            raise ValueError(msg)
        """

        resulting_error_type = error_type.AddDelta(params)
    elif error_type_name == "CategorySwap":

        resulting_error_type = error_type.CategorySwap()
    elif error_type_name == "MissingValue":
        """if params is not None and not isinstance(params.get("missing_value"), str):
            msg = f"'MissingValue' error type requires a str or None type parameter. It was {type(params.get('missing_value'))}."
            msg += "  See error_types config for details."
            raise ValueError(msg)
        """

        resulting_error_type = error_type.MissingValue(params)
    elif error_type_name == "Extraneous":
        """if (
            params is None
            or (template := params.get("extraneous_value_template")) is None
            or not isinstance(template, str)
            or "{value}" not in template
        ):
            msg = f"Column {column} 'Extraneous' error type requires a parameter 'extraneous_value_template: str'."
            msg += "  See error_types config for details."
            raise ValueError(msg)
        """

        resulting_error_type = error_type.Extraneous(params)
    elif error_type_name == "Mistype":

        resulting_error_type = error_type.Mistype(params)
    elif error_type_name == "Mojibake":
        """if (params is not None and not isinstance(params.get("encoding_sender"), str) and not isinstance(params.get("encoding_receiver"), str)):
            msg = f"Column {column} 'Mojibake' error type requires parameters 'encoding_sender: str' and 'encoding_receiver: str'."
            msg += "  See error_types config for details."
            raise ValueError(msg)
        """

        resulting_error_type = error_type.Mojibake(params)
    elif error_type_name == "Outlier":
        """if (params is not None and not isinstance(params.get("outlier_coefficient"), float) and not isinstance(params.get("outlier_noise_coeff"), float)):
            msg = f"Column {column} 'Outlier' error type requires parameters 'outlier_coefficient: float' and 'outlier_noise_coeff: float'."
            msg += "  See error_types config for details."
            raise ValueError(msg)
        """

        resulting_error_type = error_type.Outlier(params)
    elif error_type_name == "Permutate":

        resulting_error_type = error_type.Permutate(params)
    elif error_type_name == "Replace":

        resulting_error_type = error_type.Replace(params)
    elif error_type_name == "Typo":

        resulting_error_type = error_type.Typo(params)
    elif error_type_name == "WrongUnit":
        if "wrong_unit_scaling" in params:
            # Parse the string lambda into a function
            params["wrong_unit_scaling"] = eval(params["wrong_unit_scaling"])  # noqa: S307

        resulting_error_type = error_type.WrongUnit(params)
    else:
        msg = f"Column {column} 'error-types' must be one of those in the error_types directory."
        raise ValueError(msg)

    return resulting_error_type




def parse_yaml_config(config: dict[str, Any]) -> tuple[
    dict[str | int, list[ErrorMechanism]],
    dict[str | int, list[ErrorType]],
    dict[str | int, list[float]],
    dict[str | int, int],
    set[str | int]
]:
    """Parses a yaml configuration file and returns the error mechanisms and types.

    Returns:
        tuple[dict[str | int, list[ErrorMechanism]], dict[str | int, list[ErrorType]], dict[str | int, list[float]], dict[str | int, int]]:
            - The first element is a dictionary mapping column names to a list of ErrorMechanism objects.
            - The second element is a dictionary mapping column names to a list of ErrorType objects.
            - The third element is a dictionary mapping column names to a list of error rates.
            - The fourth element is a dictionary mapping column names to the number of error models to generate.
            - The fifth element is a set of columns named in the configuration file.
    """
    columns: set[str | int] = set()
    mechanisms_dict: dict[str | int, list[ErrorMechanism]] = {}
    types_dict: dict[str | int, list[ErrorType]] = {}
    num_models_dict: dict[str | int, int] = {}
    error_rates_dict: dict[str | int, list[float]] = {}

    for column, details in config.items():
        # Add column to columns
        if isinstance(column, str):
            columns.add(str(column))
        elif isinstance(column, int):
            columns.add(int(column))
        else:
            msg = f"Column must be an int or str, not {type(column)}"
            raise TypeError(msg)

        # Parse Error Mechanisms
        error_mechanisms_list = []
        for mech, conditions in details.get("error-mechanisms", {}).items():
            print("Mech: ", mech, "conditions: ", conditions)
            error_mech = parse_error_mechanisms(mech, conditions, column)
            error_mechanisms_list.append(error_mech)

        mechanisms_dict[column] = error_mechanisms_list

        # Parse Error Types
        error_types_list = []
        for error_type_name, params in details.get("error-types", {}).items():
            # Implement logic for parsing error types and ensuring the required parameters are present
            error_type_parsed = parse_error_types(error_type_name, params, column)
            error_types_list.append(error_type_parsed)

        types_dict[column] = error_types_list

        # Parse Number of Error Models
        num_models = details.get("num-error-models", 0)
        num_error_models_check(num_models, column)
        num_models_dict[column] = num_models

        # Parse Error Rates -- requires num_models to be set
        total_error_rate = details.get("total-error-rate", 0.0)
        error_rate_check(total_error_rate, column)
        error_rates_dict[column] = random_subdivision(num_models_dict[column], total_error_rate)

    print("Mechanisms: ", mechanisms_dict, "\nTypes: ", types_dict, "\nError Rates: ", error_rates_dict, "\nNum Models: ", num_models_dict, "\nColumns: ", columns)  # noqa: ERA001
    return mechanisms_dict, types_dict, error_rates_dict, num_models_dict, columns


def create_errors_from_config(
    data: pd.DataFrame,
    config_file: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates errors in a given DataFrame, following a user-defined configuration.

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.
        config_file (str): The path to the configuration file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - The first element is a copy of 'data' with errors.
            - The second element is the associated error mask.
    """
    # Setup
    config = load_yaml_config(config_file)
    data_copy = data.copy()
    error_mask = pd.DataFrame(data=False, index=data.index, columns=data.columns)

    # Parse yaml config
    error_mechanisms, error_types, error_rates, num_error_models, columns = parse_yaml_config(config)

    # Create Mid Level Config
    mid_level_config = MidLevelConfig(
        {
            key:[
                    ErrorModel(
                        random.choice(error_mechanisms[key]),
                        random.choice(error_types[key]),
                        error_rates[key].pop(random.randrange(len(error_rates[key])))
                    )
                    for _ in range(num_error_models[key])
                ]
            for key in columns if num_error_models[key] > 0
        }
    )
    print("Mid Level Config: ", mid_level_config)
    print(mid_level_config.columns)
    print([x for x in mid_level_config.columns][0])
    print(type([x for x in mid_level_config.columns][0]))

    # Apply errors
    dirty_data, error_mask = mid_level.create_errors(data_copy, mid_level_config)

    return dirty_data, error_mask


def create_errors(
    data: pd.DataFrame,
    overall_max_error: float,
    error_types: dict[str | int, list[ErrorType]] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates errors in a given DataFrame, following a user-defined configuration.

    Description:
        It creates a default set (i.e. could be given) of error mechanisms, error types, and the error rate.
        The number of error mechanisms to generate is randomly decided and distributed to each column with at least one error type associated.
        For each column with an associated error type, the overall maximum error rate is randomly split into 'number-of-allocated-error-models' pieces
        Then for each column with an associated error_type, the MidLevelConfig samples randomly with replacement from the error mechanisms and the error types,
        and samples the error rates without replacement.

        This yields a random error creation api requiring only the data, the maximum error rate per column,
            and specific error types if any columns not to be affected or affected with specific error types

        A few problems I see already:
            - We may want more choice over the error mechanisms to be chosen.
            - May want more choice over maximum error rate (per column for example)

        Idea:
            - I'm not sure if this is high-level enough but (seems like it would basically just generate random MidLevelConfigs and apply them),
            One could create a config yaml file with the following structure:

            column name-1:
                error-mechanisms:
                    possible-error-mechanism-1:
                        conditioning-column-1:
                    ...
                    possible-error-mechanism-n:
                        conditioning-column-n:
                error-types:
                    possible-error-type-1:
                        associated-params-1:
                    ...
                    possible-error-type-n:
                        associated-params-n:
                total-error-rate:
                num-error-models:
            ...
            column name-n:
                error-mechanisms:
                    possible-error-mechanism-1:
                        conditioning-column-1:
                    ...
                    possible-error-mechanism-n:
                        conditioning-column-n:
                error-types:
                    possible-error-type-1:
                        associated-params-1:
                    ...
                    possible-error-type-n:
                        associated-params-n:
                total-error-rate:
                num-error-models:

            - Then all that is needed in the create_errors function is the dataset and the path to this file or something
            - The only drawback is that this requires a lot of knowledge about and specification of the code base/ dataset
                - Solution -> A default config could be provided
            - Could be good because then, with these constraints, there are many different possible random, dirtied, datasets that could be generated
            - Is higher level than the mid-level api because there is no concrete specification of the error models to be applied

    Args:
        data (pd.DataFrame): The pandas DataFrame to create errors in.
        overall_max_error (float): The maximum error rate any column will have. (no more than this proportion of cells per column will be dirtied)
        error_types (dict[str | int, list[ErrorType]] | None, optional): A dictionary mapping column names of 'data' to a list of ErrorType objects to sample.
            If supplied None, the error_type: MissingValue will be applied. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - The first element is a copy of 'data' with errors.
            - The second element is the associated error mask.
    """
    if overall_max_error > 1 or overall_max_error < 0:  # There were out of bounds error rates
        msg = f"Error rates must be between 0 and 1. But were: {overall_max_error}"
        raise ValueError(msg)

    # Setup
    data_copy = data.copy()
    error_mask = pd.DataFrame(data=False, index=data.index, columns=data.columns)

    num_columns = len(data_copy.columns)
    num_error_models = random.randint(1, 3*num_columns)  # Choose a random number of errors between 1 and 3*number of columns
    column_keys = list(data_copy.columns.values)

    # This next line could use more work, maybe we want more than just one condition_to_column
    error_mechanisms = [error_mechanism.ECAR(), error_mechanism.ENAR(), error_mechanism.EAR(condition_to_column=random.randint(0, num_columns - 1))]

    if error_types is None:
        error_types = {
            key: [error_type.MissingValue()]
            for key in column_keys
        }

    # Randomly split the number of errors (num_errors) among the columns that have error types associated with them
    columns_with_error_types = [key for key in column_keys if len(error_types[key]) > 0]
    allocation = np.random.multinomial(num_error_models, [1/len(columns_with_error_types)] * len(columns_with_error_types))  # noqa: NPY002
    model_splits = dict(zip(columns_with_error_types, allocation))

    # Sample error rates per column with error types
    error_rates = {key: random_subdivision(model_splits[key], overall_max_error) for key in model_splits}

    # Debug statement
    # print("Error rate list: ", error_rates, "\nnum_error_models: ", num_error_models, "\nsplits: ", model_splits)  # noqa: ERA001

    # Create config - randomly sample from the error mechanisms, types, and rates
    config = MidLevelConfig(
        {
            key:[
                    ErrorModel(
                        random.choice(error_mechanisms),
                        random.choice(error_types[key]),
                        error_rates[key].pop(random.randrange(len(error_rates[key])))  # Pop the chosen error rate
                    )
                    for _ in range(model_splits[key])
                ]
            for key in columns_with_error_types
            if model_splits[key] > 0
        }
    )
    print(config)
    print(config.columns)

    # Apply errors
    dirty_data, error_mask = mid_level.create_errors(data_copy, config)  # Using mid_level ensures there are no collisions

    return dirty_data, error_mask


# Notes: Some things here need to change/ be better thought out
# - The error types need to be compatible with the data types of the columns they are applied to
# - The sampling of the error mechanisms and types should be more sophisticated
# - The error mechanisms need to be more configurable.
# I think it may require more discussion with the team to figure out how to do this.



