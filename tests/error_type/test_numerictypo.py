import pandas as pd

from tab_err import error_mechanism, error_type
from tab_err.api.low_level import create_errors
from pandas.api.types import is_numeric_dtype


def test_numerictypo() -> None:
    """Test that NumericTypo replaces digits and commas with a random neighbouring character."""
    test_data = pd.DataFrame(
        {
            "A": [1, 2.0, 3, 45, 678, 90],
        }
    )
    modified_df, _ = create_errors(test_data, "A", 1, error_mechanism.ECAR(), error_type.NumericTypo())
    assert (test_data != modified_df).all(axis=None) # All values were perturbed
    assert is_numeric_dtype(modified_df["A"])