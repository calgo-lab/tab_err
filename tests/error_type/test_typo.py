import pandas as pd

from tab_err import error_mechanism, error_type
from tab_err.api.low_level import create_errors


def test_typo() -> None:
    """Test that Typo replaces empty strings with a random character."""
    test_data = pd.DataFrame(
        {
            "A": ["", "Alice", "Bob", "Bob", "Clara", "David"],
        }
    )
    modified_df, _ = create_errors(test_data, "A", 1, error_mechanism.ECAR(), error_type.Typo())
    assert modified_df.iloc[0, 0] != ""
