import pandas as pd

from tab_err import error_mechanism, error_type
from tab_err.api.low_level import create_errors


def test_sticky_key() -> None:
    """Test that StickyKey adds characters and produces seeded results."""
    test_data = pd.DataFrame({"A": ["type", "mongo", "text"]})
    modified_df, _ = create_errors(test_data, "A", 1, error_mechanism.ECAR(seed=1), error_type.StickyKey(seed=1))
    repeated_df, _ = create_errors(test_data, "A", 1, error_mechanism.ECAR(seed=1), error_type.StickyKey(seed=1))

    assert (modified_df["A"].str.len() > test_data["A"].str.len()).all()
    pd.testing.assert_frame_equal(modified_df, repeated_df)
