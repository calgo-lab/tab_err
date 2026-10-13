import json

import pandas as pd
import pytest

from tab_err import ErrorModel, ErrorType
from tab_err.api.mid_level import MidLevelConfig, create_errors
from tab_err.error_mechanism import EAR, ECAR
from tab_err.error_type import ErrorTypeConfig, Outlier, WrongUnit


@pytest.mark.parametrize("error_type_class", ErrorType.__subclasses__())
def test_error_type_dict_roundtrip(error_type_class: type[ErrorType]) -> None:
    """`ErrorType.from_dict(x.to_dict())` restores the class, the config and the seed."""
    original = error_type_class(ErrorTypeConfig(typo_error_period=7), seed=42)

    restored = ErrorType.from_dict(original.to_dict())

    assert type(restored) is error_type_class
    assert restored.config == original.config
    assert restored._seed == original._seed  # noqa: SLF001


def test_mid_level_config_dict_roundtrip(test_data: dict[str, pd.DataFrame]) -> None:
    """A MidLevelConfig that went through JSON produces exactly the same errors as the original."""
    data = test_data["data_100rows_3columns"]
    config = MidLevelConfig(
        {
            "A": [ErrorModel(EAR(condition_to_column="B", seed=42), Outlier(seed=42), 0.2)],
            "B": [ErrorModel(ECAR(seed=42), WrongUnit(seed=42), 0.2)],
        }
    )

    dirty, mask = create_errors(data, config)
    restored_dirty, restored_mask = create_errors(data, MidLevelConfig.from_dict(json.loads(json.dumps(config.to_dict()))))

    pd.testing.assert_frame_equal(dirty, restored_dirty)
    pd.testing.assert_frame_equal(mask, restored_mask)
