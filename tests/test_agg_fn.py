import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from categorical_encoder.base import HierachicalCategoricalEncoder
from categorical_encoder.smoothing import step_function


@pytest.fixture
def simple_data() -> DataFrame:
    return DataFrame(
        {
            "column1": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "column2": ["0", "0", "1", "1", "0", "0", "1", "1"],
            "target": [0.0, 0, 1, 1, 2, 2, 3, 3],
        },
    )


def test_agg_fn_sum(simple_data):
    encoder = HierachicalCategoricalEncoder(
        columns=["column1", "column2"],
        smoothing_fn=step_function(min_samples=1),
        agg_fn="sum",
    )
    encoder.fit(simple_data, simple_data["target"])

    encoding = encoder.encoding
    expected = DataFrame(
        {
            "_l0_": ["None", "None", "None", "None"],
            "column1": ["0", "0", "1", "1"],
            "column2": ["0", "1", "0", "1"],
            "__encoding__": [0.0, 2.0, 4.0, 6.0],
        },
    )
    assert_frame_equal(expected, encoding)

    test_data = DataFrame(
        {
            "column1": ["0", "0", "1", "1", "1", "2"],
            "column2": ["0", "1", "0", "1", "2", "1"],
        },
    )
    expected = DataFrame(
        {
            "column1": ["0", "0", "1", "1", "1", "2"],
            "column2": ["0", "1", "0", "1", "2", "1"],
            "__encoding__": [0.0, 2.0, 4.0, 6.0, 10.0, 12.0],
        },
    )
    with_encoding = encoder.transform(test_data)
    assert_frame_equal(expected, with_encoding)
