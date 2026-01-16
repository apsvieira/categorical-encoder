import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from categorical_encoder.base import HierachicalCategoricalEncoder
from categorical_encoder.smoothing import convex_combination


@pytest.fixture
def simple_data() -> DataFrame:
    return DataFrame(
        {
            "column1": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "column2": ["0", "0", "1", "1", "0", "0", "1", "1"],
            "target": [0.0, 0, 1, 1, 2, 2, 3, 3],
        },
    )


def test_convex_combination(simple_data):
    encoder = HierachicalCategoricalEncoder(
        columns=["column1", "column2"],
        smoothing_fn=convex_combination(x_min=1, x_max=2),
        agg_fn="mean",
    )
    encoder.fit(simple_data, simple_data["target"])

    encoding = encoder.encoding
    expected = DataFrame(
        {
            "_l0_": ["None", "None", "None", "None"],
            "column1": ["0", "0", "1", "1"],
            "column2": ["0", "1", "0", "1"],
            "__encoding__": [0.0, 1.0, 2.0, 3.0],
        },
    )
    assert_frame_equal(expected, encoding)

    test_data = DataFrame(
        {
            "column1": ["0", "0", "1", "1", "0", "1", "2"],
            "column2": ["0", "1", "0", "1", "2", "2", "1"],
        },
    )
    expected = DataFrame(
        {
            "column1": ["0", "0", "1", "1", "0", "1", "2"],
            "column2": ["0", "1", "0", "1", "2", "2", "1"],
            "__encoding__": [0.0, 1.0, 2.0, 3.0, 0.5, 2.5, 1.5],
        },
    )
    with_encoding = encoder.transform(test_data)
    assert_frame_equal(expected, with_encoding)
