import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from categorical_encoder.base import HierachicalCategoricalEncoder


@pytest.fixture()
def simple_data() -> DataFrame:
    return DataFrame(
        {
            "column1": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "column2": ["0", "1", "0", "1", "0", "1", "0", "1"],
            "target": [0, 0, 1, 1, 2, 2, 3, 3],
        },
    )


def test_single_column_encoding(simple_data):
    # Test for column 1
    encoder = HierachicalCategoricalEncoder(
        columns=["column1"],
        min_samples=1,
        agg_fn="mean",
    )
    encoder.fit(simple_data, simple_data["target"])

    encoding = encoder.encoding
    expected = DataFrame(
        {
            "_l0_": ["None", "None"],
            "column1": ["0", "1"],
            "__encoding__": [0.5, 2.5],
        },
    )

    assert encoding is not None
    assert_frame_equal(expected, encoding)

    # Test for column 2
    encoder = HierachicalCategoricalEncoder(
        columns=["column2"],
        min_samples=1,
        agg_fn="mean",
    )
    encoder.fit(simple_data, simple_data["target"])

    encoding = encoder.encoding
    expected = DataFrame(
        {
            "_l0_": ["None", "None"],
            "column2": ["0", "1"],
            "__encoding__": [1.5, 1.5],
        },
    )
    assert encoding is not None
    assert_frame_equal(expected, encoding)


def test_single_column_encoding_without_min_sample_size(simple_data):
    encoder = HierachicalCategoricalEncoder(
        columns=["column1"],
        min_samples=100,
        agg_fn="mean",
    )
    encoder.fit(simple_data, simple_data["target"])

    encoding = encoder.encoding
    expected = DataFrame(
        {
            "_l0_": ["None", "None"],
            "column1": ["0", "1"],
            "__encoding__": [1.5, 1.5],
        },
    )

    assert encoding is not None
    assert_frame_equal(expected, encoding)


def test_multi_column_encoding(simple_data):
    encoder = HierachicalCategoricalEncoder(
        columns=["column1", "column2"],
        min_samples=1,
        agg_fn="mean",
    )
    encoder.fit(simple_data, simple_data["target"])

    encoding = encoder.encoding
    expected = DataFrame(
        {
            "_l0_": ["None", "None", "None", "None"],
            "column1": ["0", "0", "1", "1"],
            "column2": ["0", "1", "0", "1"],
            "__encoding__": [0.5, 0.5, 2.5, 2.5],
        },
    )

    assert encoding is not None
    assert_frame_equal(expected, encoding)
