import pytest
from pandas import DataFrame
from pandas.testing import assert_frame_equal

from categorical_encoder.base import HierachicalCategoricalEncoder
from categorical_encoder.smoothing import step_function


@pytest.fixture()
def simple_data() -> DataFrame:
    return DataFrame(
        {
            "column1": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "column2": ["0", "0", "1", "1", "0", "0", "1", "1"],
            "target": [0, 0, 1, 1, 2, 2, 3, 3],
        },
    )


def test_single_column_encoding(simple_data):
    # Test for column 1
    encoder = HierachicalCategoricalEncoder(
        columns=["column1"],
        smoothing_fn=step_function(min_samples=1),
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
    assert_frame_equal(expected, encoding)

    with_encoding = encoder.transform(simple_data)
    expected = DataFrame(
        {
            "column1": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "column2": ["0", "0", "1", "1", "0", "0", "1", "1"],
            "target": [0, 0, 1, 1, 2, 2, 3, 3],
            "__encoding__": [0.5, 0.5, 0.5, 0.5, 2.5, 2.5, 2.5, 2.5],
        },
    )
    assert_frame_equal(expected, with_encoding)

    # Test for column 2
    encoder = HierachicalCategoricalEncoder(
        columns=["column2"],
        smoothing_fn=step_function(min_samples=1),
        agg_fn="mean",
    )
    encoder.fit(simple_data, simple_data["target"])

    encoding = encoder.encoding
    expected = DataFrame(
        {
            "_l0_": ["None", "None"],
            "column2": ["0", "1"],
            "__encoding__": [1.0, 2.0],
        },
    )
    assert_frame_equal(expected, encoding)

    with_encoding = encoder.transform(simple_data)
    expected = DataFrame(
        {
            "column1": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "column2": ["0", "0", "1", "1", "0", "0", "1", "1"],
            "target": [0, 0, 1, 1, 2, 2, 3, 3],
            "__encoding__": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0],
        },
    )
    assert_frame_equal(expected, with_encoding)


def test_single_column_encoding_without_min_sample_size(simple_data):
    # Column 1
    encoder = HierachicalCategoricalEncoder(
        columns=["column1"],
        smoothing_fn=step_function(min_samples=100),
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
    assert_frame_equal(expected, encoding)

    with_encoding = encoder.transform(simple_data)
    expected = DataFrame(
        {
            "column1": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "column2": ["0", "0", "1", "1", "0", "0", "1", "1"],
            "target": [0, 0, 1, 1, 2, 2, 3, 3],
            "__encoding__": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        },
    )
    assert_frame_equal(expected, with_encoding)

    # Column 2
    encoder = HierachicalCategoricalEncoder(
        columns=["column2"],
        smoothing_fn=step_function(min_samples=100),
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
    assert_frame_equal(expected, encoding)

    with_encoding = encoder.transform(simple_data)
    expected = DataFrame(
        {
            "column1": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "column2": ["0", "0", "1", "1", "0", "0", "1", "1"],
            "target": [0, 0, 1, 1, 2, 2, 3, 3],
            "__encoding__": [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        },
    )
    assert_frame_equal(expected, with_encoding)


def test_multi_column_encoding(simple_data):
    # Columns 1 and 2
    encoder = HierachicalCategoricalEncoder(
        columns=["column1", "column2"],
        smoothing_fn=step_function(min_samples=1),
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

    with_encoding = encoder.transform(simple_data)
    expected = DataFrame(
        {
            "column1": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "column2": ["0", "0", "1", "1", "0", "0", "1", "1"],
            "target": [0, 0, 1, 1, 2, 2, 3, 3],
            "__encoding__": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        },
    )
    assert_frame_equal(expected, with_encoding)

    # Columns 2 and 1
    encoder = HierachicalCategoricalEncoder(
        columns=["column2", "column1"],
        smoothing_fn=step_function(min_samples=1),
        agg_fn="mean",
    )
    encoder.fit(simple_data, simple_data["target"])

    encoding = encoder.encoding
    expected = DataFrame(
        {
            "_l0_": ["None", "None", "None", "None"],
            "column2": ["0", "0", "1", "1"],
            "column1": ["0", "1", "0", "1"],
            "__encoding__": [0.0, 2.0, 1.0, 3.0],
        },
    )
    assert_frame_equal(expected, encoding)

    with_encoding = encoder.transform(simple_data)
    expected = DataFrame(
        {
            "column1": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "column2": ["0", "0", "1", "1", "0", "0", "1", "1"],
            "target": [0, 0, 1, 1, 2, 2, 3, 3],
            "__encoding__": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        },
    )
    assert_frame_equal(expected, with_encoding)


def test_multi_column_encoding_with_min_sample_size(simple_data):
    encoder = HierachicalCategoricalEncoder(
        columns=["column1", "column2"],
        smoothing_fn=step_function(min_samples=4),
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
    assert_frame_equal(expected, encoding)

    with_encoding = encoder.transform(simple_data)
    expected = DataFrame(
        {
            "column1": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "column2": ["0", "0", "1", "1", "0", "0", "1", "1"],
            "target": [0, 0, 1, 1, 2, 2, 3, 3],
            "__encoding__": [0.5, 0.5, 0.5, 0.5, 2.5, 2.5, 2.5, 2.5],
        },
    )
    assert_frame_equal(expected, with_encoding)

    # Inverse order
    encoder = HierachicalCategoricalEncoder(
        columns=["column2", "column1"],
        smoothing_fn=step_function(min_samples=4),
        agg_fn="mean",
    )
    encoder.fit(simple_data, simple_data["target"])

    encoding = encoder.encoding
    expected = DataFrame(
        {
            "_l0_": ["None", "None", "None", "None"],
            "column2": ["0", "0", "1", "1"],
            "column1": ["0", "1", "0", "1"],
            "__encoding__": [1.0, 1.0, 2.0, 2.0],
        },
    )
    assert_frame_equal(expected, encoding)

    with_encoding = encoder.transform(simple_data)
    expected = DataFrame(
        {
            "column1": ["0", "0", "0", "0", "1", "1", "1", "1"],
            "column2": ["0", "0", "1", "1", "0", "0", "1", "1"],
            "target": [0, 0, 1, 1, 2, 2, 3, 3],
            "__encoding__": [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0],
        },
    )
    assert_frame_equal(expected, with_encoding)


def test_multi_column_encode_missing(simple_data):
    encoder = HierachicalCategoricalEncoder(
        columns=["column1", "column2"],
        smoothing_fn=step_function(min_samples=1),
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
            "column1": ["0", "0", "1", "1", "1", "2"],
            "column2": ["0", "1", "0", "1", "2", "1"],
        },
    )
    expected = DataFrame(
        {
            "column1": ["0", "0", "1", "1", "1", "2"],
            "column2": ["0", "1", "0", "1", "2", "1"],
            "__encoding__": [0.0, 1.0, 2.0, 3.0, 2.5, 1.5],
        },
    )
    with_encoding = encoder.transform(test_data)
    assert_frame_equal(expected, with_encoding)
