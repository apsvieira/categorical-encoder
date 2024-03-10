"""Base estimator for categorical encoders."""

from typing import List, Union

import pandas as pd
from pandas import DataFrame, NamedAgg, Series
from sklearn.base import BaseEstimator, TransformerMixin


class HierachicalCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    A HierachicalCategoricalEncoder is a categorical encoder that can handle
    multiple columns at once to produce a single encoding value.
    This is useful for encoding hierarchical data, such as various levels of
    geographical information (eg Country, State, City, etc.)
    """

    def __init__(
        self,
        columns: Union[str, List[str]],
        min_samples: int,
        agg_fn: Union[str, NamedAgg],
        target_column: str = "__target__",
    ) -> None:
        if not isinstance(columns, (str, list)):
            raise TypeError("columns must be a string or list of strings")

        if min_samples < 0:
            raise ValueError("min_samples must be a positive integer")

        super().__init__()

        if isinstance(columns, str):
            columns = [columns]

        if not isinstance(agg_fn, NamedAgg):
            agg_fn = NamedAgg(column="__encoding__", aggfunc=agg_fn)

        self.columns = columns
        self.min_samples = min_samples
        self.agg_fn = agg_fn
        self.target_column = target_column

        self.encoding: DataFrame = None

    def fit(self, X: DataFrame, y: Series) -> "HierachicalCategoricalEncoder":
        """Construct encoding values."""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        data = X.copy()
        data[self.target_column] = y.copy()
        data["_l0_"] = "None"
        aggs = [
            NamedAgg("count", "count"),
            self.agg_fn,
        ]

        # Base case: the first level of encoding is just the target column aggregated
        level_0 = data.groupby("_l0_", as_index=False)[[self.target_column]].agg(aggs)
        level_0.columns = ["_l0_", "count", self.agg_fn.column]
        prior = level_0

        columns = ["_l0_"] + self.columns
        for i, _ in enumerate(self.columns):
            encoding = data.groupby(columns[: i + 2], as_index=False)[
                self.target_column
            ].agg(aggs)
            merged = _merge_levels(
                prior,
                encoding,
                columns[: i + 1],
                columns[i + 1],
                self.agg_fn,
                self.min_samples,
            )
            prior = merged
            self.encoding = merged
            print("encoding", self.encoding)

        # If there are multiple levels, we need to iterate through each
        # level using the level before it as an encoding prior.
        # We can do this by calculating the encoding for each level and
        # interpolating the prior encoding when necessary.

        return self


def _merge_levels(
    prior: DataFrame,
    current: DataFrame,
    on: List[str],
    current_level: str,
    agg_fn: NamedAgg,
    min_samples: int,
) -> DataFrame:
    """Merge two levels of encoding."""
    merged = prior.merge(
        current,
        how="left",
        on=on,
        suffixes=("_prior", ""),
    )

    merged[agg_fn.column] = merged[agg_fn.column].where(
        merged["count"] >= min_samples,
        merged[agg_fn.column + "_prior"],
    )
    print("merged", merged)

    return merged[on + [current_level, agg_fn.column]]
