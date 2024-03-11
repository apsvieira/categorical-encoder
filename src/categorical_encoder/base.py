"""Base estimator for categorical encoders."""

from typing import Union

from pandas import DataFrame, NamedAgg, Series, concat
from sklearn.base import BaseEstimator, TransformerMixin


class HierachicalCategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    An encoder that represents a hierarchy of categorical columns.

    This is useful for encoding hierarchical data, such as various levels of
    geographical information (eg Country, State, City, etc.)
    """

    def __init__(
        self,
        columns: Union[str, list[str]],
        min_samples: int,
        agg_fn: Union[str, NamedAgg],
        target_col: str = "__target__",
    ) -> None:
        """
        Initialize the encoder.

        Parameters
        ----------
        columns : Union[str, list[str]]
            The columns to encode. If a string is passed, it will be treated as a
            single column. If a list is passed, it will be treated as multiple
            columns to encode hierarchically.
        min_samples : int
            The minimum number of samples required to calculate the encoding.
            If the number of samples is less than this value, the encoding will
            be interpolated from the prior level.
        agg_fn : Union[str, NamedAgg]
            The aggregation function to use for encoding. This can be a string
            (eg "mean", "median", "sum", etc.) or a NamedAgg object.
        target_col : str
            The name of the target column.
            This is used to represent the target values internally.

        """
        if not isinstance(columns, (str, list)):
            msg = "columns must be a string or list of strings"
            raise TypeError(msg)

        if min_samples < 0:
            msg = "min_samples must be a positive integer"
            raise ValueError(msg)

        super().__init__()

        if isinstance(columns, str):
            columns = [columns]

        if not isinstance(agg_fn, NamedAgg):
            agg_fn = NamedAgg(column="__encoding__", aggfunc=agg_fn)

        self.columns = columns
        self.min_samples = min_samples
        self.agg_fn = agg_fn
        self._target_col = target_col

        self._levels = []

    def fit(
        self,
        X: DataFrame,
        y: Series,
    ) -> "HierachicalCategoricalEncoder":
        """Construct encoding values."""
        if X.shape[0] != y.shape[0]:
            msg = "X and y must have the same number of rows"
            raise ValueError(msg)

        data = X.copy()
        data[self._target_col] = y.copy()
        data["_l0_"] = "None"
        aggs = [
            NamedAgg("count", "count"),
            self.agg_fn,
        ]

        # Base case: the first level of encoding is just the target column aggregated
        level_0 = data.groupby("_l0_", as_index=False)[[self._target_col]].agg(aggs)
        level_0.columns = ["_l0_", "count", self.agg_fn.column]
        level_0 = level_0.drop(["count"], axis=1)
        prior = level_0
        levels = [level_0]

        # If there are multiple levels, we need to iterate through each
        # level using the level before it as an encoding prior.
        # We can do this by calculating the encoding for each level and
        # interpolating the prior encoding when necessary.
        columns = ["_l0_", *self.columns]
        for i, _ in enumerate(self.columns):
            encoding = data.groupby(columns[: i + 2], as_index=False)[
                self._target_col
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
            levels.append(merged)

        self._levels = levels

        return self

    @property
    def encoding(self) -> DataFrame:
        """Return the encoding."""
        return self._levels[-1]

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform the input data using the encoding."""
        if len(self._levels) == 0:
            msg = "fit must be called before transform"
            raise ValueError(msg)

        not_matched = X.copy()
        not_matched["_l0_"] = "None"

        # Perform partial merges to get the encoding.
        # Should use as much information as there is, and fill in with
        # priors whenever a value is new or has too few samples.
        # This loop is probably terrible for performance.
        # It's very straight forward, and I couldn't think of a way to
        # do this in a single merge or even using less indexing.
        matched_rows = []
        for encoding in self._levels[::-1]:
            if not_matched.shape[0] == 0:
                break

            merge_cols = encoding.columns.tolist()[:-1]
            matched = not_matched.merge(encoding, how="left", on=merge_cols)
            mask = matched[self.agg_fn.column].isna()
            matched_rows.append(matched.loc[~mask, :].reset_index(drop=True))
            not_matched = not_matched.loc[mask, :].reset_index(drop=True)

        data = concat(matched_rows, axis=0, ignore_index=True)
        data = data.drop(["_l0_"], axis=1)

        # Should add only the encoding column.
        if data.shape[1] != X.shape[1] + 1:
            msg = (
                f"transformed data has an incorrect number of columns ({data.shape[1]})"
            )
            raise ValueError(msg)

        return data


def _merge_levels(
    prior: DataFrame,
    current: DataFrame,
    on: list[str],
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

    return merged[[*on, current_level, agg_fn.column]]
