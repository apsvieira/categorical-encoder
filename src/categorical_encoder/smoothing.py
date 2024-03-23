"""Collection of default smoothing functions."""

from typing import Callable

from pandas import Series
from toolz.functoolz import curry

SmoothingFnType = Callable[[Series, Series, Series, int], Series]
"""Type signature for smoothing functions."""


@curry
def step_function(
    encoding: Series,
    num_samples: Series,
    prior: Series,
    min_samples: int,
) -> Series:
    """
    Step function for smoothing.

    Parameters
    ----------
    encoding : Series
        The current encoding.
    num_samples : Series
        The number of samples for each category.
    prior : Series
        The prior encoding.
    min_samples : int
        The minimum number of samples required to calculate the encoding.
        If the number of samples is less than this value, the prior encoding
        will be used instead.

    Returns
    -------
    Series
        The smoothed encoding.

    """
    return encoding.where(
        num_samples >= min_samples,
        prior,
    )
