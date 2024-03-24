"""Collection of default smoothing functions."""

from typing import Callable

from pandas import Series
from toolz.functoolz import curry

SmoothingFnType = Callable[[Series, Series, Series], Series]
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


@curry
def convex_combination(
    encoding: Series,
    num_samples: Series,
    prior: Series,
    x_min: int,
    x_max: int,
) -> Series:
    """
    Convex combination for smoothing.

    Takes two sample sizes x_min and x_max and returns a convex combination
    of the prior and current encoding according to the number of samples.
    If the number of samples is less than x_min, the prior encoding will be used.
    If the number of samples is greater than x_max, the current encoding will be used.

    Parameters
    ----------
    encoding : Series
        The current encoding.
    num_samples : Series
        The number of samples for each category.
    prior : Series
        The prior encoding.
    x_min : int
        The minimum number of samples required to calculate the encoding.
    x_max : int
        The number of samples at which the encoding will be equal to the current value.

    Returns
    -------
    Series
        The smoothed encoding.

    """
    if x_min > x_max:
        msg = f"{x_min=} must be less than or equal to {x_max=}"
        raise ValueError(msg)

    capped_x = num_samples.clip(x_min, x_max)
    return prior + (encoding - prior) * (capped_x - x_min) / (x_max - x_min)
