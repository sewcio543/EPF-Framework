import logging
import random
from datetime import datetime
from typing import Optional, Union

import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import ExpandingWindowSplitter as _EWS
from sktime.forecasting.model_selection import SlidingWindowSplitter as _SWS
from sktime.forecasting.model_selection._split import BaseSplitter

DEFAULT_STEP_LENGTH = 24
DEFAULT_FRAC = 0.1
DEFAULT_SEED = 42


class ExpandingWindowSplitter(_EWS):
    def _split_windows(self, **kwargs):
        """Overriden spit_window method that logs modeling progress"""
        gen = super()._split_windows(**kwargs)
        for counter, (train, test) in enumerate(gen, start=1):
            mes = f"{counter} forecast -- last index: {test[-1]}"
            logging.info(mes)
            yield train, test


class SlidingWindowSplitter(_SWS):
    def _split_windows(self, **kwargs):
        """Overriden spit_window method that logs modeling progress"""
        gen = super()._split_windows(**kwargs)
        for counter, (train, test) in enumerate(gen, start=1):
            mes = f"{counter} forecast -- last index: {test[-1]}"
            logging.info(mes)
            yield train, test


class TestingSplitter(ExpandingWindowSplitter):
    """
    Splitter for generating expanding windows with a random subset for testing.
    Wrapper around ExpandingWindowSplitter that returns each window of splitter
    with given probability.

    Useful for testing more computationally demanding models
    that take more time to train, supported by TSBacktesting.

    Check out
    ---------
    app.modeling.backtesting TSBacktesting
    """

    def __init__(self, *args, frac: float = DEFAULT_FRAC, **kwargs) -> None:
        """Initializes TestingSplitter object

        Parameters
        ----------
        frac : float, optional
            fraction of windows to yield from entire validation set.
            1 being all windows (working as standard ExpandingWindowSplitter),
            0 not yielding any window for evaluation.

        Raises
        ------
        ValueError
            When frac is not between 0 and 1
        """
        super().__init__(*args, **kwargs)
        if not (0 <= frac <= 1):
            raise ValueError(f"frac must be between 0 and 1, got {frac}")
        self._frac = frac

    def _split_windows(self, **kwargs):
        """Overriden spit_window method that yield with frac probability"""
        # seed for reproducibility across different models (the same windows modelled)
        random.seed(DEFAULT_SEED)
        gen = self._split_windows_generic(expanding=True, **kwargs)
        counter = 1
        for train, test in gen:
            rand = random.random()
            if rand <= self._frac:
                mes = f"{counter} forecast -- last index: {test[-1]}"
                counter += 1
                logging.info(mes)
                yield train, test


def get_splitter(
    intial_window: int,
    step_length: int = DEFAULT_STEP_LENGTH,
    testing: bool = False,
    frac: float = DEFAULT_FRAC,
) -> BaseSplitter:
    """
    Get the appropriate splitter for generating expanding windows.

    Parameters
    ----------
    intial_window : int
        Length of splitter initial window.
        The length of training set of time series data.
    step_length : int, optional
        Length of the forecast step. Defaults to 24.
    testing : bool, optional
        Whether to use the TestingSplitter for testing. Defaults to False.
    frac : float, optional
        Fraction of expanding windows to use for testing, between 0 and 1.
        Defaults to 0.1.

    Returns
    -------
    BaseSplitter
        sktime Splitter object for generating expanding windows,
        compatible with sktime evaluate function and TSBacktesting.

    Check out
    ---------
    app.modeling.backtesting TSBacktesting
    """
    fh = ForecastingHorizon(list(range(1, step_length + 1)), is_relative=True)
    if testing:
        return TestingSplitter(
            fh=fh, initial_window=intial_window, frac=frac, step_length=step_length
        )
    return ExpandingWindowSplitter(
        fh=fh, initial_window=intial_window, step_length=step_length
    )


DATE = Union[str, datetime, None]


def split_series(
    y: pd.Series,
    train_start: DATE = None,
    train_end: DATE = None,
    test_len: Optional[int] = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Split a time series into train and test sets based on the specified parameters.

    Parameters
    ----------
    y : pd.Series
        The time series data to split.
    train_start : DATE, optional
        The start date or index from which to include data in the training set.
        Defaults to None.
    train_end : DATE, optional
        The end date or index until which to include data in the training set.
        Defaults to None.
    test_len : int, optional
        The length of the test set. If specified, the test set will include
        the first 'test_len' elements. Defaults to None.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        A tuple containing the train and test sets.

    Raises
    ------
    ValueError
        If both 'train_end' and 'test_len' are None, as at leas
        one of them must be specified to split the series.

    Notes
    -----
    - The function creates a copy of the input series 'y'
        to avoid modifying the original data.
    - The train set includes the data from 'train_start' (inclusive) until
        'train_end' (exclusive) or until the first 'test_len'
        elements if 'test_len' is specified.
    - The test set includes the remaining data
        from the original series after the train set.
    - If 'train_start' is specified, the function will filter the series
        to include data from 'train_start' onwards.
    - If 'train_end' is specified, the function will use it as the boundary
        for the train set. Otherwise, a boolean mask is created to mark the train set indices.
    - If 'test_len' is specified, the test set will be truncated
        to have at most 'test_len' elements.

    Examples
    --------
    >>> y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> train, test = split_series(y, train_start=3, test_len=3)
    >>> print(train)
    3    4
    4    5
    5    6
    dtype: int64
    >>> print(test)
    6    7
    7    8
    8    9
    dtype: int64
    """
    y = y.copy()

    if train_start is not None:
        y = y.loc[y.index >= train_start]

    if train_end is None and test_len is None:
        raise ValueError(
            "One of the parameters: train_end or test_len must be "
            "specified to split series"
        )
    if train_end is None:
        train_mask = pd.Series(False, index=y.index)
        train_mask[:test_len] = True
    else:
        train_mask = y.index < train_end

    train = y.loc[train_mask]
    test = y.loc[~train_mask]

    if test_len is not None:
        test = test.iloc[:test_len]

    return train, test
