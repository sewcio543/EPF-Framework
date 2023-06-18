import logging
import warnings
from typing import Callable, Optional

import pandas as pd
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.model_selection._split import BaseSplitter
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

from ..data_managers.namespaces import data_ns

DEFAULT_METRICS = {
    "MAPE": MeanAbsolutePercentageError(),
    "MAE": MeanAbsoluteError(),
    "RMSE": MeanSquaredError(square_root=True),
}

DECORATORS = Optional[list[Callable[[Callable], pd.DataFrame]]]


# TODO: implement asynch run
def _evaluate(
    y: pd.Series,
    model: BaseForecaster,
    splitter: BaseSplitter,
    X: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Evaluate a time series forecasting model using a given splitter.

    Parameters
    ----------
    y: pd.Series
        The target time series data for backtesting.
    model: BaseForecaster
        The time series forecasting model to evaluate.
    splitter: BaseSplitter
        The splitter object used for train-test splitting.
    X: pd.DataFrame, optional
        The exogenous variables (if any). Defaults to None.

    Returns
    -------
    pd.Series
        The concatenated predictions made by the model with datetime index.
    """
    results = pd.Series()

    try:
        for train, test in splitter.split(y):
            # fitting
            exo = X.iloc[train] if isinstance(X, pd.DataFrame) else None
            model_trained = model.fit(y.iloc[train], X=exo, fh=splitter.fh)
            # predicting
            exo = X.iloc[test] if isinstance(X, pd.DataFrame) else None
            pred = model_trained.predict(fh=splitter.fh, X=exo)
            results = pd.concat((results, pred))  # type: ignore
    # function can be interrupted at any time, Series with successfully predicted
    # sets are returned, exception is raised to be handled in TSBacktesting component
    except KeyboardInterrupt:
        raise
    finally:
        return results


class TSBacktesting:
    """
    A time series backtesting component for evaluating forecasting models.

    Backtesting is a process of evaluating the performance of a predictive model
    on historical data. It involves simulating the model on past data to check
    how well it would have predicted the actual outcomes.

    In the context of time series forecasting, backtesting involves testing different
    forecasting models on historical time series data and evaluating their performance.
    TSBacktesting allows users to evaluate different forecasting models using
    various performance metrics and compare their results.

    Component also provides a way to calculate the errors of each model
    with specific metrics, which can be useful for comparing their performance.
    """

    def __init__(
        self,
        splitter: BaseSplitter,
        models: dict[str, BaseForecaster],
        metrics: dict[str, BaseForecastingErrorMetric] = DEFAULT_METRICS,
        decorators: DECORATORS = None,
    ) -> None:
        """
        Initialize the time series backtesting component.

        Parameters
        ----------
        splitter : sktime BaseSplitter
            A splitter object for generating train/test splits of the data.
            Sktime-interface splitter that implements split method that returns
            generator for train and test sets.
        models : dict, optional
            A dictionary of sktime forecasting models to evaluate.
            Keys are model names and values are sktime models.
        metrics : dict, optional
            A dictionary of sktime forecasting error metrics to use for evaluation.
            Keys are metric names and values are sktime metrics. By dafaut, basic
            sktime metrics (MAPE, MAE, RMSE). Sktime metics can be substituted
            with any callable that takes two iterables as input and returns single
            number with error as output.
        decorators: list[callable], optional
            List of decorator function that would enhance evaluate method by adding
            extra functionality, they should take callable as an input, and return
            the DataFrame with modeling results. By default, no decorator is applied.
        """
        self._models = models
        self._splitter = splitter
        self._metrics = metrics
        self._errors = pd.DataFrame()
        if decorators is not None:
            for func in decorators:
                self.evaluate = func(self.evaluate)  # type: ignore

    def _silence_loggers(self) -> None:
        """Turns off cmdstanpy logger used by sktime"""
        import logging

        # logger used by sktime
        logger = logging.getLogger("cmdstanpy")
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)

        # silence models warnings like ex. ConverganceError
        warnings.filterwarnings("ignore")

    @property
    def errors_(self) -> pd.DataFrame:
        """Return the evaluation metrics for each model"""
        return self._errors.sort_values(
            by=self._errors.columns.to_list(), ascending=True
        )

    def evaluate(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Evaluate each model using time series backtesting and return the forecasts.

        Parameters
        ----------
        y : pd.Series
            The target time series to forecast.
        X: pd.DataFrame, optional
            DataFrame with exogenious variables as features passed
            to sktime evaluate function. It must have the same index as y.

        Returns
        -------
        pandas.DataFrame
            A dataframe of forecasts for each model.
        """
        self._silence_loggers()
        res = {}
        for i, (name, model) in enumerate(self._models.items(), start=1):
            logging.info(f"Model {i} -- {name}")
            try:
                res[name] = _evaluate(y, model=model, splitter=self._splitter, X=X)
            except KeyboardInterrupt:
                break

        results = pd.DataFrame(res)
        self._errors = self._calculate_errors(results, y)
        results[data_ns.ACTUAL] = y.loc[results.index]
        return results

    def _calculate_errors(self, results: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Calculate the evaluation metrics for each model.

        Parameters
        ----------
        results : pandas.DataFrame
            A dataframe of forecasts for each model.
        y : pandas.Series
            The target time series to forecast.

        Returns
        -------
        pandas.DataFrame
            A dataframe of evaluation metrics for each model.
        """
        metrics = results.apply(lambda x: self._get_metrics(x, y)).to_dict()
        return pd.DataFrame(metrics).T

    def _get_metrics(
        self,
        forecast: pd.Series,
        actuals: pd.Series,
    ) -> dict:
        """
        Calculate the specified performance metrics for a single evaluated model
        using the actual values and predicted values.
        This method returns a dictionary object containing
        the calculated error metrics for the evaluated model.

        Parameters:
        -----------
        forecast: pd.Series
            A Series object containing the predicted values
            with datetime index for a single evaluated model.
        actuals: pd.Series
            A Series object containing the actual values to be used
            with datetime index for error metric calculation.

        Returns:
        --------
        metrics: dict
            A dictionary object containing the calculated error
            metrics for the evaluated model.
        """
        actuals = actuals.dropna()
        forecast = forecast.dropna()
        # get only the intersection of index for calculating metrics
        mask = forecast.index.intersection(actuals.index)
        return {
            name: metric(actuals.loc[mask], forecast.loc[mask])
            for name, metric in self._metrics.items()
        }
