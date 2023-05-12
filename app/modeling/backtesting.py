from typing import Iterable

import pandas as pd
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection._split import BaseSplitter
from sktime.forecasting.naive import NaiveForecaster
from sktime.performance_metrics.forecasting import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)
from sktime.performance_metrics.forecasting._classes import BaseForecastingErrorMetric

DEFAULT_MODELS = {
    "SEASONAL_NAIVE_MEAN": NaiveForecaster(strategy="mean", sp=24),
    "SEASONAL_NAIVE_MEAN_3_DAYS": NaiveForecaster(
        strategy="mean", sp=24, window_length=72
    ),
}

DEFAULT_METRICS = {
    "MAPE": MeanAbsolutePercentageError(),
    "MAE": MeanAbsoluteError(),
    "RMSE": MeanSquaredError(square_root=True),
}


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
        models: dict = DEFAULT_MODELS,
        metrics: dict[str, BaseForecastingErrorMetric] = DEFAULT_METRICS,
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
            Keys are model names and values are sktime models. By deafult, simple
            naive models for testing purposes.
        metrics : dict, optional
            A dictionary of sktime forecasting error metrics to use for evaluation.
            Keys are metric names and values are sktime metrics. By dafaut, basic
            sktime metrics (MAPE, MAE, RMSE). Sktime metics can be substituted
            with any callable that takes two iterables as input and returns single
            number with error as output.
        """
        self._models = models
        self._splitter = splitter
        self._metrics = metrics
        self._errors = pd.DataFrame()

    @property
    def errors_(self) -> pd.DataFrame:
        """
        Return the evaluation metrics for each model.

        Returns
        -------
        pandas.DataFrame
            A dataframe of evaluation metrics for each model.

        """
        return self._errors

    def evaluate(self, y: pd.Series) -> pd.DataFrame:
        """
        Evaluate each model using time series backtesting and return the forecasts.

        Parameters
        ----------
        y : pandas.Series
            The target time series to forecast.

        Returns
        -------
        pandas.DataFrame
            A dataframe of forecasts for each model.
        """
        res = {}
        for name, model in self._models.items():
            cv_res = evaluate(
                model, cv=self._splitter, y=y, return_data=True, strategy="update"
            )
            preds = pd.concat(iter(cv_res.y_pred)).rename(name)
            res[name] = preds
        results = pd.DataFrame(res)
        self._errors = self._calculate_errors(results, y)
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
        actual_mask = results.index.intersection(y.index)
        actuals = y.loc[actual_mask]
        metrics = results.apply(lambda x: self._get_metrics(x, actuals)).to_dict()
        return pd.DataFrame(metrics).T

    def _get_metrics(
        self,
        forecast: Iterable,
        actuals: Iterable,
    ) -> dict:
        """
        Calculate the specified performance metrics for a single evaluated model
        using the actual values and predicted values.
        This method returns a dictionary object containing
        the calculated error metrics for the evaluated model.

        Parameters:
        -----------
        forecast: Iterable
            A iterable object containing the predicted values
            for a single evaluated model.
        actuals: Iterable
            A iterable object containing the actual values to be used
            for error metric calculation.

        Returns:
        --------
        metrics: dict
            A dictionary object containing the calculated error
            metrics for the evaluated model.
        """
        return {
            name: metric(actuals, forecast) for name, metric in self._metrics.items()
        }
