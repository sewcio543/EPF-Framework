import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sktime.transformations.series.outlier_detection import HampelFilter

from ...data_managers.namespaces import data_ns, files_ns
from ...data_managers.readers.weather_reader import TEMPERATURE, WIND_SPEED

PATH = Union[Path, str]


class BaseTransformer(ABC):
    def fit(self, X: pd.DataFrame, y):
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...


class TrendCreator(BaseTransformer):
    @property
    def column(self) -> str:
        return "TREND"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df[self.column] = np.arange(len(df))
        return df


class WeekendIndicatorCreator(BaseTransformer):
    @property
    def column(self) -> str:
        return "WEEKEND"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df[self.column] = list(map(lambda x: int(x.day_of_week in {5, 6}), df.index))
        return df


class DayOfWeekIndicatorCreator(BaseTransformer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        day = np.array([date.day_name() for date in df.index])
        day_dummies = pd.get_dummies(day, drop_first=True)
        day_dummies = day_dummies.set_index(df.index)
        df = pd.concat((df, day_dummies), axis=1)
        return df


class SeasonIndicatorCreator(BaseTransformer):
    """
    A transformer class for creating season indicators
    in a DataFrame based on the date index. Implements one-hot-encoding
    for creating three season boolean columns (one is dropped).
    """

    def _get_season(self, x: datetime) -> str:
        """
        Returns the season based on the given datetime.

        Parameters
        ----------
        x : datetime
            The datetime for which the season needs to be determined.

        Returns
        -------
        str
            The season corresponding to the given datetime.
        """
        # random year
        y = 2000
        x = x.replace(year=y)

        if x < datetime(y, 3, 21) or x > datetime(y, 12, 22):
            return "WINTER"
        elif x < datetime(y, 6, 22):
            return "SPRING"
        elif x < datetime(y, 9, 23):
            return "SUMMER"
        else:
            return "AUTUMN"

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding season indicators.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with season indicators.
        """
        df = X.copy()
        season = np.array([self._get_season(x) for x in df.index])
        season_dummies = pd.get_dummies(season, drop_first=True)
        season_dummies = season_dummies.set_index(df.index)
        df = pd.concat((df, season_dummies), axis=1)
        return df


class LinearInterpolator(BaseTransformer):
    @property
    def column(self) -> str:
        return data_ns.VALUE

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df[data_ns.VALUE] = df[data_ns.VALUE].interpolate()
        return df


class OutlierFlagCreator(BaseTransformer):
    """
    A transformer class for creating outlier flags
    in a DataFrame using the Hampel filter from sktime.
    """

    @property
    def column(self) -> str:
        return "OUTLIER" if self.return_bool else data_ns.VALUE

    def __init__(
        self, window_length: Optional[int] = None, return_bool: bool = True
    ) -> None:
        """
        Initializes a new instance of the OutlierFlagCreator class.

        Parameters
        ----------
        window_length : Optional[int], optional
            The length of the window used by the Hampel filter. If not provided,
            a default window length of 24 * 7 (one week) will be used.
        return_bool : bool, optional
            Indicates whether to return boolean outlier flags (True) or replace
            the original values with outlier flags (False). Default is True.
        """
        self.window_length = window_length or (24 * 7)
        self.return_bool = return_bool

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding outlier flags.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with outlier flags as new column
            if return_bool is True, else smoothed values
            with outliers removed in VALUE column.
        """
        df = X.copy()
        hf = HampelFilter(
            window_length=self.window_length, return_bool=self.return_bool
        )
        out = hf.fit_transform(df[data_ns.VALUE])
        df[self.column] = list(map(int, out)) if self.return_bool else out  # type: ignore
        return df


class BaseFileProvider(BaseTransformer):
    def __init__(self, file: Optional[PATH] = None) -> None:
        """
        Parameters
        ----------
        file : pathlike, optional
            The path to data source file in CSV format.
            If not provided, the default file path for source will be used.
        """

        if file is None:
            file = self._default_path

        file = Path(file)

        if not file.exists():
            raise ValueError(f"file {file} does not exist")
        if file.suffix != ".csv":
            raise NotImplementedError("File must be in csv format")
        self.file = file

    @property
    @abstractmethod
    def _default_path(self) -> str:
        ...

    @property
    @abstractmethod
    def column(self) -> str:
        ...

    def _read_file(self) -> pd.Series:
        """
        Reads data source file and returns a series of numeric values.

        Returns
        -------
        pd.Series
            A series representing new feature for modeling
        """
        data = pd.read_csv(
            self.file,
            parse_dates=[data_ns.TIME],
            index_col=data_ns.TIME,
            usecols=[0, 1],
        )
        data = data.squeeze()
        data.name = self.column
        return data

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding an new column,
        based on marge on index of exo source file.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame with datetime index to be transformed
            by appending new column with exo feature.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with new column added

        Raises
        ------
        ValueError
            If the DataFrame does not have a datetime index.
        """
        df = X.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame to transform should have datetime index")
        exo = self._read_file()
        df = self._transform(df, exo=exo)
        return df

    def _transform(self, X: pd.DataFrame, exo) -> pd.DataFrame:
        """
        Source specific transforming operation on input dataframe
        with use of exo feature.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame with input data to be transformed
        exo : Any
            Data representing exogenious variable to be added to
            input data or used for transformations.

        Returns
        -------
        pd.DataFrame
            Transformed input DataFrame.
        """
        df = pd.merge(X, exo, how="left", left_index=True, right_index=True)
        # in case of less granular frequency or missing values, data is filled.
        # first fill next day, then bfill in case some days from beginning
        # are missing (data starts from not midnight time)
        df[self.column] = df[self.column].ffill().fillna(method="bfill")
        return df


class FuelPricesProvider(BaseFileProvider):
    @property
    def _default_path(self) -> str:
        return os.path.join(
            files_ns.DATA_FOLDER, files_ns.CURATED_FOLDER, files_ns.FUEL_PRICES
        )

    @property
    def column(self) -> str:
        return "FUEL_PRICE"


class CO2PricesProvider(BaseFileProvider):
    @property
    def _default_path(self) -> str:
        return os.path.join(
            files_ns.DATA_FOLDER, files_ns.CURATED_FOLDER, files_ns.CO2_PRICES
        )

    @property
    def column(self) -> str:
        return "CO2_PRICE"


class EnergyDemandProvider(BaseFileProvider):
    @property
    def _default_path(self) -> str:
        return os.path.join(
            files_ns.DATA_FOLDER, files_ns.CURATED_FOLDER, files_ns.ENERGY_DEMAND
        )

    @property
    def column(self) -> str:
        return "DEMAND"


class WeatherProvider(BaseFileProvider):
    @property
    def _default_path(self) -> str:
        return os.path.join(
            files_ns.DATA_FOLDER, files_ns.CURATED_FOLDER, files_ns.WEATHER
        )

    def _read_file(self) -> pd.Series:
        data = pd.read_csv(
            self.file,
            parse_dates=[data_ns.TIME],
            index_col=data_ns.TIME,
            usecols=[data_ns.TIME, self.column],
        )
        data = data.squeeze()
        data.name = self.column
        return data


class TemperatureProvider(WeatherProvider):
    @property
    def column(self) -> str:
        return TEMPERATURE


class WindSpeedProvider(WeatherProvider):
    @property
    def column(self) -> str:
        return WIND_SPEED


class DayOffIndicatorCreator(BaseFileProvider):
    """
    A transformer class for creating a day-off indicator
    column in a DataFrame with a datetime index.
    """

    @property
    def column(self) -> str:
        return "IS_DAY_OFF"

    @property
    def _default_path(self) -> str:
        return os.path.join(files_ns.DATA_FOLDER, files_ns.HOLIDAYS)

    def _read_file(self) -> list:
        """
        Reads the holiday file and returns a list of holiday dates.
        Skips names of the holiday.

        Returns
        -------
        list
            A list of holiday dates.
        """
        data = super()._read_file()
        return data.index.to_list()

    def _transform(self, X: pd.DataFrame, exo) -> pd.DataFrame:
        """
        Adds boolean columns indicating whether date was a day off or not.
        Includes all holidays and weekends. By deafult includes all polish
        holidays from default holiday file.
        """
        df = X.copy()
        ind = map(
            lambda x: x.replace(hour=0) in exo
            or x.day_name() in {"Sunday", "Suturday"},
            df.index,
        )
        df[self.column] = list(map(int, ind))
        return df
