import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sktime.transformations.series.outlier_detection import HampelFilter

from ...data_managers.namespaces import data_ns, files_ns

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
            return "Winter"
        elif x < datetime(y, 6, 22):
            return "Spring"
        elif x < datetime(y, 9, 23):
            return "Summer"
        else:
            return "Autumn"

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


class DayOffIndicatorCreator(BaseTransformer):
    """
    A transformer class for creating a day-off indicator
    column in a DataFrame with a datetime index.
    """

    DATE_COL = "DATE"

    @property
    def column(self) -> str:
        return "Is_Day_Off"

    def __init__(self, file: Optional[PATH] = None) -> None:
        """
        Initializes a new instance of the DayOffIndicatorCreator class.

        Parameters
        ----------
        file : pathlike, optional
            The path to the holiday file in CSV format.
            If not provided, the default holiday file path will be used.
        """

        if file is None:
            file = os.path.join(files_ns.DATA_FOLDER, files_ns.HOLIDAYS)

        file = Path(file)

        if not file.exists():
            raise ValueError(f"file {file} does not exist")
        self.file = file
        if file.suffix != ".csv":
            raise NotImplementedError("Holiday file must be in csv format")

    def _read_file(self) -> list:
        """
        Reads the holiday file and returns a list of holiday dates.
        Skips names of the holiday.

        Returns
        -------
        list
            A list of holiday dates.
        """
        file = pd.read_csv(
            self.file, parse_dates=[self.DATE_COL], usecols=[self.DATE_COL]
        )
        return file.squeeze().to_list()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding an 'Is_Day_Off' column,
        indicating whether each day is a holiday or a weekend day.

        Parameters
        ----------
        X : pd.DataFrame
            The input DataFrame with datetime index to be transformed.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame with the 'Is_Day_Off' column added.

        Raises
        ------
        ValueError
            If the DataFrame does not have a datetime index.
        """
        df = X.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame to transform should have datetime index")
        holidays = self._read_file()
        df[self.column] = list(
            map(
                lambda x: x.replace(hour=0) in holidays
                or x.day_name() in {"Sunday", "Suturday"},
                df.index,
            )
        )
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
        df[self.column] = list(map(int, out)) if self.return_bool else out
        return df


class FuelPricesProvider(BaseTransformer):
    def __init__(self, file: Optional[PATH] = None) -> None:
        """
        Initializes a new instance of the DayOffIndicatorCreator class.

        Parameters
        ----------
        file : pathlike, optional
            The path to the holiday file in CSV format.
            If not provided, the default holiday file path will be used.
        """

        if file is None:
            file = os.path.join(
                files_ns.DATA_FOLDER, files_ns.CURATED_FOLDER, files_ns.FUEL_PRICES
            )

        file = Path(file)

        if not file.exists():
            raise ValueError(f"file {file} does not exist")
        self.file = file
        if file.suffix != ".csv":
            raise NotImplementedError("Holiday file must be in csv format")

    @property
    def column(self) -> str:
        return "FUEL_PRICES"

    def _read_file(self) -> pd.Series:
        """
        Reads the holiday file and returns a list of holiday dates.
        Skips names of the holiday.

        Returns
        -------
        list
            A list of holiday dates.
        """
        data = pd.read_csv(
            self.file,
            parse_dates=[data_ns.TIME],
            index_col=data_ns.TIME,
        )
        return data.squeeze()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        new = self._read_file().rename(self.column)
        df = pd.merge(df, new, how="left", left_index=True, right_index=True)
        # first fill next day, then bfill in case some days from beginning
        # are missing (data starts from not midnight time)
        df[self.column] = df[self.column].ffill().fillna(method="bfill")
        return df
