from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sktime.transformations.series.outlier_detection import HampelFilter


class BaseTransformer(ABC):
    def fit(self, X: pd.DataFrame, y):
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        ...


class TrendCreator(BaseTransformer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["TREND"] = np.arange(len(df))
        return df


class WeekendIndicatorCreator(BaseTransformer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["WEEKEND"] = list(map(lambda x: int(x.day_of_week in {5, 6}), df.index))
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
    def get_season(self, x: datetime) -> str:
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
        df = X.copy()
        season = np.array([self.get_season(x) for x in df.index])
        season_dummies = pd.get_dummies(season, drop_first=True)
        season_dummies = season_dummies.set_index(df.index)
        df = pd.concat((df, season_dummies), axis=1)
        return df


class DayOffIndicatorCreator(BaseTransformer):
    def __init__(self, file: str = "holidays.csv") -> None:
        self.file = file

    def _read_file(self) -> list:
        file = pd.read_csv(self.file, parse_dates=["Date"], usecols=["Date"])
        return file.squeeze().to_list()

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        holidays = self._read_file()
        df["Is_Day_Off"] = list(
            map(
                lambda x: x.replace(hour=0) in holidays
                or x.day_name() in {"Sunday", "Suturday"},
                df.index,
            )
        )
        return df


class LinearInterpolator(BaseTransformer):
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["VALUE"] = df["VALUE"].interpolate()
        return df


class OutlierFlagCreator(BaseTransformer):
    def __init__(
        self, window_length: Optional[int] = None, return_bool: bool = True
    ) -> None:
        self.window_length = window_length or (24 * 7)
        self.return_bool = return_bool

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        hf = HampelFilter(
            window_length=self.window_length, return_bool=self.return_bool
        )
        out = hf.fit_transform(df["VALUE"])
        if self.return_bool:
            df["OUTLIER"] = list(map(int, out))
        else:
            df["VALUE"] = out
        return df
