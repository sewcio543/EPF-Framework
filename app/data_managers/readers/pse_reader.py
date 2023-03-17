from datetime import timedelta

import pandas as pd

from ..namespaces import data_ns
from .base_reader import CSVReader

TIMEZONE_MARK = "A"


class PSEReader(CSVReader):
    HOUR_COLUMN = data_ns.HOUR
    DATE_COLUMN = data_ns.DATE
    _SEP = ";"

    def _format(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = self._add_time_column(data=df)
        df = self._format_numeric_columns(data=df)
        return df

    def _add_time_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """Adds new datetime column with entry time to DataFrame"""
        df = data.copy()
        # removes duplicated entries
        df = self._remove_timezone_shifts(data=df)
        # replaces 24 hour with 0 hour
        df = self._replace_midnight_entries(data=df)
        # unfies hour to 2-digit number
        df = self._unify_hour_format(data=df)
        # creates datetime pd.Series with entry times
        time = pd.to_datetime(
            df[[self.DATE_COLUMN, self.HOUR_COLUMN]].agg(" ".join, axis=1),
            format=self.DATE_FORMAT,
        )
        # shifts midnight entries day to the following one
        time = self._shift_midnight_date(time=time)
        df[data_ns.TIME] = time
        return df

    def _remove_timezone_shifts(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Removes entries that contain duplicated times.
        It happens because of timezone shift in Poland on the last Sunday of October.
        The timezone is changed back to winter time (UTC+01:00) every year,
        2 AM hour is duplicated and marked 'A'.
        """
        df = data.copy()
        mask = df[self.HOUR_COLUMN].apply(lambda x: TIMEZONE_MARK in str(x))
        df = df.drop(index=df[mask].index)
        return df

    def _replace_midnight_entries(self, data: pd.DataFrame) -> pd.DataFrame:
        """Replaces midnight hours that are marked as 24 with 0 hour"""
        midnight = "24"
        df = data.copy()
        df[[self.HOUR_COLUMN]] = df[[self.HOUR_COLUMN]].replace({midnight: "0"})
        return df

    def _shift_midnight_date(self, time: pd.Series) -> pd.Series:
        """
        Replaces day with the following for every observation with midnight hour.
        This operation is done, because midnight hour is marked as 24 in
        input data and was replaced with 0, one day must be added.
        """
        time = time.copy()
        mask = time.dt.hour == 0
        time.loc[mask] = time.loc[mask].apply(lambda x: x + timedelta(days=1))
        return time

    def _unify_hour_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """Unifies hour format to two digits, hours between 0 and 9 starting with 0"""
        df = data.copy()
        df[self.HOUR_COLUMN] = df[self.HOUR_COLUMN].apply(
            lambda x: x if len(x) > 1 else f"0{x}"
        )
        return df

    def _format_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Formats numeric columns and converts their data type to numeric"""
        df = data.copy()
        replace = {" ": "", "\xa0": "", ",": ".", "-": ""}
        df[self._meta.numeric_cols] = (
            df[self._meta.numeric_cols]
            .replace(replace, regex=True)
            .apply(pd.to_numeric)
        )
        return df
