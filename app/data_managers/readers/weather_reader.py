import pandas as pd

from ..namespaces import data_ns
from .base_reader import CSVReader

UNITS = ["°F", "°%", "°mph", "°in", r"\xa0", "Â", " "]
TEMPERATURE = "Temperature"
WIND_SPEED = "Wind_Speed"


class WeatherReader(CSVReader):
    def _format(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = self._format_numeric_columns(data=df)
        df = self._format_date_column(df)
        return df

    def _format_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Formats numeric columns and converts their data type to numeric"""
        df = data.copy()
        df = self._remove_units_and_signs(data=df)
        df[self._meta.numeric_cols] = df[self._meta.numeric_cols].apply(pd.to_numeric)
        df = self._convert_units(df)
        return df

    def _remove_units_and_signs(self, data: pd.DataFrame) -> pd.DataFrame:
        """Removes suffixes and signs from data"""
        df = data.copy()
        df = df.replace(UNITS, "", regex=True)
        return df

    def _convert_units(self, data: pd.DataFrame) -> pd.DataFrame:
        """Converts units from american to standard metrics"""

        def to_celcius(o: float, dec: int = 0) -> float:
            num = ((o - 32) * 5) / 9
            return round(num, dec)

        def to_kmh(o: float, dec: int = 0) -> float:
            num = o * 1.609344
            return round(num, dec)

        df = data.copy()
        df[TEMPERATURE] = df[TEMPERATURE].apply(to_celcius)
        df[WIND_SPEED] = df[WIND_SPEED].apply(to_kmh)
        return df

    def _format_date_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """Formats date colum and drops redundant entries"""
        df = data.copy()
        date = df[data_ns.DATE] + " " + df["Time"]
        df[data_ns.TIME] = pd.to_datetime(date)
        df = self._drop_half_hours(df)
        return df

    def _drop_half_hours(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drops entries from incomplete hours like 1:30 AM"""
        df = data.copy()
        mask = df[data_ns.TIME].apply(lambda x: x.minute == 0)
        df = df.loc[mask]
        return df
