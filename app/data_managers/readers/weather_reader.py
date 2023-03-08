import pandas as pd

from .base_reader import CSVReader

UNITS = ["%", "°  C", "Km/h", " Km", " mm", "°"]


class WeatherReader(CSVReader):
    def _format(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = self._format_numeric_columns(data=df)
        return df

    def _format_numeric_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Formats numeric columns and converts their data type to numeric"""
        df = data.copy()
        df = self._remove_units_and_signs(data=df)
        df[self._meta.numeric_cols] = df[self._meta.numeric_cols].apply(pd.to_numeric)
        return df

    def _remove_units_and_signs(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        replace = {unit: "" for unit in UNITS} | {",": "."}
        df = df.replace(replace, regex=True)
        # removes parenthesis
        df = df.applymap(lambda x: str(x).rstrip(")").lstrip("("))
        return df
