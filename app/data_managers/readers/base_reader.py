from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import pandas as pd

from ..namespaces import data_ns
from ..utils import SourceMetaData
from ..utils.data_checker import check_data
from .scraping.base_scraper import BaseScraper

SOURCE = Union[str, list, Path]


class BaseReader(ABC):
    """Abstract class for reading and formatting data sources"""

    DATE_FORMAT = "%Y-%m-%d %H"
    _READ_KWARGS = {}

    def __init__(self, source: str) -> None:
        """
        Args:
            source (str): data source name
        """
        # gets source metadata from sources_ns.py
        self._meta = SourceMetaData(source=source)
        super().__init__()

    def _format(self, data: pd.DataFrame) -> pd.DataFrame:
        """Formats raw data specifically to data source"""
        return data

    @abstractmethod
    def _read_source(self, source: Union[str, Path]) -> pd.DataFrame:
        """Reads input data file and returns raw DataFrame"""
        ...

    def read(self, source: SOURCE) -> pd.DataFrame:
        """Reads and formats input data

        Args:
            source: list or pathlike
                List of strings or Path objects with input data.
                Single non-list object is also acceptable.

        Returns:
            pd.DataFrame: Formatted DataFrame
        """
        df = self._read(source=source)
        df = self._rename_columns(df)
        df = self._cast_to_str(df)
        df = self._format(df)
        df = self._drop_redundant_columns(df)
        df = self._set_time_index(df)
        df = self._drop_duplicated_index(df)
        check_data(df)
        return df

    def _read(self, source: SOURCE) -> pd.DataFrame:
        if not isinstance(source, list):
            source = [source]
        df = pd.concat((self._read_source(file) for file in source), ignore_index=True)
        return df

    def _cast_to_str(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign object type to every column that is not specified as numeric"""
        df = data.copy()
        non_num = list(filter(lambda x: x not in self._meta.numeric_cols, df.columns))
        df[non_num] = df[non_num].astype(str)
        return df

    def _rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Renames columns names using source specific rename dict"""
        df = data.copy()
        df = df.rename(self._meta.renames, axis=1)
        return df

    def _drop_redundant_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Drops redundant columns that are not numeric, id or time"""
        df = data.copy()
        valid = self._meta.numeric_cols + [data_ns.TIME]
        redundant = filter(lambda x: x not in valid, df.columns)
        df = df.drop(redundant, axis=1)
        return df

    def _set_time_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Sets time columns as DataFrame index"""
        df = data.copy()
        df[data_ns.TIME] = pd.to_datetime(df[data_ns.TIME], format=self.DATE_FORMAT)
        df = df.set_index(data_ns.TIME).sort_index()
        return df

    def _drop_duplicated_index(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.groupby(level=0).last()


class CSVReader(BaseReader):
    """Abstract class for reading csv files"""

    def _read_source(self, source: str) -> pd.DataFrame:
        return pd.read_csv(source, encoding="windows-1252", **self._READ_KWARGS)


class ExcelReader(BaseReader):
    """Abstract class for reading excel files"""

    def _read_source(self, source: str) -> pd.DataFrame:
        return pd.read_excel(source, engine="openpyxl", **self._READ_KWARGS)


class WebScraper(BaseReader):
    """Abstract class for scraping data from Web"""

    def __init__(self, source: str, scraper: BaseScraper) -> None:
        super().__init__(source)
        self._scraper = scraper
