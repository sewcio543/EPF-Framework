from abc import ABC, abstractmethod
from typing import Optional, Union

import pandas as pd

from ..namespaces import data_ns
from ..utils import SourceMetaData
from .scraping.base_scraper import BaseScraper


class BaseReader(ABC):
    """Abstract class for reading and formatting data sources"""

    DATE_FORMAT = "%Y-%m-%d %H"

    def __init__(self, source: str) -> None:
        """
        Args:
            source (str): data source name
        """
        # gets source metadata from sources_ns.py
        self._meta = SourceMetaData(source=source)
        super().__init__()

    @abstractmethod
    def _format(self, data: pd.DataFrame) -> pd.DataFrame:
        """Formats raw data specifically to data source"""
        ...

    @abstractmethod
    def _read_source(self, source: str) -> pd.DataFrame:
        """Reads input data file and returns raw DataFrame"""
        ...

    def read(self, source: Union[str, list[str]]) -> pd.DataFrame:
        """Reads and formats input data

        Args:
            file (str): path to file with input data

        Returns:
            pd.DataFrame: Formatted DataFrame
        """
        df = self._read(source=source)
        df = self._rename_columns(df)
        df = self._cast_to_str(df)
        df = self._format(df)
        df = self._drop_redundant_columns(df)
        self._check_data(df)
        df = self._set_time_index(df)
        df = self._drop_duplicated_index(df)
        return df

    def _read(self, source: Union[str, list[str]]) -> pd.DataFrame:
        if isinstance(source, str):
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

    def _check_data(self, data: pd.DataFrame) -> None:
        missing_cols = [
            col for col in self._meta.numeric_cols if col not in data.columns
        ]
        assert not missing_cols, f"{missing_cols} columns are missing"
        assert data_ns.TIME in data.columns, f"missing {data_ns.TIME} column"

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
        df = df.set_index(data_ns.TIME)
        return df

    def _drop_duplicated_index(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.groupby(level=0).last()


class CSVReader(BaseReader):
    """Abstract class for reading csv files"""

    _SEP = ","
    _READ_KWARGS = {}

    def _read_source(self, source: str) -> pd.DataFrame:
        kwargs = {"sep": self._SEP} | self._READ_KWARGS
        return pd.read_csv(source, **kwargs)


class ExcelReader(BaseReader):
    """Abstract class for reading excel files"""

    _READ_KWARGS = {}

    def _read_source(self, source: str) -> pd.DataFrame:
        return pd.read_excel(source, **self._READ_KWARGS)


class WebScraper(BaseReader):
    """Abstract class for scraping data from Web"""

    SOURCE: Union[str, list[str]]

    def __init__(
        self,
        source: str,
        driver: Optional[str] = None,
        headless: bool = True,
        wait_time: int = 5,
        verbose: bool = False,
        load_time: float = 0.1,
    ) -> None:
        super().__init__(source)
        self._scraper = BaseScraper(
            driver=driver,
            headless=headless,
            wait_time=wait_time,
            verbose=verbose,
            load_time=load_time,
        )
