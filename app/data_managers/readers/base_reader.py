from abc import ABC, abstractmethod

import pandas as pd

from ..namespaces import data_ns
from ..utils import SourceMetaData


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
    def _read_file(self, file: str) -> pd.DataFrame:
        """Reads input data file and returns raw DataFrame"""
        ...

    def read(self, file: str) -> pd.DataFrame:
        """Reads and formats input data

        Args:
            file (str): path to file with input data

        Returns:
            pd.DataFrame: Formatted DataFrame
        """
        df = self._read_file(file=file)
        df = self._rename_columns(data=df)
        df = self._cast_to_str(data=df)
        df = self._format(data=df)
        df = self._drop_redundant_columns(data=df)
        df = self._set_time_index(data=df)
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
        df = df.set_index(data_ns.TIME)
        return df


class CSVReader(BaseReader):
    """Abstract class for reading csv files"""

    _SEP = ","
    _READ_KWARGS = {}

    def _read_file(self, file: str) -> pd.DataFrame:
        kwargs = {"sep": self._SEP} | self._READ_KWARGS
        return pd.read_csv(file, **kwargs)


class ExcelReader(BaseReader):
    """Abstract class for reading excel files"""

    _READ_KWARGS = {}

    def _read_file(self, file: str) -> pd.DataFrame:
        return pd.read_excel(file, **self._READ_KWARGS)
