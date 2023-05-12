import os
import shutil
from abc import ABC, abstractmethod

import pandas as pd

from ..namespaces import data_ns
from .utils import create_directory


class BaseUploader(ABC):
    _ext: str

    def __init__(self, file: str, copy: bool = True) -> None:
        self.file = file
        self._copy = copy
        assert file.endswith(self._ext), f"File must be of {self._ext} format"
        self._exist = os.path.exists(file)
        create_directory(file)

    def upload(self, data: pd.DataFrame) -> None:
        existing = self._read() if self._exist else pd.DataFrame()
        new = data.loc[~data.index.isin(existing.index)]
        concat = pd.concat((existing, new)).sort_index()
        if self._exist and self._copy:
            self._copy_file()
        self._upload(concat)

    def _copy_file(self) -> None:
        path = self.file.replace(f".{self._ext}", f"_copy.{self._ext}")
        shutil.copy(self.file, path)

    @abstractmethod
    def _upload(self, data: pd.DataFrame) -> None:
        ...

    @abstractmethod
    def _read(self) -> pd.DataFrame:
        ...


class CSVUploader(BaseUploader):
    _ext = "csv"

    def _read(self) -> pd.DataFrame:
        df = pd.read_csv(self.file, parse_dates=[data_ns.TIME], index_col=data_ns.TIME)
        return df

    def _upload(self, data: pd.DataFrame) -> None:
        data.to_csv(self.file)


class ExcelUploader(BaseUploader):
    _ext = ".xlsx"

    def _read(self) -> pd.DataFrame:
        df = pd.read_excel(
            self.file, parse_dates=[data_ns.TIME], index_col=data_ns.TIME
        )
        return df

    def _upload(self, data: pd.DataFrame) -> None:
        data.to_excel(self.file)
