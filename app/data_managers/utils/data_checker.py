import pandas as pd

from ..namespaces import data_ns


class DataModelException(Exception):
    ...


class IndexNameError(DataModelException):
    ...


class NotDatetimeIndexError(DataModelException):
    ...


class DuplicatedIndexError(DataModelException):
    ...


def check_data(data: pd.DataFrame) -> None:
    check_index(data.index)


def check_index(index: pd.Index) -> None:
    if index.name != data_ns.TIME:
        raise IndexNameError(
            f"Index name of data should be {data_ns.TIME}, got {index.name}"
        )

    if not isinstance(index, pd.DatetimeIndex):
        raise NotDatetimeIndexError("Index must be pd.DatetimeIndex")

    dupl = index.duplicated()
    if dupl.any():
        raise DuplicatedIndexError(
            f"DatetimeIndex contains duplicated entries {index[dupl]}"
        )
