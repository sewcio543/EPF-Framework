from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Callable

import pandas as pd
from matplotlib import pyplot as plt

from app.data_managers.namespaces import data_ns, files_ns

from ..data_managers.uploaders.utils import create_directory
from .plotting import plot_forecast


def get_time() -> str:
    return datetime.now().strftime("%d-%m_%H_%M_%S")


def save_results(f: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    @wraps(f)
    def wrapper(*args, **kwargs):
        out = f(*args, **kwargs)
        folder = Path(files_ns.DATA_FOLDER, files_ns.RESULTS_FOLDER)
        create_directory(folder)

        now = get_time()
        path = folder / f"{now}.csv"
        out.to_csv(path)
        return out

    return wrapper


def save_plots(
    slice: slice = slice(None), freq: str = "H"
) -> Callable[..., pd.DataFrame]:
    def inner(f: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        @wraps(f)
        def wrapper(*args, **kwargs) -> pd.DataFrame:
            out = f(*args, **kwargs)

            folder = Path(files_ns.DATA_FOLDER, files_ns.PLOTS_FOLDER)
            create_directory(folder)

            for model in out.columns:
                if model == data_ns.ACTUAL:
                    continue
                plot_forecast(
                    out[model].iloc[slice],
                    out[data_ns.ACTUAL].iloc[slice],
                    freq=freq,
                    model=model,
                )
                now = get_time()
                path = folder / f"{model}_{now}.png"
                plt.savefig(path)
            return out

        return wrapper

    return inner  # type: ignore
