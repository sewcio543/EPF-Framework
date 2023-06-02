import os
from argparse import Namespace

data_ns = Namespace(
    TIME="TIME",
    VALUE="VALUE",
    FORECAST="FORECAST",
    NUMERIC_COLUMNS="NUMERIC_COLUMNS",
    RENAMES="RENAMES",
    DATE="DATE",
    HOUR="HOUR",
    FREQ="FREQ",
)

_HOLIDAYS = os.path.join("HOLIDAYS", "holidays.csv")

files_ns = Namespace(
    DATA_FOLDER="data",
    CURATED_FOLDER="CURATED",
    HOLIDAYS=_HOLIDAYS,
    FUEL_PRICES="FUEL_PRICES.csv",
    CO2_PRICES="CO2_SETTLEMENT_PRICES.csv",
    ENERGY_DEMAND="ENERGY_DEMAND.csv",
    ENERGY_PRICE="ENERGY_SETTLEMENT_PRICE.csv",
    WEATHER="WEATHER.csv",
)
