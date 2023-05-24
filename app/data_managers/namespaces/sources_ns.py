from argparse import Namespace

from .data_ns import data_ns

sources_ns = Namespace(
    ENERGY_DEMAND={
        data_ns.NUMERIC_COLUMNS: [data_ns.FORECAST, data_ns.VALUE],
        data_ns.RENAMES: {
            "Dobowa prognoza zapotrzebowania KSE": data_ns.FORECAST,
            "Rzeczywiste zapotrzebowanie KSE": data_ns.VALUE,
            "Godz.": data_ns.HOUR,
            "Data": data_ns.DATE,
        },
        data_ns.FREQ: "H",
    },
    ENERGY_MARKET_PRICE={
        data_ns.NUMERIC_COLUMNS: [data_ns.VALUE],
        data_ns.RENAMES: {
            "RCE": data_ns.VALUE,
            "Godzina": data_ns.HOUR,
            "Data": data_ns.DATE,
        },
        data_ns.FREQ: "H",
    },
    ENERGY_SETTLEMENT_PRICE={
        data_ns.NUMERIC_COLUMNS: [data_ns.VALUE],
        data_ns.RENAMES: {
            "CRO": data_ns.VALUE,
            "Godzina": data_ns.HOUR,
            "Data": data_ns.DATE,
        },
        data_ns.FREQ: "H",
    },
    WEATHER={
        data_ns.NUMERIC_COLUMNS: [
            "Temperature",
            "Wind_Speed",
        ],
        data_ns.RENAMES: {
            "Wind Speed": "Wind_Speed",
        },
        data_ns.FREQ: "H",
    },
    FUEL_PRICES={
        data_ns.NUMERIC_COLUMNS: [data_ns.VALUE],
        data_ns.RENAMES: {"Data zmiany": data_ns.TIME, "Cena": data_ns.VALUE},
        data_ns.FREQ: "D",
    },
    CO2_SETTLEMENT_PRICES={
        data_ns.NUMERIC_COLUMNS: [data_ns.VALUE],
        data_ns.RENAMES: {"Unnamed: 0": data_ns.TIME, "[zł/Mg CO2]": data_ns.VALUE},
        data_ns.FREQ: "D",
    },
)
