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
        data_ns.FREQ: "D",
    },
    ENERGY_PRICE={
        data_ns.NUMERIC_COLUMNS: [data_ns.VALUE],
        data_ns.RENAMES: {
            "RCE": data_ns.VALUE,
            "Godzina": data_ns.HOUR,
            "Data": data_ns.DATE,
        },
        data_ns.FREQ: "D",
    },
    WEATHER={
        data_ns.NUMERIC_COLUMNS: [
            "Precipitation",
            "Wind_Blow",
            "Wind_Speed",
            "Temperature",
            "Visibility",
            "Humidity",
            "Overcast",
        ],
        data_ns.RENAMES: {
            "Podmuchy wiatru": "Wind_Blow",
            "Prędkość wiatru": "Wind_Speed",
            "Temperatura": "Temperature",
            "Widoczność": "Visibility",
            "Wilgotność": "Humidity",
            "Zachmurzenie": "Overcast",
            "Opad atmosferyczny": "Precipitation",
            "Time": data_ns.TIME,
        },
        data_ns.FREQ: "D",
    },
)
