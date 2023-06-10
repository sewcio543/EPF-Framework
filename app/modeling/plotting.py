import pandas as pd
from matplotlib import pyplot as plt


def plot_forecast(
    forecast: pd.Series, actuals: pd.Series, freq: str = "H", model: str = ""
):
    plt.figure(figsize=(8, 5))
    plt.plot(actuals.resample(freq).mean(), color="blue", label="Rzeczywista cena")
    plt.plot(forecast.resample(freq).mean(), color="orange", label=model)
    plt.xticks(rotation=30, ha="right")

    plt.title("Prognozy ceny energii elektrycznej")
    plt.xlabel("Data")
    plt.ylabel("Cena [zł]")
    plt.tight_layout()
    plt.legend()
