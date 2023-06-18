import pandas as pd
from matplotlib import pyplot as plt


def plot_forecast(
    forecast: pd.Series, actuals: pd.Series, freq: str = "H", model: str = ""
):
    plt.figure(figsize=(8, 5))
    plt.plot(actuals.resample(freq).mean(), color="blue", label="Actuals")
    plt.plot(forecast.resample(freq).mean(), color="orange", label=model)
    plt.xticks(rotation=30, ha="right")

    plt.title("Electricity price forecasting")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.legend()
