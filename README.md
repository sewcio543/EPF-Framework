# Electricity Price Forecasting Framework

This project is a comprehensive framework designed for electricity price forecasting.
It provides a set of powerful components and tools that facilitate various aspects of the forecasting process.

## Features

1. **Backtesting**: The framework enables you to assess the performance of different forecasting models.
It supports sktime models and metrics. Comes with decorators that enhance the functionality, saving plots and results.

2. **Feature Creation**: Wide range of feature creation tools, empowering you to extract meaningful information from raw electricity price data.
You can generate various exogenious features and other indicators that come in handy for modeling.

3. **Built-in datasets**: Comes with range of datasets with exogenious features that can be useful for modeling purposes.
There are components that facilitate their extraction and integration.
For more information check out modeling demo notebook.
Provided datasets are specifically sourced from Polish data sources.
By using these pre-loaded datasets, users can quickly dive into the forecasting process.

## Demos

* ingestion.ipynb - notebook for ingestion of data into curated folder
* modeling.ipynb - example of how to forecast electricity price with built-in components
