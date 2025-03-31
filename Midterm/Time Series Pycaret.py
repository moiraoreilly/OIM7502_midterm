"""Name: Moira O'Reilly
Library: PyCaret
URL: https://pycaret.gitbook.io/docs
Description: PyCaret is a low-code machine learning library in Python that automates
machine learning workflows. It is essentially a Python wrapper around several machine learning
libraries (XGBoost, LightGBM, scikit-learn). It allows less experienced data scientists to
perform tasks that would have previously required more expertise.
Time series forecasting is a technique for predicting future events based on historical data."""

# Importing the necessary libraries
from pycaret.datasets import get_data
from pycaret.time_series import *

# Loading the datasets
airline = get_data('airline')

# Setting up Time Series Experiment
time = setup(data=airline, fh=12)
time_best_model = compare_models()

# Predicting next 12 months
future_forecast = predict_model(time_best_model, fh=12)
print(future_forecast)

# Plotting the forecast
plot_model(time_best_model, plot='forecast')

# Saving the model
save_model(time_best_model, 'timeseries_forecast_model')



