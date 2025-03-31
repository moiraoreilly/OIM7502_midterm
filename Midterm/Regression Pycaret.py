"""Name: Moira O'Reilly
Library: PyCaret
URL: https://pycaret.gitbook.io/docs
Description: PyCaret is a low-code machine learning library in Python that automates
machine learning workflows. It is essentially a Python wrapper around several machine learning
libraries (XGBoost, LightGBM, scikit-learn). It allows less experienced data scientists to
perform tasks that would have previously required more expertise.
Regression is a type of supervised learning algorithm that is used to
predict the continuous value of a target variable."""

# Importing the necessary libraries
from pycaret.datasets import get_data
from pycaret.regression import *

# Loading the datasets
insurance = get_data('insurance')

# Setting up Regression Experiment
regression = setup(data=insurance, target='charges')
r_model = create_model('lr')
r_best_model = compare_models()
r_tuned_model = tune_model(r_best_model)

# Predict the model
predictions = predict_model(r_tuned_model, data=insurance)

# Save the model
save_model(r_tuned_model, 'best_regression_model')
