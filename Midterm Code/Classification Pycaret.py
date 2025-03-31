"""Name: Moira O'Reilly
Library: PyCaret
URL: https://pycaret.gitbook.io/docs
Description: PyCaret is a low-code machine learning library in Python that automates
machine learning workflows. It is essentially a Python wrapper around several machine learning
libraries (XGBoost, LightGBM, scikit-learn). It allows less experienced data scientists to
perform tasks that would have previously required more expertise.
Classification is a type of supervised learning algorithm that is used
to predict the category of a target variable."""

# Importing the necessary libraries
from pycaret.datasets import get_data
from pycaret.classification import *


# Loading the datasets
juice = get_data('juice')


# Setting up Classification Experiment
classification = setup(data=juice, target='Purchase')
c_model = create_model('lr')
c_best_model = compare_models()
c_tuned_model = tune_model(c_best_model)
c_evaluate_model = evaluate_model(c_tuned_model)

# Save the model
save_model(c_tuned_model, 'best_classification_model')

