"""Name: Moira O'Reilly
Library: PyCaret
URL: https://pycaret.gitbook.io/docs
Description: PyCaret is a low-code machine learning library in Python that automates
machine learning workflows. It is essentially a Python wrapper around several machine learning
libraries (XGBoost, LightGBM, scikit-learn). It allows less experienced data scientists to
perform tasks that would have previously required more expertise.
Anomaly detection identifies data points or patterns that deviate from thr norm.
It is often used to detect problems like fraud or security breaches."""

# Importing the necessary libraries
from pycaret.datasets import get_data
from pycaret.anomaly import *

# Loading the datasets
anomaly = get_data('anomaly')

# Setting up Anomaly Detection Experiment
anomaly_model = setup(data=anomaly)
knn = create_model('knn')
knn_df = assign_model(knn)
plot_model(knn, plot='tsne')
evaluate_model(knn)

# Save the model
save_model(knn, 'best_anomaly_model')

