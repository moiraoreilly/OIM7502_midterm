"""Name: Moira O'Reilly
Library: PyCaret
URL: https://pycaret.gitbook.io/docs
Description: PyCaret is a low-code machine learning library in Python that automates
machine learning workflows. It is essentially a Python wrapper around several machine learning
libraries (XGBoost, LightGBM, scikit-learn). It allows less experienced data scientists to
perform tasks that would have previously required more expertise.
Clustering is an unsupervised machine learning technique designed to group unlabeled
examples based on their similarity to each other."""

# Importing the necessary libraries
from pycaret.datasets import get_data
from pycaret.clustering import *

# Loading the datasets
health = get_data('public_health')

# Setting up Clustering Experiment
clustering = setup(data=health)
kmeans = create_model('kmeans')
kmeans_results = assign_model(kmeans)
plot_model(kmeans, plot='cluster')
save_model(kmeans, 'saved_kmeans_model')

