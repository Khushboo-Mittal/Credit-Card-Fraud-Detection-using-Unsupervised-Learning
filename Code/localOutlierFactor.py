import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from pymongo import MongoClient
from sklearn.metrics import silhouette_score
import joblib  # Import joblib for saving models
import pickle  # Import pickle for loading data
import streamlit as st  # Ensure you have Streamlit imported

def load_data_from_mongodb(db, collection_name):
    collection = db[collection_name]
    data = collection.find_one()  # Retrieve the first document
    if data and 'data' in data:
        return pickle.loads(data['data'])  # Deserialize the pickled binary data
    return None

def perform_hyperparameter_tuning_lof(X_train):
    param_grid = {
        'n_neighbors': [5, 10, 15],
        'contamination': [0.01, 0.05, 0.1]
    }
    
    model = LocalOutlierFactor()

    def silhouette_scorer(estimator, X):
        cluster_labels = estimator.fit_predict(X)
        return silhouette_score(X, cluster_labels)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=2, n_jobs=-1, verbose=2, scoring=silhouette_scorer)

    grid_search.fit(X_train)

    best_model = grid_search.best_estimator_

    print("Best hyperparameters for LOF:", grid_search.best_params_)

    return best_model, grid_search.best_params_

def evaluate_model(model, X_train):
    try:
        y_pred_train = model.fit_predict(X_train)
        superval_silhouette_avg = silhouette_score(X_train, y_pred_train)
        return superval_silhouette_avg
    except Exception as e:
        print("Error during model evaluation:", e)
        return None

def save_model(model, model_path):
    # Open a file in write-binary mode to save the model
    with open(model_path, "wb") as f:
        # Serialize the model and save it to the file
        pickle.dump(model, f)

def train_model(mongodb_host, mongodb_port, mongodb_db,model_path):
    # Access the One-Class SVM model path from Streamlit session state
    # model_path = st.session_state.oneclass_svm_path  # Get the path from Streamlit session state
    
    # Connect to the MongoDB database
    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]

    # Load the training data from MongoDB
    X_train = load_data_from_mongodb(db, 'x_train')
    
    if X_train is None:
        print("No training data found.")
        return None, None
    
    X_train = X_train.values  # Ensure it's a NumPy array

    # Perform hyperparameter tuning to find the best LOF model
    best_model, best_params = perform_hyperparameter_tuning_lof(X_train)
    
    # Fit the best model to the training data
    best_model.fit(X_train)
    
    print("Best LOF model fitted successfully.")

    # Evaluate the best model on the training data
    superval_silhouette_avg = evaluate_model(best_model, X_train)
    
    # Save the best model to a file using joblib
    save_model(best_model, model_path)
    
    print('Model training completed successfully!')

    return superval_silhouette_avg, best_params