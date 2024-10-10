# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Harshita and Prachi
        # Role: Architects
        # Code ownership rights: Mohini T and Vansh R
    # Version:
        # Version: V 1.0 (20 September 2024)
            # Developers: Harshita and Prachi
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet creates the Isolation Forest Model to train, evaluate, and predict if credit card is fraudulent according to Transaction behaviour.

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Streamlit 1.36.0
            
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from pymongo import MongoClient
from sklearn.metrics import  silhouette_score
import pickle

# Load test, validation, and super validation data from MongoDB
def load_data_from_mongodb(db, collection_name):
    # Connect to MongoDB
    collection = db[collection_name]
    data = collection.find_one()  # Retrieve the first document
    if data and 'data' in data:
        return pickle.loads(data['data'])  # Deserialize the pickled binary data
    return None



def perform_hyperparameter_tuning(X_train):
    # Define the grid of hyperparameters to search over
    param_grid = {
        'n_estimators': [100, 50, 70],  # Number of base estimators in ensemble
        'max_samples': [0.01, 0.05, 0.025],  # Maximum number of samples to draw from the dataset
        'max_features':[0.05,0.5,0.7], #Number of Features to draw from dataset to train each base estimator
        'contamination': [0.01, 0.05, 0.02],  # Proportion of outliers in the sample
        'bootstrap':  [False,False ] # Whether bootstrap samples are used when building trees
    }
    
    # Initialize an IsolationForest Ensemble
    model = IsolationForest(random_state=42)
    
    def silhouette_scorer(estimator, X):
        cluster_labels = estimator.fit_predict(X)
        return silhouette_score(X,cluster_labels)
    
    # Initialize a GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=2, n_jobs=-1, verbose=2,scoring=silhouette_scorer)
    
    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train)
    
    # Retrieve the best model with the optimal hyperparameters
    best_model = grid_search.best_estimator_
    
    # Print the best hyperparameters found
    print("Best hyperparameters:", grid_search.best_params_)
    
    # Return the best model
    return best_model, grid_search.best_params_

def evaluate_model(model, X_train):
    try:
        # Predict the training labels using the trained model
        y_pred_train = model.predict(X_train)
        
        # Calculate the silhouette score of the model on the training data
        superval_silhouette_avg = silhouette_score(X_train, y_pred_train)
        
        # Return the silhouette score
        return superval_silhouette_avg
    
    except Exception as e:
        # Print an error message if there is an issue during evaluation
        print("Error during model evaluation:", e)

def save_model(model, model_path):
    # Open a file in write-binary mode to save the model
    with open(model_path, "wb") as f:
        # Serialize the model and save it to the file
        pickle.dump(model, f)


def train_model(mongodb_host, mongodb_port, mongodb_db, model_path):
    
    # Connect to the MongoDB database
    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]
    
    # Load the training data from MongoDB
    X_train = load_data_from_mongodb(db,'x_train')
    X_train = X_train.values
    
    # Perform hyperparameter tuning to find the best model
    best_model, best_params = perform_hyperparameter_tuning(X_train)
   
    # Fit the best model to the training data
    best_model.fit(X_train)
    
    # Print a message indicating the model has been fitted successfully
    print("Best model fitted successfully.")
    
    # Evaluate the best model on the training data
    superval_silhouette_avg = evaluate_model(best_model, X_train)
    
    # Save the best model to a file
    save_model(best_model, model_path)
    
    # Print a message indicating the model training is completed
    print('Model training completed successfully!')
    
    # Return the silhouette score and best parameters
    return superval_silhouette_avg, best_params

