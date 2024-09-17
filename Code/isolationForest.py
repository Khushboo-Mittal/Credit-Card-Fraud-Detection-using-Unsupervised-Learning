# Import necessary libraries
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from pymongo import MongoClient

def read_training_data(db):
    # Access the collections
    x_train_collection = db['x_train']
    x_test_collection = db['x_test']

    # Load data into Pandas DataFrames
    x_train_df = pd.DataFrame(list(x_train_collection.find()))
    x_test_df = pd.DataFrame(list(x_test_collection.find()))


def perform_hyperparameter_tuning(X_train):
    # Define the grid of hyperparameters to search over
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of base estimators in ensemble
        'max_samples': [0.1, 0.5, 1],  # Maximum number of samples to draw from the dataset
        'max_features':[None,0.5,1], #Number of Features to draw from dataset to train each base estimator
        'contamination': [0.01, 0.1, 0.2],  # Proportion of outliers in the sample
        'behaviour': ['new', 'deprecated'],  # Specify the behaviour of the model
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    }
    
    # Initialize an IsolationForest Ensemble
    model = IsolationForest(random_state=42)
    
    # Initialize a GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2)
    
    # Fit the GridSearchCV to the training data
    grid_search.fit(X_train)
    
    # Retrieve the best model with the optimal hyperparameters
    best_model = grid_search.best_estimator_
    
    # Print the best hyperparameters found
    print("Best hyperparameters:", grid_search.best_params_)
    
    # Return the best model
    return best_model, grid_search.best_params_


