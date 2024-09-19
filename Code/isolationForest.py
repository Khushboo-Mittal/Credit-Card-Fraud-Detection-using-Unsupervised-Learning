# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
from pymongo import MongoClient
from sklearn.metrics import make_scorer, silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pickle
def read_training_data(db):
    # Access the collections
    x_train_collection = db['x_train']
    x_test_collection = db['x_test']

    # Load data into Pandas DataFrames
    x_train_df = pd.DataFrame(list(x_train_collection.find()))
    x_test_df = pd.DataFrame(list(x_test_collection.find()))
    x_train_np = x_train_df.values
    x_test_np = x_test_df.values
    return x_train_np, x_test_np



def perform_hyperparameter_tuning(X_train):
    # Define the grid of hyperparameters to search over
    param_grid = {
        'n_estimators': [100, 50, 70],  # Number of base estimators in ensemble
        'max_samples': [0.1, 0.05, 0.75],  # Maximum number of samples to draw from the dataset
        'max_features':[0.05,0.5,0.7], #Number of Features to draw from dataset to train each base estimator
        'contamination': [0.01, 0.05, 0.02],  # Proportion of outliers in the sample
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    }
    
    # Initialize an IsolationForest Ensemble
    model = IsolationForest(random_state=42)
    def silhouette_scorer(estimator, X):
        cluster_labels = estimator.fit_predict(X)
        return silhouette_score(X,cluster_labels)
    # Initialize a GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=3, n_jobs=-1, verbose=2,scoring=silhouette_scorer)
    
    
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
        
        # Calculate the accuracy of the model on the training data
        superval_silhouette_avg = silhouette_score(X_train, y_pred_train)
        
        # Return the training accuracy
        return superval_silhouette_avg
    
    except Exception as e:
        # Print an error message if there is an issue during evaluation
        print("Error during model evaluation:", e)

def save_model(model, model_path):
    # Open a file in write-binary mode to save the model
    with open(model_path, "wb") as f:
        # Serialize the model and save it to the file
        pickle.dump(model, f)

# def load_from_mongo(db, collection_name):
#     collection = db[collection_name]
#     data_entry = collection.find_one({}, {'_id': 0, 'data': 1})
#     if data_entry:
#         return pickle.loads(data_entry['data'])
#     return None

def train_model(mongodb_host, mongodb_port, mongodb_db, model_path):
    
    # Connect to the Redis database
    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]
    
    # Load the training data from MongoDB
    X_train,X_test = read_training_data(db)
    
    
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
    
    # Return the training accuracy
    return superval_silhouette_avg, best_params

