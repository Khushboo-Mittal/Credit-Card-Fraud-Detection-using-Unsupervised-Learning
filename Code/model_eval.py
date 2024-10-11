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
     
    # Description: This code snippet contains utility functions to evaluate a model using test, validation,
    # and super validation data stored in a MongoDB database.
        # MongoDB: Yes
     
# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:
            # Python 3.11.5   
            # Pandas 2.2.2
            # Scikit-learn 1.5.0

# Import necessary libraries
from pymongo import MongoClient # For connecting to MongoDB database
import pickle # For loading the model from a pickle file
import pandas as pd # For data manipulation
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score # For model evaluation

# Load test, validation, and super validation data from MongoDB
def load_data_from_mongodb(db, collection_name):
    # Connect to MongoDB
    collection = db[collection_name]
    data = collection.find_one()  # Retrieve the first document
    if data and 'data' in data:
        return pickle.loads(data['data'])  # Deserialize the pickled binary data
    return None

def evaluate_test_data(X_test, model,flag):
    # Predict labels for the test set
    if flag==1:
        pred = model.fit_predict(X_test)
    else:
        pred = model.predict(X_test)
    
    # Calculate accuracy score for the test set
    test_silhouette_avg = silhouette_score(X_test, pred)

    # Calculate Davies Bouldin index or DBI
    test_db_index = davies_bouldin_score(X_test, pred)

    # Calculate Calinski Harabasz Index
    test_ch_index = calinski_harabasz_score(X_test, pred)
    
    return test_silhouette_avg, test_db_index, test_ch_index, # test_explained_variance

def evaluate_validation_data(X_val, model,flag):
    # Predict labels for the validation set
    if flag==1:
        val_pred = model.fit_predict(X_val)
    else:
        val_pred = model.predict(X_val)
    
    # Calculate accuracy score for the validation set
    val_silhouette_avg = silhouette_score(X_val, val_pred)

    # Calculate Davies Bouldin index or DBI for the validation set
    val_db_index = davies_bouldin_score(X_val, val_pred)

    # Calculate Calinski Harabasz Index for the validation set
    val_ch_index = calinski_harabasz_score(X_val, val_pred)
    
    return val_silhouette_avg, val_db_index, val_ch_index, # val_explained_variance

def evaluate_supervalidation_data(X_superval, model,flag):
    # Predict labels for the supervalidation set
    if flag==1:
        superval_pred = model.fit_predict(X_superval)
    else:
        superval_pred = model.predict(X_superval)
    
    # Calculate accuracy score for the supervalidation set
    superval_silhouette_avg = silhouette_score(X_superval, superval_pred)

    # Calculate Davies Bouldin index or DBI for the supervalidation set
    superval_db_index = davies_bouldin_score(X_superval, superval_pred)

    # Calculate Calinski Harabasz Index for the supervalidation set
    superval_ch_index = calinski_harabasz_score(X_superval, superval_pred)
    
    return superval_silhouette_avg,  superval_db_index, superval_ch_index, # superval_explained_variance

def evaluate_model(mongodb_host, mongodb_port, mongodb_db, model_path):
    client = MongoClient(host=mongodb_host, port=mongodb_port)
    db = client[mongodb_db]
    
    X_test = load_data_from_mongodb(db, 'x_test')
    X_val = load_data_from_mongodb(db, 'x_val')
    X_superval = load_data_from_mongodb(db, 'x_superval')
    
    # Ensure column names are strings for consistency
    X_test = X_test.rename(str, axis="columns")
    X_val = X_val.rename(str, axis="columns")
    X_superval = X_superval.rename(str, axis="columns")
    
    X_test = X_test.values
    X_val = X_val.values
    X_superval = X_superval.values
    

    # Load the best model from the pickle file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    if 'local_outlier_factor_model.pkl' in model_path:
        flag = 1
    else:
        flag=0

    # Evaluate the model on test data
    test_silhouette_avg, test_db_index, test_ch_index = evaluate_test_data(X_test, model,flag)
    # Evaluate the model on validation data
    val_silhouette_avg, val_db_index, val_ch_index = evaluate_validation_data(X_val, model,flag)
    # Evaluate the model on super validation data
    superval_silhouette_avg, superval_db_index, superval_ch_index = evaluate_supervalidation_data(X_superval, model,flag)
    
    # Return evaluation metrics for test, validation, and super validation data
    return test_silhouette_avg,  test_db_index, test_ch_index, val_silhouette_avg, val_db_index, val_ch_index, superval_silhouette_avg, superval_db_index, superval_ch_index