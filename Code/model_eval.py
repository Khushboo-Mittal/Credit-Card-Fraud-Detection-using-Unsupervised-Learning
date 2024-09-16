# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Mohini T and Vansh R
        # Role: Architects
        # Code ownership rights: Mohini T and Vansh R
    # Version:
        # Version: V 1.0 (11 July 2024)
            # Developers: Mohini T and Vansh R
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet contains utility functions to evaluate a model using test, validation,
    # and super validation data stored in a Redis database.
        # Redis: Yes
     
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
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix # For model evaluation

# Load test, validation, and super validation data from MongoDB
def load_data_from_mongodb(collection_name,db):
    # Connect to MongoDB
    collection = db[collection_name]
    data = collection.find_one()  # Retrieve the first document
    if data and 'pickled_data' in data:
        return pickle.loads(data['pickled_data'])  # Deserialize the pickled binary data
    return None

def evaluate_test_data(X_test, y_test, model):
    # Predict labels for the test set
    y_pred_test = model.predict(X_test)
    # Calculate accuracy score for the test set
    test_accuracy = accuracy_score(y_test, y_pred_test)
    # Calculate ROC AUC score for the test set
    if hasattr(model, "predict_proba"):
        test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    else:
        test_roc_auc = None
    # Return accuracy, ROC AUC, confusion matrix, and classification report for the test set
    return test_accuracy, test_roc_auc, confusion_matrix(y_test, y_pred_test), classification_report(y_test, y_pred_test)

def evaluate_validation_data(X_val, y_val, model):
    # Predict labels for the validation set
    y_pred_val = model.predict(X_val)
    # Calculate accuracy score for the validation set
    val_accuracy = accuracy_score(y_val, y_pred_val)
    # Calculate ROC AUC score for the validation set
    if hasattr(model, "predict_proba"):
        val_roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    else:
        val_roc_auc = None
    # Return accuracy, ROC AUC, confusion matrix, and classification report for the validation set
    return val_accuracy, val_roc_auc, confusion_matrix(y_val, y_pred_val), classification_report(y_val, y_pred_val)

def evaluate_supervalidation_data(X_superval, y_superval, model):
    # Predict labels for the super validation set
    y_pred_superval = model.predict(X_superval)
    # Calculate accuracy score for the super validation set
    superval_accuracy = accuracy_score(y_superval, y_pred_superval)
    # Calculate ROC AUC score for the super validation set
    if hasattr(model, "predict_proba"):
        superval_roc_auc = roc_auc_score(y_superval, model.predict_proba(X_superval)[:, 1])
    else:
        superval_roc_auc = None
    # Return accuracy, ROC AUC, confusion matrix, and classification report for the super validation set
    return superval_accuracy, superval_roc_auc, confusion_matrix(y_superval, y_pred_superval), classification_report(y_superval, y_pred_superval)

def evaluate_model(redis_host, redis_port, redis_db, model_path):

    X_test = load_data_from_mongodb('X_test')
    y_test = load_data_from_mongodb('y_test')
    X_val = load_data_from_mongodb('X_val')
    y_val = load_data_from_mongodb('y_val')
    X_superval = load_data_from_mongodb('X_superval')
    y_superval = load_data_from_mongodb('y_superval')
    
    # Ensure column names are strings for consistency
    X_test = X_test.rename(str, axis="columns")
    X_val = X_val.rename(str, axis="columns")
    X_superval = X_superval.rename(str, axis="columns")

    # Load the best model from the pickle file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Evaluate the model on test data
    test_accuracy, test_roc_auc, test_confusion_matrix, test_classification_report = evaluate_test_data(X_test, y_test, model)
    # Evaluate the model on validation data
    val_accuracy, val_roc_auc, val_confusion_matrix, val_classification_report = evaluate_validation_data(X_val, y_val, model)
    # Evaluate the model on super validation data
    superval_accuracy, superval_roc_auc, superval_confusion_matrix, superval_classification_report = evaluate_supervalidation_data(X_superval, y_superval, model)
    
    # Return evaluation metrics for test, validation, and super validation data
    return test_accuracy, test_roc_auc, test_confusion_matrix, test_classification_report, val_accuracy, val_roc_auc, val_confusion_matrix, val_classification_report, superval_accuracy, superval_roc_auc, superval_confusion_matrix, superval_classification_report