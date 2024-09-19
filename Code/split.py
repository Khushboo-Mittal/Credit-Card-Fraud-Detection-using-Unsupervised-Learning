# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Developer details: 
        # Name: Prachi and Harshita
        # Role: Architects
        # Code ownership rights: Mohini T and Vansh R
    # Version:
        # Version: V 1.0 (17 September 2024)
            # Developers: Prachi and Harshita
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet preprocesses input data for a machine learning model by scaling
    # numerical columns, encoding categorical columns, and extracting date components for further analysis.
        # MongoDB: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Pandas 2.2.2
            # Scikit-learn 1.5.0

import pandas as pd                                     # For data manipulation
from sklearn.model_selection import train_test_split    # To split data into train, test, validation, and super validation sets
from pymongo import MongoClient                         # For using MongoDB as a cache to store the split data
import pickle                                           # For serializing and deserializing data for storage in Redis

# Importing necessary .py files and functions
from preprocess import load_and_preprocess_data # For preprocessing data

def connect_to_mongodb(host, port, db_name):
    # Connect to MongoDB
    client = MongoClient(host=host, port=port)
    db = client[db_name]
    return db

def drop_transaction_id_column(preprocessed_data):
    # Drop 'transaction_id' column if it exists
    preprocessed_data = preprocessed_data.drop(columns=['transaction_id'], errors='ignore')
    return preprocessed_data  # Return data without 'transaction_id' column

def save_preprocessed_data(preprocessed_data):
    # Save merged data as csv for inspection
    preprocessed_data.to_csv('preprocessed_data.csv', index=False)  # Save to CSV

def split(preprocessed_data):
    # Split features and target (assuming 'churn' is the target column)
    X = preprocessed_data[:] # Features
    
    # Split the data into train, test, validation, and super validation sets
    X_train, X_temp = train_test_split(X, test_size=0.4, random_state=42)  # 60% train, 40% temp
    X_test, X_temp = train_test_split(X_temp, test_size=0.625, random_state=42)  # 0.625 * 0.4 = 0.25 for test
    X_val, X_superval= train_test_split(X_temp, test_size=0.5, random_state=42)  # 0.4 * 0.25 = 0.1 for validation and super validation
    return X_train, X_test, X_val, X_superval,   # Return split data

def store_to_mongo(data, db, collection_name): # Store data into MongoDB
    collection = db[collection_name] # Select the collection
    collection.insert_one({'data': data}) # Insert the data into the collection
    
def save_split_data(db, X_train, X_test, X_val, X_superval):
    # Store to MongoDB
    store_to_mongo(pickle.dumps(X_train),db, 'x_train')
    store_to_mongo(pickle.dumps(X_test),db,'x_test')
    store_to_mongo(pickle.dumps(X_val),db,'x_val')
    store_to_mongo(pickle.dumps(X_superval) ,db,'x_superval')


def split_data(mongodb_host, mongodb_port, mongodb_db, data_postgres_processed):
    
    # Connect to Redis
    db = connect_to_mongodb(mongodb_host, mongodb_port, mongodb_db)
   
    
    # Drop 'customer_id' column
    preprocessed_data = drop_transaction_id_column(data_postgres_processed)
    
    # Uncomment the below line to see how the merged processed data looks
    save_preprocessed_data(preprocessed_data)
    
    # Split data
    X_train, X_test, X_val, X_superval = split(preprocessed_data)
    
    # Save split data
    save_split_data(db, X_train, X_test, X_val, X_superval)
    
    print('Data preprocessed, and split successfully!')
