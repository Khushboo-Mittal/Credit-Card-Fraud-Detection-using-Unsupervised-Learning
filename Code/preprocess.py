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
     
    # Description: This code snippet contains utility functions to evaluate a model using test, validation,
    # and super validation data stored in a MongoDB database.
        # PostgreSQL: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Pandas 2.2.2
            # Scikit-learn 1.5.0

import pandas as pd                                             # Importing pandas for data manipulation
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Importing tools for data preprocessing
import db_utils                                                 # Importing utility functions for database operations
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_postgres_data(data):
    # Separate transaction_id
    transaction_id = data['transaction_id']
    # Define columns to be scaled, excluding 'transaction_id'
    
    #Convert to datetime format
    data['transaction_date'] = pd.to_datetime(data['transaction_date'],format='%d-%m-%Y')

    # Extract components
    data['transaction_year'] = data['transaction_date'].dt.year
    data['transaction_month'] = data['transaction_date'].dt.month
    data['transaction_day'] = data['transaction_date'].dt.day

    # Drop the transaction_date column
    data = data.drop('transaction_date', axis=1)
    numerical_cols = [
        'transaction_amount', 'cardholder_age', 'account_balance', 'calander_income','transaction_year','transaction_month','transaction_day'
    ]

    categorical_cols = [
        'merchant_category', 'card_type', 'transaction_location', 'cardholder_gender', 
    ]

    # Create a temporary DataFrame for scaling
    temp_data = data[numerical_cols].copy()

    scaler = StandardScaler() # Initialize the StandardScaler
    temp_data = pd.DataFrame(scaler.fit_transform(temp_data), columns=numerical_cols) # Scale numerical columns

    # Encode categorical columns
    encoder = LabelEncoder() # Initialize the LabelEncoder
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col]) # Encode categorical columns

    # Rejoin transaction_id and scaled numerical columns
    data = data.drop(columns=numerical_cols) # Drop original numerical columns
    data = pd.concat([data, temp_data], axis=1) # Concatenate scaled numerical columns back
    # data['transaction_id'] = transaction_id # Reassign transaction_id


    # # Convert text descriptions into numerical features
    # Initialize the TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=100)

    # Transform the transaction_description column
    description_features = tfidf.fit_transform(data['transaction_description'])

    # Convert to dataframe
    description_data = pd.DataFrame(description_features.toarray(), columns=tfidf.get_feature_names_out())

    # Rejoin the data and drop original column
    data = pd.concat([data, description_data], axis=1)
    data = data.drop('transaction_description', axis=1, inplace=False)
    return data


def load_and_preprocess_data(postgres_username, postgres_password, postgres_host, postgres_port, postgres_database):

    # Load data from PostgreSQL
    postgres_engine = db_utils.connect_postgresql(postgres_username, postgres_password, postgres_host, postgres_port, postgres_database)
    data_postgres = pd.read_sql_table('transaction_data', postgres_engine) # Load PostgreSQL data

    # Preprocess data
    data_postgres_processed = preprocess_postgres_data(data_postgres) # Preprocess PostgreSQL data

    return data_postgres_processed
