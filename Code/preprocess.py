# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     # Developer details: 
        # Name: Harshita Jangde and Prachi Tavse
        # Role: Architects
    # Version:
        # Version: V 1.0 (20 September 2024)
            # Developers: Harshita Jangde and Prachi Tavse
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet preprocesses the data for a machine learning model by scaling
    # numerical columns, encoding categorical columns, and extracting date components for before feeding it to train the model
        # PostgreSQL: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.10.8   
            # Pandas 2.0.3
            # Scikit-learn 1.4.2

import pandas as pd     # Importing pandas for data manipulation
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Importing tools for data preprocessing

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
        'merchant_category', 'card_type', 'transaction_location', 'cardholder_gender', 'transaction_description'
    ]

    # Create a temporary DataFrame for scaling
    temp_data = data[numerical_cols].copy()
    temp_data.columns = temp_data.columns.str.strip()

    scaler = StandardScaler() # Initialize the StandardScaler
    temp_data = pd.DataFrame(scaler.fit_transform(temp_data)) # Scale numerical columns

    # Encode categorical columns
    encoder = LabelEncoder() # Initialize the LabelEncoder
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col]) # Encode categorical columns

    # Rejoin transaction_id and scaled numerical columns
    data = data.drop(columns=numerical_cols) # Drop original numerical columns
    data = pd.concat([data, temp_data], axis=1) # Concatenate scaled numerical columns back
    data['transaction_id'] = transaction_id # Reassign transaction_id

    return data


def load_and_preprocess_data(postgres_username, postgres_password, postgres_host, postgres_port, postgres_database):

    # Load data from PostgreSQL
    postgres_engine = create_engine(f'postgresql://{postgres_username}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_database}')
    data_postgres = pd.read_sql_table('transaction_data', postgres_engine) # Load PostgreSQL data

    # Preprocess data
    data_postgres_processed = preprocess_postgres_data(data_postgres) # Preprocess PostgreSQL data

    return data_postgres_processed
