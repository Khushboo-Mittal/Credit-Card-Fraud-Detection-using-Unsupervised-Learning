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
     
    # Description: This code snippet ingests transaction data from a CSV file, preprocesses it, and stores it in
    # PostgreSQL database.
        # PostgreSQL: Yes 

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Pandas 2.2.2

import pandas as pd # Importing pandas for data manipulation
import db_utils # Importing utility functions for database operations

def ingest_data(data_path, postgres_username, postgres_password, postgres_host, postgres_port, postgres_database):
    
    data = pd.read_csv(data_path) # Read data from CSV file

    postgres_data = data[['transaction_id', 'transaction_date', 'transaction_amount', 'merchant_category', 'card_type', 'transaction_location',
                          'cardholder_age', 'cardholder_gender', 'transaction_description', 'account_balance', 'calander_income']]
    
    
    # Connect to PostgreSQL
    postgres_engine = db_utils.connect_postgresql(postgres_username, postgres_password, postgres_host, postgres_port, postgres_database)
    
    # Insert data into PostgreSQL
    db_utils.insert_data_to_postgresql(postgres_data, 'transaction_data', postgres_engine)
