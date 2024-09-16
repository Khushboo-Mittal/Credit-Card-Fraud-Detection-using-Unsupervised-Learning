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
     
    # Description: This code snippet ingests customer data from a CSV file, preprocesses it, and stores it in
    # PostgreSQL and Cassandra databases.
        # PostgreSQL: Yes 
        # Cassandra: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Pandas 2.2.2

import pandas as pd # Importing pandas for data manipulation
import db_utils # Importing utility functions for database operations

def ingest_data(data_path, postgres_username, postgres_password, postgres_host, postgres_port, postgres_database, cassandra_host, cassandra_port, cassandra_keyspace):
    
    data = pd.read_csv(data_path) # Read data from CSV file

    # Separate data for PostgreSQL and Cassandra (will have to change this)
    postgres_data = data[['customer_id', 'subscription_type', 'annual_fee', 'payment_method', 'account_age', 'number_of_logins',
                          'total_spent', 'num_tickets_raised', 'avg_response_time', 'satisfaction_score', 'country', 'device', 'churn']]
    
    
    # Connect to PostgreSQL
    postgres_engine = db_utils.connect_postgresql(postgres_username, postgres_password, postgres_host, postgres_port, postgres_database)
    
    # Insert data into PostgreSQL
    db_utils.insert_data_to_postgresql(postgres_data, 'customer_data', postgres_engine)