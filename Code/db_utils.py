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
     
    # Description: This code snippet contains utility functions to connect to PostgreSQL and Cassandra databases,
    # create tables, and insert data into them.
        # PostgreSQL: Yes
        # Cassandra: Yes
        # Redis: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:
            # Python 3.11.5   
            # SQLAlchemy 2.0.31
            # Cassandra-driver 3.29.1

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, Date

def connect_postgresql(username, password, host, port, database):
    # Function to connect to PostgreSQL database using the provided configuration
    
    engine = create_engine(f"postgresql://{username}:{password}@{host}:{port}/{database}")
    return engine  # Return SQLAlchemy engine for PostgreSQL connection


def create_postgresql_table(engine):
    # Function to create a PostgreSQL table for customer data
    metadata = MetaData()  # Metadata object to hold information about the table
    customer_data = Table('customer_data', metadata,
                          Column('customer_id', String, primary_key=True),  # Primary key column
                          Column('subscription_type', String),  # Subscription type column
                          Column('annual_fee', Float),  # Annual fee column
                          Column('payment_method', String),  # Payment method column
                          Column('account_age', Integer),  # Account age column
                          Column('number_of_logins', Integer),  # Number of logins column
                          Column('total_spent', Float),  # Total amount spent column
                          Column('num_tickets_raised', Integer),  # Number of tickets raised column
                          Column('avg_response_time', Float),  # Average response time column
                          Column('satisfaction_score', Integer),  # Satisfaction score column
                          Column('country', String),  # Country column
                          Column('device', String),  # Device type column
                          Column('churn', Boolean))  # Churn status column
    metadata.create_all(engine)  # Create the table in the database


def insert_data_to_postgresql(data, table_name, engine):
    # Function to insert data into a PostgreSQL table
    
    # Create the table if not exists
    create_postgresql_table(engine)
    
    data.to_sql(table_name, engine, if_exists='replace', index=False)  # Insert data into the specified table
