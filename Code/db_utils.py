# META DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     # Developer details: 
        # Name: Harshita Jangde and Prachi Tavse
        # Role: Architects
    # Version:
        # Version: V 1.0 (20 September 2024)
            # Developers: Harshita Jangde and Prachi Tavse
            # Unit test: Pass
            # Integration test: Pass
     
    # Description: This code snippet contains utility functions to connect to PostgreSQL database,
    # create tables, and insert data into them.
        # PostgreSQL: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:
            # Python 3.10.8
            # SQLAlchemy 2.0.10

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime

def connect_postgresql(username, password, host, port, database):
    # Function to connect to PostgreSQL database using the provided configuration
    
    engine = create_engine(f"postgresql://{username}:{password}@{host}:{port}/{database}")
    return engine  # Return SQLAlchemy engine for PostgreSQL connection


def create_postgresql_table(engine):
    # Function to create a PostgreSQL table for customer data
    metadata = MetaData()  # Metadata object to hold information about the table
    transaction_data = Table('transaction_data', metadata,
                          Column('transaction_id', Integer, primary_key=True),  # Primary key column
                          Column('transaction_date', DateTime), #Date of Transaction column
                          Column('transaction_amount', Float),  # Transaction column
                          Column('merchant_category', String),  # Merchant Category column
                          Column('card_type', String),  # Card Type column
                          Column('transaction_location', String),  # Transaction Location column
                          Column('cardholder_age', Integer),  # Cardholder Age column
                          Column('cardholder_gender', String),  # Cardholder Gender column
                          Column('transaction_description', String),  # Transaction Description column
                          Column('account_balance', Float),  # Account Balance column
                          Column('calander_income', Float))  # Cardholder Income column
    metadata.create_all(engine)  # Create the table in the database


def insert_data_to_postgresql(data, table_name, engine):
    # Function to insert data into a PostgreSQL table
    
    # Create the table if not exists
    create_postgresql_table(engine)
    
    data.to_sql(table_name, engine, if_exists='replace', index=False)  # Insert data into the specified table
