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
     
    # Description: This code snippet preprocesses input data for a machine learning model by scaling numerical
    # columns, encoding categorical columns, and extracting date components for further analysis.
        # PostgreSQL: Yes
        # Cassandra: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:
            # Python 3.11.5     
            # Pandas 2.2.2
            # Scikit-learn 1.5.0

import pandas as pd                                             # For data manipulation
import pickle                                                   # For loading the model from a pickle file
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For preprocessing input data
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_input_data(transaction_date, transaction_amount, merchant_category, card_type, transaction_location,
                          cardholder_age, cardholder_gender, transaction_description, account_balance, calander_income):
    # Prepare input data as a DataFrame
    data = pd.DataFrame({
        'transaction_date': [transaction_date],  # transaction_date
        'transaction_amount': [transaction_amount],  # transaction_amount
        'merchant_category': [merchant_category],  # merchant_category
        'card_type': [card_type],  # card_type
        'transaction_location': [transaction_location],  # transaction_location
        'cardholder_age': [cardholder_age],  # cardholder_age
        'cardholder_gender': [cardholder_gender],  # cardholder_gender
        'transaction_description': [transaction_description],  # transaction_description
        'account_balance': [account_balance],  # account_balance
        'calander_income': [calander_income],  # calander_income
    })
    
    # Preprocess categorical and numerical columns
    numerical_cols = [
        'transaction_amount', 'cardholder_age', 'account_balance', 'calander_income', 'transaction_year',
        'transaction_month', 'transaction_day'
    ]
    
    categorical_cols = [
        'merchant_category', 'card_type', 'transaction_location', 'cardholder_gender', 
    ]
    
    # Handle date columns
    data['transaction_date'] = pd.to_datetime(data['signup_date'])  # Convert signup date to datetime
    
    # Extract features from date columns
    data['transaction_year'] = data['transaction_date'].dt.year  # Extract year from transaction date
    data['transaction_month'] = data['transaction_date'].dt.month  # Extract month from transaction date
    data['transaction_day'] = data['transaction_date'].dt.day  # Extract day from transaction date
    
    # Drop original date columns
    data = data.drop(columns=['transaction_date'])

    # Convert text descriptions into numerical features
    # Initialize the TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=100)

    # Transform the transaction_description column
    description_features = tfidf.fit_transform(data['transaction_description'])
    
    # Convert to dataframe
    description_data = pd.DataFrame(description_features.toarray(), columns=tfidf.get_feature_names_out())

    # Rejoin the data and drop original column
    data = pd.concat([data, description_data], axis=1)
    data = data.drop('transaction_description', axis=1, inplace=True)

    #Scale description feature values
    scaler = StandardScaler()
    for col in description_features:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
                              
    # Ensure numerical columns are of correct type
    for col in numerical_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, coerce errors to NaN
    
    # Scale numerical columns
    scaler = StandardScaler()
    for col in numerical_cols:
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
    
    # Encode categorical columns
    encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = encoder.fit_transform(data[col])  # Encode categorical columns as integers
    
    return data

def predict_output(transaction_date, transaction_amount, merchant_category, card_type, transaction_location,
                          cardholder_age, cardholder_gender, transaction_description, account_balance, calander_income, model_path):
    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Preprocess input data
    data = preprocess_input_data(transaction_date, transaction_amount, merchant_category, card_type, transaction_location,
                          cardholder_age, cardholder_gender, transaction_description, account_balance, calander_income)
    
    # Ensure column order matches model's expectations
    # X_columns = [
    #     'subscription_type', 'payment_method', 'country', 'device',
    #     'annual_fee', 'account_age', 'number_of_logins', 'total_spent',
    #     'num_tickets_raised', 'avg_response_time', 'satisfaction_score',
    #     'last_login_year', 'last_login_month', 'last_login_day',
    #     'signup_year', 'signup_month', 'signup_day', 'usage_hours_per_month'
    # ]
    
    # X = data[X_columns]  # Arrange columns in the correct order
    
    # Predict output
    try:
        prediction = model.predict(X)[0]  # Make a prediction (assume only one prediction is made)
        return f"Model Prediction: {prediction}"  # Return the prediction
    except Exception as e:
        print("Error during prediction:", e)  # Print any error that occurs
        return None
