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

def preprocess_input_data(signup_date, last_login, annual_fee, subscription_type, payment_method,
                          account_age, number_of_logins, total_spent, num_tickets_raised, avg_response_time,
                          satisfaction_score, country, device, usage_hours_per_month):
    # Prepare input data as a DataFrame
    data = pd.DataFrame({
        'signup_date': [signup_date],  # Signup date
        'last_login': [last_login],  # Last login date
        'annual_fee': [annual_fee],  # Annual fee
        'subscription_type': [subscription_type],  # Subscription type
        'payment_method': [payment_method],  # Payment method
        'account_age': [account_age],  # Account age
        'number_of_logins': [number_of_logins],  # Number of logins
        'total_spent': [total_spent],  # Total amount spent
        'num_tickets_raised': [num_tickets_raised],  # Number of tickets raised
        'avg_response_time': [avg_response_time],  # Average response time
        'satisfaction_score': [satisfaction_score],  # Satisfaction score
        'country': [country],  # Country
        'device': [device],  # Device type
        'usage_hours_per_month': [usage_hours_per_month]  # Usage hours per month
    })
    
    # Preprocess categorical and numerical columns
    numerical_cols = [
        'annual_fee', 'account_age', 'number_of_logins', 'total_spent',
        'num_tickets_raised', 'avg_response_time', 'satisfaction_score',
        'last_login_year', 'last_login_month', 'last_login_day',
        'signup_year', 'signup_month', 'signup_day', 'usage_hours_per_month'
    ]
    
    categorical_cols = [
        'subscription_type', 'payment_method', 'country', 'device'
    ]
    
    # Handle date columns
    data['signup_date'] = pd.to_datetime(data['signup_date'])  # Convert signup date to datetime
    data['last_login'] = pd.to_datetime(data['last_login'])  # Convert last login date to datetime
    
    # Extract features from date columns
    data['signup_year'] = data['signup_date'].dt.year  # Extract year from signup date
    data['signup_month'] = data['signup_date'].dt.month  # Extract month from signup date
    data['signup_day'] = data['signup_date'].dt.day  # Extract day from signup date
    data['last_login_year'] = data['last_login'].dt.year  # Extract year from last login date
    data['last_login_month'] = data['last_login'].dt.month  # Extract month from last login date
    data['last_login_day'] = data['last_login'].dt.day  # Extract day from last login date
    
    # Drop original date columns
    data = data.drop(columns=['signup_date', 'last_login'])
    
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

def predict_output(signup_date, last_login, annual_fee, subscription_type, payment_method,
                   account_age, number_of_logins, total_spent, num_tickets_raised, avg_response_time,
                   satisfaction_score, country, device, usage_hours_per_month, model_path):
    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Preprocess input data
    data = preprocess_input_data(signup_date, last_login, annual_fee, subscription_type, payment_method,
                                 account_age, number_of_logins, total_spent, num_tickets_raised, avg_response_time,
                                 satisfaction_score, country, device, usage_hours_per_month)
    
    # Ensure column order matches model's expectations
    X_columns = [
        'subscription_type', 'payment_method', 'country', 'device',
        'annual_fee', 'account_age', 'number_of_logins', 'total_spent',
        'num_tickets_raised', 'avg_response_time', 'satisfaction_score',
        'last_login_year', 'last_login_month', 'last_login_day',
        'signup_year', 'signup_month', 'signup_day', 'usage_hours_per_month'
    ]
    
    X = data[X_columns]  # Arrange columns in the correct order
    
    # Predict output
    try:
        prediction = model.predict(X)[0]  # Make a prediction (assume only one prediction is made)
        return f"Model Prediction: {prediction}"  # Return the prediction
    except Exception as e:
        print("Error during prediction:", e)  # Print any error that occurs
        return None