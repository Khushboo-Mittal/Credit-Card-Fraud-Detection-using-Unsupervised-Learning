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
            
    # Description: This code snippet preprocesses input data for a machine learning model by scaling numerical
    # columns, encoding categorical columns, and extracting date components for further analysis.
        # PostgreSQL: Yes
        # MongoDB: Yes

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:
            # Python 3.11.5     
            # Pandas 2.2.2
            # Scikit-learn 1.5.0

import pandas as pd                                             # For data manipulation
import pickle                                                   # For loading the model from a pickle file
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For preprocessing input data

def preprocess_input_data(transaction_date, transaction_amount, merchant_category, card_type, transaction_location,
                          cardholder_age, cardholder_gender, transaction_description, account_balance, calander_income):
    # Prepare input data as a DataFrame
    data = pd.DataFrame({
        'transaction_date': [transaction_date],  # transaction_date
        'merchant_category': [merchant_category],  # merchant_category
        'card_type': [card_type],  # card_type
        'transaction_location': [transaction_location],  # transaction_location
        'cardholder_gender': [cardholder_gender],  # cardholder_gender
        'transaction_amount': [transaction_amount],  # transaction_amount
        'cardholder_age': [cardholder_age],  # cardholder_age
        'account_balance': [account_balance],  # account_balance
        'calander_income': [calander_income],  # calander_income
        'transaction_description': [transaction_description],  # transaction_description
    })
    # Handle date columns
    data['transaction_date'] = pd.to_datetime(data['transaction_date'])  # Convert signup date to datetime
    
    # Extract features from date columns
    data['transaction_year'] = data['transaction_date'].dt.year  # Extract year from transaction date
    data['transaction_month'] = data['transaction_date'].dt.month  # Extract month from transaction date
    data['transaction_day'] = data['transaction_date'].dt.day  # Extract day from transaction date
    
    # Drop original date columns
    data = data.drop(columns=['transaction_date'])

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

    # Rejoin scaled numerical columns
    data = data.drop(columns=numerical_cols) # Drop original numerical columns
    data = pd.concat([data, temp_data], axis=1) # Concatenate scaled numerical columns back
    
    return data

def predict_output(transaction_date, transaction_amount, merchant_category, card_type, transaction_location,
                          cardholder_age, cardholder_gender, transaction_description, account_balance, calander_income, model_path):
    # Load the trained model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Preprocess input data
    data = preprocess_input_data(transaction_date, transaction_amount, merchant_category, card_type, transaction_location,
                          cardholder_age, cardholder_gender, transaction_description, account_balance, calander_income)
    
    
    # Predict output
    try:
        if 'local_outlier_factor_model.pkl' in model_path:
            prediction = model.fit_predict(data.values)
        else:
            prediction = model.predict(data.values)  # Make a prediction (assume only one prediction is made)
        return f"Model Prediction: {prediction}"  # Return the prediction
    except Exception as e:
        print("Error during prediction:", e)  # Print any error that occurs
        return None
