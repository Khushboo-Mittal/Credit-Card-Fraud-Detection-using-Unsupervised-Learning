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
     
    # Description: This code snippet creates a web app using Streamlit to train, evaluate, and predict churn using
    # three different ensemble models: Bagging, Voting Classifier, and Stacking Classifier.

# CODE - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # Dependency: 
        # Environment:     
            # Python 3.11.5
            # Streamlit 1.36.0

import streamlit as st  # Used for creating the web app
import datetime  # Used for setting default value in streamlit form

# Importing the .py helper files
from ingest import ingest_data  # Importing the ingest_data function from ingest.py
from preprocess import load_and_preprocess_data  # Importing the load_and_preprocess_data function from preprocess.py
from split import split_data  # Importing the split_data function from split.py
from isolationForest import train_model as train_model_isolationForest # Importing the train_model function from model_training.py
from model_eval import evaluate_model  # Importing the evaluate_model function from model_eval.py
from model_predict import predict_output  # Importing the predict_output function from model_predict.py


# Setting the page configuration for the web app
st.set_page_config(page_title="Churn Prediction", page_icon=":chart_with_upwards_trend:", layout="centered")

# Adding a heading to the web app
st.markdown("<h1 style='text-align: center; color: white;'>Churn Prediction </h1>", unsafe_allow_html=True)
st.divider()

# Declaring session states(streamlit variables) for saving the path throught page reloads
# This is how we declare session state variables in streamlit.

# PostgreSQL 
if "postgres_username" not in st.session_state:
    st.session_state.postgres_username = "User"
    
if "postgres_password" not in st.session_state:
    st.session_state.postgres_password = "password"
    
if "postgres_host" not in st.session_state:
    st.session_state.postgres_host = "localhost"

if "postgres_port" not in st.session_state:
    st.session_state.postgres_port = "5432"
    
if "postgres_database" not in st.session_state:
    st.session_state.postgres_database = "churn_prediction"



# MongoDB
if "mongoDb_host" not in st.session_state:
    st.session_state.mongoDb_host = "localhost"
    
if "mongoDb_port" not in st.session_state:
    st.session_state.mongoDb_port = " 27017"
    
if "mongoDb_db" not in st.session_state:
    st.session_state.mongoDb_db = "1"

# Paths
if "master_data_path" not in st.session_state:
    st.session_state.master_data_path = "Data/Master/Mock_data.csv"
    
if "isolation_forest_path" not in st.session_state:
    st.session_state.bagging_model_path = "isolation_forest_model.pkl"
    
if "local_outlier_factor_path" not in st.session_state:
    st.session_state.voting_model_path = "loacl_outlier_factor_model.pkl"
    
if "oneclass_svm_path" not in st.session_state:
    st.session_state.stacking_model_path = "oneclass_svm_model.pkl"

# Creating tabs for the web app.
tab1, tab2, tab3, tab4 = st.tabs(["Model Config","Model Training","Model Evaluation", "Model Prediction"])

# Tab for Model Config
with tab1:
    st.subheader("Model Configuration")
    st.write("This is where you can configure the model.")
    st.divider()
    
    with st.form(key="Config Form"):
        tab_pg, tab_cs, tab_mongodb, tab_paths = st.tabs(["PostgreSQL", "MongoDB", "Paths"])
        
        # Tab for PostrgreSQL Configuration
        with tab_pg:
            st.markdown("<h2 style='text-align: center; color: white;'>PostgreSQL Configuration</h2>", unsafe_allow_html=True)
            st.write(" ")
            
            st.write("Enter PostgreSQL Configuration Details:")
            st.write(" ")
            
            st.session_state.postgres_username = st.text_input("Username", st.session_state.postgres_username)
            st.session_state.postgres_password = st.text_input("Password", st.session_state.postgres_password, type="password")
            st.session_state.postgres_host = st.text_input("Postgres Host", st.session_state.postgres_host)
            st.session_state.postgres_port = st.text_input("Port", st.session_state.postgres_port)
            st.session_state.postgres_database = st.text_input("Database", st.session_state.postgres_database)
        
        # Tab for MongoDB Configuration
        with tab_mongodb:
            st.markdown("<h2 style='text-align: center; color: white;'>Redis Configuration</h2>", unsafe_allow_html=True)
            st.write(" ")
            
            st.write("Enter MongoDB Configuration Details:")
            st.write(" ")
            
            st.session_state.mongoDb_host = st.text_input("MongoDB Host", st.session_state.redis_host)
            st.session_state.mongoDb_port = st.text_input("Port", st.session_state.redis_port)
            st.session_state.mongoDb_db = st.text_input("DB", st.session_state.redis_db)
        
        # Tab for Paths Configuration
        with tab_paths:
            st.markdown("<h2 style='text-align: center; color: white;'>Paths Configuration</h2>", unsafe_allow_html=True)
            st.write(" ")
            
            st.write("Enter Path Configuration Details:")
            st.write(" ")
            
            st.session_state.master_data_path = st.text_input("Master Data Path", st.session_state.master_data_path)
            st.session_state.isolation_forest_path = st.text_input("Isolation Forest Model Path", st.session_state.isolation_forest_path)
            st.session_state.local_outlier_factor_path = st.text_input("Local Outlier Factor Model Path", st.session_state.local_outlier_factor_path)
            st.session_state.oneclass_svm_path = st.text_input("One-Class SVM Model Path", st.session_state.oneclass_svm_path)
            
        if st.form_submit_button(label="Save Config", use_container_width=True):
            st.write("Configurations Saved Successfully! ✅")

# Tab for Model Training
with tab2:
    st.subheader("Model Training")
    st.write("This is where you can train the model.")
    st.divider()
    
    # Training the Models
    selected_model = st.selectbox("Select Model", ["IsolationForest", "LocalOutlier factor", "One-class SVM"])
    if st.button("Train Model", use_container_width=True):  # Adding a button to trigger model training
        with st.status("Training model..."):  # Displaying a status message while training the model
            
            st.write("Ingesting data...")  # Displaying a message for data ingestion
            ingest_data(st.session_state.master_data_path, st.session_state.postgres_username, st.session_state.postgres_password, st.session_state.postgres_host, st.session_state.postgres_port, st.session_state.postgres_database)  # Calling the ingest_data function
            st.write("Data Ingested Successfully! ✅")  # Displaying a success message
            
            st.write("Preprocessing data...")  # Displaying a message for data preprocessing
            data_postgres_processed= load_and_preprocess_data(st.session_state.postgres_username, st.session_state.postgres_password, st.session_state.postgres_host, st.session_state.postgres_port, st.session_state.postgres_database, st.session_state.cassandra_host, st.session_state.cassandra_port, st.session_state.cassandra_keyspace)  # Calling the load_and_preprocess_data function
            st.write("Data Preprocessed Successfully! ✅")  # Displaying a success message
            
            st.write("Splitting data into train, test, validation, and super validation sets...")  # Displaying a message for data splitting
            split_data(st.session_state.mongoDb_host, st.session_state.mongoDb_port, st.session_state.mongoDb_db, data_postgres_processed) # Calling the split_data function
            st.write("Data Split Successfully! ✅")  # Displaying a success message
            
            st.write("Training model...")  # Displaying a message for model training
            
            # Choosing the model to train based on the user's selection
            if selected_model == "IsolationForest":
                # Calling the train_model function and storing the training accuracy and best hyperparameters
                training_accuracy, best_params = train_model_isolationForest(st.session_state.mongoDb_host, st.session_state.mongoDb_port, st.session_state.mongoDb_db, st.session_state.isolation_forest_path)
            # elif selected_model == "LocalOutlier factor":
            #     training_accuracy, best_params = train_model_localOutlierFactor(st.session_state.mongoDb_host, st.session_state.mongoDb_port, st.session_state.mongoDb_db, st.session_state.local_outlier_factor_path)
            # elif selected_model == "One-class SVM":
            #     training_accuracy, best_params = train_model_oneClassSVM(st.session_state.mongoDb_host, st.session_state.mongoDb_port, st.session_state.mongoDb_db, st.session_state.oneclass_svm_path)
            st.write("Model Trained Successfully! ✅")  # Displaying a success message
        
        # Displaying the training accuracy
        st.success(f"{selected_model} Model Successfully trained with training accuracy: {training_accuracy:.5f}")
        st.write(f"Best Hyperparameters")
        st.text(best_params)

# Tab for Model Evaluation
with tab3:
    st.subheader("Model Evaluation")
    st.write("This is where you can see the current metrics of the trained models")
    st.divider()
    
    # Displaying the metrics for the Bagging Model
    st.markdown("<h3 style='text-align: center; color: white;'>Bagging Model</h3>", unsafe_allow_html=True)
    st.divider()
    
    # Get the model test, validation, and super validation metrics
    isolation_test_accuracy, isolation_test_roc_auc, isolation_test_confusion_matrix, isolation_test_classification_report, isolation_val_accuracy, isolation_val_roc_auc, isolation_val_confusion_matrix, isolation_val_classification_report, isolation_superval_accuracy, isolation_superval_roc_auc, isolation_superval_confusion_matrix, isolation_superval_classification_report = evaluate_model(st.session_state.mongoDb_host, st.session_state.mongoDb_port, st.session_state.mongoDb_db, st.session_state.isolation_forest_path)
    
    # Display model metrics in three columns
    isolation_col1, isolation_col2, isolation_col3 = st.columns(3)
    
    # Helper function to center text vertically at the top using markdown
    def markdown_top_center(text):
        return f'<div style="display: flex; justify-content: center; align-items: flex-start; height: 100%;">{text}</div>'

    # Displaying metrics for test, validation, and super validation sets
    with isolation_col1:
        st.markdown(markdown_top_center("Test Metrics:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(f"Accuracy: {isolation_test_accuracy:.5f}"), unsafe_allow_html=True)
        st.write(" ")
        st.markdown(markdown_top_center(f"ROC AUC: {isolation_test_roc_auc:.5f}"), unsafe_allow_html=True)
        st.write(" ")
        st.markdown(markdown_top_center("Confusion Matrix:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(isolation_test_confusion_matrix), unsafe_allow_html=True)

    with isolation_col2:
        st.markdown(markdown_top_center("Validation Metrics:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(f"Accuracy: {isolation_val_accuracy:.5f}"), unsafe_allow_html=True)
        st.write(" ")
        st.markdown(markdown_top_center(f"ROC AUC: {isolation_val_roc_auc:.5f}"), unsafe_allow_html=True)
        st.write(" ")
        st.markdown(markdown_top_center("Confusion Matrix:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(isolation_val_confusion_matrix), unsafe_allow_html=True)

    with isolation_col3:
        st.markdown(markdown_top_center("Super Validation Metrics:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(f"Accuracy: {isolation_superval_accuracy:.5f}"), unsafe_allow_html=True)
        st.write(" ")
        st.markdown(markdown_top_center(f"ROC AUC: {isolation_superval_roc_auc:.5f}"), unsafe_allow_html=True)
        st.write(" ")
        st.markdown(markdown_top_center("Confusion Matrix:"), unsafe_allow_html=True)
        st.markdown(markdown_top_center(isolation_superval_confusion_matrix), unsafe_allow_html=True)
        
    st.divider()
    
    # Display classification reports for test, validation, and super validation sets
    st.markdown("<h3 style='text-align: center; color: white;'>Classification Reports</h3>", unsafe_allow_html=True)
    st.write(" ")
    
    st.text("Test Classification Report:")
    st.text(isolation_test_classification_report)
    
    st.divider()

    st.text("Validation Classification Report:")
    st.text(isolation_val_classification_report)
    
    st.divider()

    st.text("Super Validation Classification Report:")
    st.text(isolation_superval_classification_report)
    
    st.divider()
    
    # # Displaying the metrics for the Voting Classifier Model
    # st.markdown("<h3 style='text-align: center; color: white;'>Voting Classifier Model</h3>", unsafe_allow_html=True)
    # st.divider()
    
    # # Get the model test, validation, and super validation metrics
    # voting_test_accuracy, voting_test_roc_auc, voting_test_confusion_matrix, voting_test_classification_report, voting_val_accuracy, voting_val_roc_auc, voting_val_confusion_matrix, voting_val_classification_report, voting_superval_accuracy, voting_superval_roc_auc, voting_superval_confusion_matrix, voting_superval_classification_report = evaluate_model(st.session_state.redis_host, st.session_state.redis_port, st.session_state.redis_db, st.session_state.voting_model_path)

    # # Display model metrics in three columns
    # voting_col1, voting_col2, voting_col3 = st.columns(3)
    
    # # Displaying metrics for test, validation, and super validation sets
    # with voting_col1:
    #     st.markdown(markdown_top_center("Test Metrics:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(f"Accuracy: {voting_test_accuracy:.5f}"), unsafe_allow_html=True)
    #     st.write(" ")
        
    #     # Note that ROC AUC is not available for Voting Classifier
    #     if voting_test_roc_auc is not None:
    #         st.markdown(markdown_top_center(f"ROC AUC: {voting_test_roc_auc:.5f}"), unsafe_allow_html=True)
    #     else:
    #         st.markdown(markdown_top_center("ROC AUC: N/A"), unsafe_allow_html=True)
    #     st.write(" ")
    #     st.markdown(markdown_top_center("Confusion Matrix:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(voting_test_confusion_matrix), unsafe_allow_html=True)
        
    # with voting_col2:
    #     st.markdown(markdown_top_center("Validation Metrics:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(f"Accuracy: {voting_val_accuracy:.5f}"), unsafe_allow_html=True)
    #     st.write(" ")
    #     if voting_val_roc_auc is not None:
    #         st.markdown(markdown_top_center(f"ROC AUC: {voting_val_roc_auc:.5f}"), unsafe_allow_html=True)
    #     else:
    #         st.markdown(markdown_top_center("ROC AUC: N/A"), unsafe_allow_html=True)
    #     st.write(" ")
    #     st.markdown(markdown_top_center("Confusion Matrix:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(voting_val_confusion_matrix), unsafe_allow_html=True)
        
    # with voting_col3:
    #     st.markdown(markdown_top_center("Super Validation Metrics:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(f"Accuracy: {voting_superval_accuracy:.5f}"), unsafe_allow_html=True)
    #     st.write(" ")
    #     if voting_superval_roc_auc is not None:
    #         st.markdown(markdown_top_center(f"ROC AUC: {voting_superval_roc_auc:.5f}"), unsafe_allow_html=True)
    #     else:
    #         st.markdown(markdown_top_center("ROC AUC: N/A"), unsafe_allow_html=True)
    #     st.write(" ")
    #     st.markdown(markdown_top_center("Confusion Matrix:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(voting_superval_confusion_matrix), unsafe_allow_html=True)
        
    # st.divider()
    
    # # Display classification reports for test, validation, and super validation sets
    # st.markdown("<h3 style='text-align: center; color: white;'>Classification Reports</h3>", unsafe_allow_html=True)
    # st.write(" ")
    
    # st.text("Test Classification Report:")
    # st.text(voting_test_classification_report)
    
    # st.divider()
    
    # st.text("Validation Classification Report:")
    # st.text(voting_val_classification_report)
    
    # st.divider()
    
    # st.text("Super Validation Classification Report:")
    # st.text(voting_superval_classification_report)
    
    # st.divider()
    
    # # Displaying the metrics for the Stacking Classifier Model
    # st.markdown("<h3 style='text-align: center; color: white;'>Stacking Classifier Model</h3>", unsafe_allow_html=True)
    # st.divider()
    
    # # Get the model test, validation, and super validation metrics
    # stacking_test_accuracy, stacking_test_roc_auc, stacking_test_confusion_matrix, stacking_test_classification_report, stacking_val_accuracy, stacking_val_roc_auc, stacking_val_confusion_matrix, stacking_val_classification_report, stacking_superval_accuracy, stacking_superval_roc_auc, stacking_superval_confusion_matrix, stacking_superval_classification_report = evaluate_model(st.session_state.redis_host, st.session_state.redis_port, st.session_state.redis_db, st.session_state.stacking_model_path)
    
    # # Display model metrics in three columns
    # stacking_col1, stacking_col2, stacking_col3 = st.columns(3)
    
    # # Displaying metrics for test, validation, and super validation sets
    # with stacking_col1:
    #     st.markdown(markdown_top_center("Test Metrics:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(f"Accuracy: {stacking_test_accuracy:.5f}"), unsafe_allow_html=True)
    #     st.write(" ")
    #     st.markdown(markdown_top_center(f"ROC AUC: {stacking_test_roc_auc:.5f}"), unsafe_allow_html=True)
    #     st.write(" ")
    #     st.markdown(markdown_top_center("Confusion Matrix:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(stacking_test_confusion_matrix), unsafe_allow_html=True)
        
    # with stacking_col2:
    #     st.markdown(markdown_top_center("Validation Metrics:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(f"Accuracy: {stacking_val_accuracy:.5f}"), unsafe_allow_html=True)
    #     st.write(" ")
    #     st.markdown(markdown_top_center(f"ROC AUC: {stacking_val_roc_auc:.5f}"), unsafe_allow_html=True)
    #     st.write(" ")
    #     st.markdown(markdown_top_center("Confusion Matrix:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(stacking_val_confusion_matrix), unsafe_allow_html=True)
        
    # with stacking_col3:
    #     st.markdown(markdown_top_center("Super Validation Metrics:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(f"Accuracy: {stacking_superval_accuracy:.5f}"), unsafe_allow_html=True)
    #     st.write(" ")
    #     st.markdown(markdown_top_center(f"ROC AUC: {stacking_superval_roc_auc:.5f}"), unsafe_allow_html=True)
    #     st.write(" ")
    #     st.markdown(markdown_top_center("Confusion Matrix:"), unsafe_allow_html=True)
    #     st.markdown(markdown_top_center(stacking_superval_confusion_matrix), unsafe_allow_html=True)
        
    # st.divider()
    
    # # Display classification reports for test, validation, and super validation sets
    # st.markdown("<h3 style='text-align: center; color: white;'>Classification Reports</h3>", unsafe_allow_html=True)
    
    # st.text("Test Classification Report:")
    # st.text(stacking_test_classification_report)
    
    # st.divider()
    
    # st.text("Validation Classification Report:")
    # st.text(stacking_val_classification_report)
    
    # st.divider()
    
    # st.text("Super Validation Classification Report:")
    # st.text(stacking_superval_classification_report)
    
    # st.divider()
      
# Tab for Model Prediction
with tab4:
    
    st.subheader("Model Prediction")
    st.write("This is where you can predict whether the credit card is fraudulent or not.")
    st.divider()

    # Creating a form for user input
    with st.form(key="PredictionForm"): 
        
        selected_model = st.selectbox(label="Select Model",
                                      options=["IsolationForest", "LocalOutlier factor", "One-class SVM"])
        
        # Mapping model names to their respective paths
        model_path_mapping = {
            "IsolationForest": st.session_state.bagging_model_path,
            "LocalOutlier factor": st.session_state.local_outlier_factor__path,
            "One-class SVM": st.session_state.oneClass_model_path
        }
        
        subscription_type = st.selectbox(label="Subscription Type", 
                                         options=["Basic", "Gold", "Premium"])
        
        payment_method = st.selectbox(label="Payment Method",
                                      options= ["CreditCard", "DebitCard", "UPI"])
        
        country = st.selectbox(label="Country",
                               options=["USA", "China", "Japan", "Taiwan", "Germany", "UK", "France", "Canada", "Australia", "Brazil"])
        
        device = st.selectbox(label="Device",
                              options=["Mobile", "Desktop", "Tablet"])
        
        annual_fee = st.selectbox(label= "Annual Fee",
                                  options= [29.99, 59.99, 79.99])

        account_age = st.number_input(label="Account Age (in years)",
                                      min_value=1,
                                      max_value=36,
                                      value=10)
        
        number_of_logins = st.number_input(label="Number of Logins",
                                           min_value=1,
                                           max_value= 1000, 
                                           value=568)
        
        total_spent = st.number_input(label="Total Spent",
                                      min_value=0.0,
                                      max_value=5000.0,
                                      value=2625.2)
        
        num_tickets_raised = st.number_input(label= "Number of Tickets Raised",
                                             min_value=0,
                                             max_value=20,
                                             value= 5)
        
        avg_response_time = st.number_input(label="Average Response Time (in hours)",
                                            min_value=0,
                                            max_value=48,
                                            value=12)
        
        satisfaction_score = st.number_input(label="Satisfaction Score",
                                             min_value=1,
                                             max_value=10,
                                             value=7)
        
        usage_hours_per_month = st.number_input(label= "Usage Hours per Month",
                                                min_value=0,
                                                max_value=200,
                                                value=123)
        
        signup_date = st.date_input(label="Signup Date", 
                                    format="MM-DD-YYYY",
                                    value=datetime.date(2023, 6, 30))
        
        last_login = st.date_input(label="Last Login Date",
                                   format="MM-DD-YYYY",
                                   value=datetime.date(2024, 2, 15))
        # datetime.date is needed to set a date in streamlit date_input
        
        # The form always needs a submit button to trigger the form submission
        if st.form_submit_button(label="Predict", use_container_width=True):
            user_input = [signup_date, last_login, annual_fee, subscription_type, payment_method,
                          account_age, number_of_logins, total_spent, num_tickets_raised, avg_response_time,
                          satisfaction_score, country, device, usage_hours_per_month, model_path_mapping[selected_model]]
            
            st.write(predict_output(*user_input))  # Calling the predict_output function with user input and displaying the output