## DIY - Unsupervised Learning
This is the Unsupervised Learning Algorithms branch.

## Unsupervised Learning Algorithms
### Isolation Forest
Isolation Forest is an anomaly detection algorithm that focuses on identifying outliers in high-dimensional datasets. It constructs an ensemble of decision trees by randomly selecting features and splitting the data, which allows the model to isolate anomalies quickly. Anomalies are detected based on their average path length in the trees, as they require fewer splits to be isolated compared to normal observations. The output labels anomalies as -1 and normal observations as 1, making it effective for tasks where identifying rare events is crucial.

### Local Outlier Factor (LOF)
Local Outlier Factor (LOF) is an outlier detection method that assesses the local density of data points relative to their neighbors. It measures how much a point's density deviates from that of its neighbors, identifying points that are in regions of lower density as outliers. By computing LOF scores, the model provides a numerical value indicating the degree to which each point is an outlier, with lower scores suggesting a higher likelihood of being an anomaly. The final classification outputs -1 for outliers and 1 for inliers, making LOF useful for detecting localized anomalies in datasets.

### One-Class SVM
One-Class SVM is an anomaly detection technique designed for situations where only normal data is available for training. It works by fitting a hyperplane that best separates the normal data from the origin in a high-dimensional space, effectively creating a boundary around the normal instances. Points falling outside this boundary are considered outliers. The model outputs a classification of 1 for normal points and -1 for anomalies, along with decision function scores that indicate how far each point is from the hyperplane. This makes One-Class SVM particularly useful for fraud detection and other applications where anomalies are rare.

## Problem Definition
The business operates in the financial sector, offering credit card services to its customers. The goal is to use the Unsupervised Learning Algorithm to detect fraudulent cards based on historical transaction data.

## Data Definition
Mock data for learning purposes with features: transaction_id, customer_id, transaction_date, transaction_amount, merchant_category, payment_method, transaction_location, account_age, num_transactions, fraud_label, transaction_time, transaction_type, device_used, and customer_country.
Note: The dataset consists of 1000 samples, which may lead to potential overfitting. This could adversely affect evaluation metrics such as the silhouette score and Davies-Bouldin index, as these metrics may not accurately reflect clustering quality in a small dataset. In real-life scenarios, larger and more diverse datasets would provide a more accurate representation of transaction behavior, leading to more reliable performance metrics for fraud detection models.

## Directory Structure
-	**Code/:** Contains all the scripts for data ingestion, transformation, loading, evaluation, model training, inference, manual prediction, and web application.
-	**Data/:** Contains the raw mock data.
### Data Splitting
-	**Training Samples:** 600
-	**Testing Samples:** 250
-	**Validation Samples:** 75
-	**Supervalidation Samples:** 75

## Program Flow
1.	**db_utils:** This code snippet contains utility functions to connect to PostgreSQL database, create tables, and insert data into them.[`db_utils.py`]
2.	**Data Ingestion:** This code snippet ingests transaction data from a CSV file, preprocesses it, and stores it in PostgreSQL database. [`ingest.py`]
3.	**Data Preprocessing:** This code snippet preprocesses input data for a machine learning model by scaling numerical columns, encoding categorical columns, and extracting date components for further analysis [`preprocess.py`]
4.	**Data Splitting:** This code snippet contains functions to split preprocessed data into test, validation, and super validation and store it in a MongoDB database. [`split.py`]
5.	**Model Training:** This is where IsolationForest, LocalOutlierFactor, and OneClassSVM models, using the training data, are trained and stored in a MongoDB database. [`isolationForestpy`, `localOutlierFactor.py`, `oneClassSvm.py`]
6.	**Model Evaluation:** This code snippet contains utility functions to evaluate a model using test, validation, and super validation data stored in a MongoDB database. [`model_eval.py`]
7.	**Model Prediction:** This code snippet predict credit card fraudulent status based on user input data.  [`model_predict.py`]
8.	**Web Application:** This code snippet creates a web app using Streamlit to train, evaluate, and predict fraudulent credit card using three different unsupervised learning models: IsolationForest, LocalOutlierFactor, and OneClassSVM. [`app.py`]

## Steps to Run
1.	Install the necessary packages: `pip install -r requirements.txt`
2.	Run the Streamlit web application: `streamlit run Code/app.py`
