# Unsupervised Learning for Fraudulent Credit Card Detection

## Business Case
Credit card fraud continues to be a major issue in the financial sector, resulting in billions of dollars in losses annually. Traditional fraud detection systems often face challenges in identifying rare fraudulent transactions, as fraud cases are typically outliers in transaction data. With the surge in digital payments and the increasing volume of transactions, conventional methods struggle to keep pace. To address this, more advanced approaches are needed—ones that can detect anomalies in data without relying on labeled fraud cases. These solutions are particularly valuable in scenarios where only legitimate transaction data is available for training, enabling the detection of novel fraud patterns that have not been encountered before.

## Project Overview
This project demonstrates how to use unsupervised learning algorithms to detect fraudulent credit card transactions from historical transaction data. It leverages three key unsupervised learning techniques to identify anomalies that may indicate fraudulent behavior, without requiring labeled data. The project implements data ingestion, preprocessing, model training, evaluation, and prediction functionalities, using Python and various libraries for machine learning, database management, and web application development.

## Algorithms Used
The following unsupervised learning models are utilized for fraud detection:

1. Isolation Forest
- **Purpose:** Detect outliers in high-dimensional datasets.
- **How it works:** Constructs an ensemble of decision trees that isolate anomalies by randomly selecting features and splitting the data.
- **Output:** -1 for anomalies, 1 for normal transactions.

2. Local Outlier Factor (LOF)
- **Purpose:** Identify localized anomalies by measuring the density deviation from neighbors.
- **How it works:** Computes LOF scores to detect data points in lower-density regions, which are outliers.
- **Output:** -1 for outliers, 1 for inliers.

3. One-Class SVM
- **Purpose:** Identify anomalies in datasets with only normal instances available for training.
- **How it works:** Fits a hyperplane to separate normal data from the origin, classifying data points outside the boundary as anomalies.
- **Output:** -1 for anomalies, 1 for normal points.


## Task
The primary task is to develop an unsupervised learning model to detect fraudulent credit card transactions based on historical transaction data. The goal is to identify anomalies or outliers in the transaction data that may represent fraudulent activity, without requiring any labeled data on fraud cases. This will be achieved through the application of unsupervised learning algorithms that can identify unusual patterns and flag potential fraud.

## Key Concepts
- **Anomaly Detection**: Identifying rare or unusual data points that deviate significantly from the norm, which could indicate fraud.
- **Unsupervised Learning**: A machine learning approach that does not require labeled data, making it suitable for situations where only legitimate transactions are available.
- **Outliers**: Data points that differ significantly from the rest of the data, often representing fraud in financial transaction data.
- **Model Evaluation**: Assessing the performance of the model to ensure its ability to identify fraud effectively.

## Directory Structure
- **Code/**: Scripts for ingestion, preprocessing, training, evaluation, and prediction.
- **Data/**: Mock transaction data for training.

## Steps to Run
1. **Install dependencies**:
   `pip install -r requirements.txt`
2. **Run the streamlit app**:
    `streamlit run Code/app.py`

## Notes:
- Refer to the README.md files in each folder for more detailed instructions on how to set up and run the specific algorithm.
- Make sure to install the necessary dependencies listed in the requirements.txt file.
- Ensure that PostgreSQL and MongoDB are set up correctly before running the scripts.
- The goal of this project is to understand and implement unsupervised learning algorithms to detect fraudulent transactions and evaluate their performance.


