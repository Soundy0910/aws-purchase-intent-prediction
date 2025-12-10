# Scalable Real-Time Purchase Intent Prediction on AWS

![AWS](https://img.shields.io/badge/AWS-SageMaker%20|%20Glue%20|%20Athena-orange)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project implements an end-to-end Machine Learning pipeline on AWS to predict user purchase intent in real-time. 

Processing a massive e-commerce dataset (**400 million+ raw events**), the system aggregates behavior into **90 million user sessions** and predicts whether a user will make a purchase. The architecture is designed to handle **out-of-core datasets** that exceed instance memory (OOM), utilizing distributed processing and iterative streaming.

## 🏗️ Architecture
**Data Flow:** `S3 Raw Data` -> `AWS Glue/Athena (Aggregation)` -> `SageMaker Processing (Feature Eng)` -> `SageMaker Training (XGBoost)` -> `SageMaker Endpoint (REST API)`

1.  **Ingestion:** Raw clickstream data stored in S3.
2.  **ETL:** AWS Athena used to aggregate 400M events into 90M session-level records (Parquet).
3.  **Preprocessing:** Custom **ScriptProcessor** job on SageMaker (`ml.m5.2xlarge`) handles feature engineering and time-based splitting.
4.  **Training:** Distributed XGBoost training via **Script Mode**.
5.  **Deployment:** Real-time inference endpoint for sub-second predictions.

## 🚀 Key Technical Challenges & Solutions

### 1. Handling Out-of-Core Data (The "OOM" Problem)
* **Challenge:** Loading 90M sessions into Pandas on a 32GB instance caused immediate Memory Exhaustion (Exit Code 137).
* **Solution:** Engineered an **Iterative Streaming Pipeline**. Instead of bulk loading, the processing script reads Parquet files in chunks, processes features, and writes to disk immediately.
* **Result:** Zero memory crashes; pipeline scales linearly with data size.

### 2. Eliminating Data Leakage
* **Challenge:** Initial models achieved 99% AUC due to "Look-Ahead Bias" (using features like `n_events` or `session_duration` that implicitly contain the label info).
* **Solution:** Performed rigorous feature selection to remove mathematical leaks and focus purely on behavioral signals (e.g., `cart_to_view_ratio`, `browsing_patterns`).
* **Result:** Realistic, robust AUC of **96.5%**.

### 3. Time-Series Splitting in a Distributed System
* **Challenge:** Random `train_test_split` causes leakage in time-series user data.
* **Solution:** Implemented a **Two-Pass Algorithm**. 
    * *Pass 1:* Scans timestamps across all distributed files to calculate the global 80th percentile cutoff date.
    * *Pass 2:* Streams data and splits based on this global cutoff to ensure strict temporal validity.

## 🛠️ Tech Stack
* **Cloud:** AWS (S3, SageMaker, Glue, Athena)
* **Languages:** Python 3, SQL
* **Libraries:** Pandas, NumPy, XGBoost, Scikit-Learn, Boto3, S3FS
* **DevOps:** SageMaker Studio, Git

## 📊 Results
* **Model Performance:** 96.5% AUC on Test Data.
* **Inference Latency:** <100ms per request on `ml.m5.xlarge`.
* **Scale:** Successfully processed 90M+ records without using Spark, purely via optimized Python scripts.

## 💻 Usage
To run this pipeline:
1.  **Processing:** Run the launcher to trigger the SageMaker Processing Job.
2.  **Training:** Execute the XGBoost Estimator with the processed S3 paths.
3.  **Inference:** Deploy the endpoint and send a JSON payload:
    ```python
    # Example Payload (Views, Cart, UniqueProd, Hour, etc.)
    payload = "5,0,4,2,14,2,0,0.0" 
    response = predictor.predict(payload)
    ```