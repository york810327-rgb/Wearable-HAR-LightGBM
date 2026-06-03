# Human Activity Recognition (HAR) using Wearable Sensor Data

![Accuracy](https://img.shields.io/badge/Accuracy-93.18%25-success) 
![Model](https://img.shields.io/badge/Model-LightGBM-blue)
![Algorithm](https://img.shields.io/badge/Algorithm-PCA-orange)

## 📖 Overview
This repository contains the codebase and research methodology for our project on **Human Activity Recognition (HAR)** based on wearable sensor data. The project was developed as a submission for the National Undergraduate Mathematical Modeling Contest. 

The primary objective is to accurately classify six daily physical activities based on high-dimensional data collected from built-in wearable sensors (accelerometers and gyroscopes). We implemented a comprehensive machine learning pipeline that includes data preprocessing, dimensionality reduction, feature importance analysis, multi-class classification, and robustness testing under noisy environments.

## 📊 Dataset
The experimental data was collected from 30 volunteers performing six standard daily activities:
* Walking
* Upstairs
* Downstairs
* Sitting
* Standing
* Laying

Time-domain and frequency-domain features were extracted using a 2.56-second sliding window, resulting in a high-dimensional feature vector.

## ⚙️ Methodology

Our analytical framework consists of three main modules:

### 1. Feature Engineering & Dimensionality Reduction
* **Standardization:** Applied `StandardScaler` to normalize the sensor data distributions.
* **PCA (Principal Component Analysis):** Utilized unsupervised PCA to compress the high-dimensional space, successfully retaining 95% of the data's variance using approximately 100 principal components.
* **Feature Selection:** Leveraged a LightGBM classifier to compute feature importance scores (Information Gain), extracting the **Top 20 most critical features** that contribute to activity recognition.

### 2. Multi-Class Classification Model
* Built a highly efficient classification model using the **LightGBM** framework.
* Configured hyper-parameters optimized for generalization (e.g., `learning_rate=0.05`, `max_depth=7`, `n_estimators=200`).
* The model efficiently handles the remaining structural multi-class sensor data and rapidly maps the features to the 6 activity labels.

### 3. Robustness Analysis
* Simulated real-world sensor inaccuracies and physical jitters by injecting Gaussian noise into the standardized test sets.
* Evaluated the model's prediction accuracy across varying noise intensity levels (standard deviation ranging from 0 to 2.0).

## 🚀 Results & Performance

* **High Accuracy:** The LightGBM model achieved an overall accuracy of **93.18%** on the independent test set.
* **F1-Scores:** Reached a Macro-F1 and Weighted-F1 score of **0.93**.
* **Activity Specifics:** The model perfectly identifies the "Laying" activity (F1-score = 1.00). Dynamic activities (Walking, Upstairs, Downstairs) maintain robust F1-scores above 0.93.
* **Robustness:** The model demonstrates excellent stability under low-noise conditions; prediction accuracy only drops marginally when the noise intensity is less than or equal to 0.5. 

## 📂 Project Structure

```text
├── prepare_the_data.py   # Data standardization and preprocessing script
├── Q1.py                 # PCA dimensionality reduction & Top 20 Feature Selection
├── Q2.py                 # LightGBM classification model training & evaluation
├── Q3.py                 # Gaussian noise injection & robustness analysis
├── X_train.txt           # Training set features
├── Y_train.txt           # Training set labels
├── X_test.txt            # Test set features
└── Y_test.txt            # Test set labels
