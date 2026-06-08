# Heart Disease Prediction Notebook

## Overview
This notebook demonstrates a machine learning pipeline for predicting heart disease. It covers data loading, preprocessing, model training (Logistic Regression and Random Forest), evaluation, and comparison. The goal is to identify the most effective model for this classification task.

## Dataset
The dataset used for this analysis is the 'Heart Disease Dataset' from Kaggle Hub. It contains various features related to heart health, with 'target' being the variable to predict (0 for no heart disease, 1 for heart disease).

## Workflow
1.  **Data Loading:** The dataset is downloaded from Kaggle Hub and loaded into a pandas DataFrame.
2.  **Data Preprocessing:**
    *   Features (X) and target (y) are separated.
    *   The data is split into training and testing sets (80% train, 20% test).
    *   `StandardScaler` is used to scale the features, which is crucial for models like Logistic Regression and can benefit Random Forest.
3.  **Model Training & Evaluation:**
    *   **Logistic Regression:** A Logistic Regression model is initialized, trained on the scaled training data, and evaluated using accuracy, classification report, and a confusion matrix.
    *   **Random Forest Classifier:** A Random Forest Classifier is initialized, trained on the scaled training data, and evaluated similarly.
4.  **Model Comparison:** The accuracy of both models is compared visually using a bar plot.
5.  **Model Saving:** The best-performing model (Random Forest Classifier) is saved using `pickle` for future use.

## Results
*   **Logistic Regression Accuracy:** Approximately **79.51%**
*   **Random Forest Classifier Accuracy:** Approximately **98.54%**

The Random Forest Classifier significantly outperformed the Logistic Regression model on this dataset, indicating its suitability for this heart disease prediction task.

## Saved Model
The trained Random Forest Classifier model has been saved as `random_forest_model.pkl`.
