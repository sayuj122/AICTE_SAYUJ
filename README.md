# Predictive Maintenance for Wind Turbines using Machine Learning

## Project Overview

This project focuses on building a predictive maintenance model for wind turbines. The primary goal is to leverage sensor data to predict potential generator failures before they occur. By accurately anticipating breakdowns, maintenance can be scheduled proactively, which significantly reduces operational costs, minimizes downtime, and prevents catastrophic equipment damage.

The core of this project is a machine learning classification model trained to distinguish between healthy and failing wind turbine generators based on a set of 40 encrypted sensor readings.
Goal

The main objective is to develop a robust classification model that can accurately predict generator failures. The business context dictates that the cost of a missed failure (a False Negative) is much higher than the cost of a false alarm (a False Positive). Therefore, the model optimization focuses on maximizing Recall, which measures the model's ability to correctly identify all actual failures.

Primary Metric: Maximize Recall for the 'failure' class (Target = 1).
Dataset

The project utilizes the "Renewind" dataset, which is a modified version of real-world sensor data from wind turbines.

    Source: Kaggle: Renewind Dataset

    Training Set: Train.csv contains 20,000 observations.

    Testing Set: Test.csv contains 5,000 observations.

    Features: The dataset includes 40 anonymized predictor variables (V1 to V40).

    Target Variable: A single binary target variable, 'Target', where:

        0: Represents a non-failure (healthy generator).

        1: Represents a failure that requires maintenance.

A key characteristic of this dataset is its significant class imbalance, with the 'failure' class being the minority. This was a critical consideration in the modeling process.
Methodology

The project follows a standard machine learning workflow:

    Data Preprocessing:

        Missing values in the dataset (columns V1 and V2) were handled by filling them with the mean of their respective columns.

    Exploratory Data Analysis (EDA):

        Initial analysis confirmed the severe class imbalance in the target variable.

        Histograms and boxplots were used to understand the distribution of each feature, revealing the presence of numerous outliers.

    Data Splitting & Validation Framework:

        The training data was split into a training set (80%) and a validation set (20%) to evaluate the model's performance on unseen data. Stratification was used to maintain the original class distribution in both sets.

    Handling Class Imbalance:

        To address the imbalanced dataset and improve the model's ability to detect the minority 'failure' class, RandomUnderSampler from the imblearn library was applied to the training set. This technique balances the classes by reducing the number of majority class samples.

    Feature Scaling:

        StandardScaler from scikit-learn was used to standardize the features, ensuring they have a mean of 0 and a standard deviation of 1. This is crucial for the performance of many machine learning algorithms.

    Model Building and Training:

        Several baseline models were evaluated, with XGBoost (XGBClassifier) showing the most promising results, especially in terms of recall.

        The final model is an XGBClassifier trained on the resampled and scaled training data. Hyperparameter tuning was performed to optimize for the best recall score.

Results & Performance

The final tuned XGBoost model, trained on the undersampled data, was evaluated on the hold-out validation set.

    Recall Score: The model achieved a high recall, successfully identifying a significant percentage of the actual generator failures in the validation data. This aligns with the primary project goal of minimizing missed failures.

    Confusion Matrix: The validation confusion matrix showed that while there were some false positives, the number of critical false negatives was kept low. For instance, in the notebook's final run, the model correctly predicted 77.60% of True Negatives and 5.06% of True Positives, while misclassifying only 0.58% as False Negatives.


How to Run the Code

This project is contained in a single Python script (predictive_maintenance_model.py) which encapsulates the entire workflow.
Prerequisites

Ensure you have Python installed, along with the following libraries:

pip install pandas numpy seaborn matplotlib scikit-learn imblearn xgboost

Steps to Execute

    Download the Data:

        Download the Train.csv file from the Kaggle dataset page.
        https://www.kaggle.com/datasets/mariyamalshatta/renewind/data
        Place it in a directory structure where the script can access it, for example: /kaggle/input/renewind/Train.csv. You may need to create these folders or modify the file path in the script.


    Output:

        The script will print the progress of each step (data splitting, resampling, training).

        Finally, it will display the model's validation results, including the classification report and a confusion matrix plot.