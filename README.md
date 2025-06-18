# Data Science Internship Projects - AICTE Oasis Infobyte SIP

This repository contains solutions to several data science tasks, completed as part of the AICTE Oasis Infobyte SIP internship program. 
Each task involves a different aspect of data analysis and machine learning, ranging from classification and regression to text processing.

## Table of Contents

1.  [Task 1: Iris Flower Classification](#task-1-iris-flower-classification)
2.  [Task 2: Unemployment Analysis with Python](#task-2-unemployment-analysis-with-python)
3.  [Task 3: Car Price Prediction with Machine Learning](#task-3-car-price-prediction-with-machine-learning)
4.  [Task 4: Email Spam Detection with Machine Learning](#task-4-email-spam-detection-with-machine-learning)
5.  [Task 5: Sales Prediction Using Python](#task-5-sales-prediction-using-python)

---

## Task 1: Iris Flower Classification

*Description:* This project focuses on classifying Iris flower species (setosa, versicolor, and virginica) based on their measurements using a machine learning model. 
It's a classic multi-class classification problem.

*Key Concepts Applied:*
* Data Loading (pd.read_csv)
* Exploratory Data Analysis (EDA)
* Data Splitting (train_test_split)
* Decision Tree Classifier
* Model Evaluation (Accuracy Score, Classification Report, Confusion Matrix)
* Cross-Validation (KFold, cross_val_score)

*Files:*
* IrisFlower.ipynb (Jupyter Notebook for the solution)

---

## Task 2: Unemployment Analysis with Python

*Description:* This task involves analyzing unemployment data to understand trends and patterns, identifying unemployment rate as a key economic indicator.
The project highlights the analysis of a sharp increase in unemployment rate during critical periods (e.g., COVID-19 impact).

*Key Concepts Applied:*
* Data Loading (pd.read_csv)
* Data Cleaning (handling missing values, robust column renaming, stripping whitespace)
* Data Preprocessing (datetime conversion for time series analysis)
* Exploratory Data Analysis (EDA)
* Visualization (matplotlib.pyplot, seaborn, plotly.express for interactive plots)
* Trend Analysis (e.g., lineplot for overall unemployment over time)
* Regional Analysis (e.g., average unemployment by region using barplot)

*Files:*
* Unemployment_Analysis.ipynb (Jupyter Notebook for the solution)
* Unemployment_in_India.csv (Dataset file - please ensure this file is present in the repository or provide download instructions)

---

## Task 3: Car Price Prediction with Machine Learning

*Description:* This project aims to predict the price of a car based on various factors like brand goodwill, car features, horsepower, and mileage. Car price prediction is a significant regression problem in the automotive domain.

*Key Concepts Applied:*
* Data Loading (pd.read_csv)
* Data Cleaning (robust column renaming, handling missing values, dropping irrelevant columns)
* Feature Engineering (e.g., extracting car brand from carname)
* Exploratory Data Analysis (EDA) including correlation matrices and scatter plots
* Regression Modeling (LinearRegression, RandomForestRegressor)
* Preprocessing Pipelines (StandardScaler, OneHotEncoder, ColumnTransformer, Pipeline)
* Model Evaluation (Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared ($R^2$) Score)
* Cross-Validation for model robustness

*Files:*
* Car_Price_Prediction.ipynb (Jupyter Notebook for the solution)
* CarPrice_Assignment.csv (Dataset file - please ensure this file is present in the repository or provide download instructions)

---

## Task 4: Email Spam Detection with Machine Learning

*Description:* This project focuses on building an email spam detector, categorizing incoming emails as either 'spam' or 'non-spam' (ham). It addresses the common issue of unwanted and potentially malicious emails.

*Key Concepts Applied:*
* Text Data Loading and Initial Cleaning (handling specific CSV formats, column renaming)
* Data Preprocessing for Text (TF-IDF Vectorization using TfidfVectorizer)
* Binary Classification Problem
* Machine Learning Pipeline (Pipeline for chaining text processing and model)
* Naive Bayes Classifier (MultinomialNB), a common and effective model for text classification
* Model Evaluation (Accuracy, Classification Report, Confusion Matrix)
* Cross-Validation for robust performance assessment

*Files:*
* Spam_Detection.ipynb (Jupyter Notebook for the solution)
* spam.csv (Dataset file - please ensure this file is present in the repository or provide download instructions)

---

## Task 5: Sales Prediction Using Python

*Description:* This project involves predicting future sales volumes based on factors such as advertising spend across different media channels (e.g., TV, Radio, Newspaper). It highlights the practical application of data science in optimizing marketing strategies and resource allocation.

*Key Concepts Applied:*
* Data Loading (pd.read_csv)
* Data Cleaning (column standardization, handling missing values)
* Exploratory Data Analysis (EDA) on sales and advertising channels, including correlation analysis
* Regression Modeling (LinearRegression, RandomForestRegressor)
* Preprocessing Pipelines (StandardScaler, ColumnTransformer, Pipeline)
* Model Evaluation (Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared ($R^2$) Score)
* Cross-Validation for robust model evaluation

*Files:*
* Sales_Prediction.ipynb (Jupyter Notebook for the solution)
* advertising.csv (Dataset file - please ensure this file is present in the repository or provide download instructions; exact name may vary)

---
