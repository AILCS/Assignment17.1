# Comparing Classifiers on Business Problem: What drives the effectiveness of a bank's directed marketing campaign?

**Assignment Notebook:** https://github.com/AILCS/Assignment11.1/blob/main/prompt_II%20(final).ipynb TODO

## Overview
In this application, we explore a dataset from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing). The classification goal is to predict if the client will subscribe a term deposit.
  
## Business Objective
The purpose is to leverage data-driven insights to increase the effectiveness of the bank’s directed marketing campaigns. By predicting which clients are most likely to accept a term deposit offer, the bank can enhance outreach efforts, improve customer engagement, and achieve higher conversion and profitability rates.

## Data Understanding
The data is from a Portuguese banking institution and is a collection of the results of multiple marketing campaigns based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. 

There are two datasets: 
1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).

The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

Input variables:  
Bank Client Data:  
1 - age (numeric)  
2 - job : type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')  
3 - marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown'; note: 'divorced' means divorced or widowed)  
4 - education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')  
5 - default: has credit in default? (categorical: 'no', 'yes', 'unknown')  
6 - housing: has housing loan? (categorical: 'no', 'yes', 'unknown')  
7 - loan: has personal loan? (categorical: 'no', 'yes', 'unknown')  
Related with the last contact of the current campaign:  
8 - contact: contact communication type (categorical: 'cellular', 'telephone')  
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')  
10 - day_of_week: last contact day of the week (categorical: 'mon', 'tue', 'wed', 'thu', 'fri')  
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.  
Other attributes:  
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)  
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)  
14 - previous: number of contacts performed before this campaign and for this client (numeric)  
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure', 'nonexistent', 'success')  
Social and economic context attributes:  
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)  
17 - cons.price.idx: consumer price index - monthly indicator (numeric)  
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)  
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)  
20 - nr.employed: number of employees - quarterly indicator (numeric)  
  
Output variable (target):  
21 - y - has the client subscribed a term deposit? (binary: 'yes', 'no')  
  
**Exploratory Data Analysis (EDA)**  
The dataset was clean, with no NULL values nor obvious outliers.  
<img width="361" height="497" alt="image" src="https://github.com/user-attachments/assets/f34706c0-a67c-4122-b867-03c53d9f79b0" />    
  
Numerical Columns: 
<img width="1118" height="293" alt="image" src="https://github.com/user-attachments/assets/bf15c163-6212-4f57-b346-39693eca76ac" />  
<img width="1132" height="1021" alt="image" src="https://github.com/user-attachments/assets/8f6eef9c-502c-424a-b03a-5c508e3248df" />  
  
Categorical Columns:  
<img width="281" height="607" alt="image" src="https://github.com/user-attachments/assets/b275e3c8-8c65-45be-a765-738f2dc63672" />  
<img width="283" height="687" alt="image" src="https://github.com/user-attachments/assets/bde57f59-d6c5-4564-a8e2-ccddaaf6324f" />  
<img width="280" height="372" alt="image" src="https://github.com/user-attachments/assets/5a642cd4-5127-4dfa-8e5f-6b0a54dcbb52" />  

Note: Dataset is imbalanced, with 89% 'no' and 11% 'yes' for the target variable.

## Data Preparation
1. Data Cleaning: Not required.  
  
2. Data Transformation
   - Numerical Features: StandardScaler()
   - Categorical Features: OneHotEncoder()
   - Target Variable: LabelEncoder()
  
3. Data Split: An 80/20 data split was applied to yield the train/test data. 

## Modelling
Modelling was done progressively in four stages.  

**Stage 1: Baseline Model - Dummy Classifier**
- The Dummy Classifier was used to establish a baseline.
  
**Stage 2: Simple Model - Logistic Regression**
- Logistic Regression with default hyperparameters was used as a simple/basic model.
  
**Stage 3: Model Comparison - Logistic Regression, KNN, Decision Tree, SVM**  
- Four models (including the simple model)  with default hyperparameters were trained on the dataset to assess how the models compare.
  1. Logistic Regression
  2. K Nearest Neighbors (KNN)
  3. Decision Tree
  4. Support Vector Machine (SVM)

**Stage 4: Model Comparison - Grid Search**  
- Four models (including the simple model)  with default hyperparameters were trained on the dataset to assess how the models compare.
  1. Logistic Regression
  2. K Nearest Neighbors (KNN)
  3. Decision Tree
  4. Support Vector Machine (SVM)
 
     
## Evaluation
**Features:**  
Selected features from the raw data included 'year' (or 'age'), 'odometer', 'fuel', 'condition', 'transmission', 'drive', and 'cylinders'. Together with One-Hot Encoding of categorical features, there are a total of 23 features.
  
**Models:**  
Three models were run on the cleaned dataset:
1. Linear Regression with polynomial features (degree: 1 to 3)
2. L2 regularisation using Ridge Regression with polynomial degree 1, alpha varied (0.001, 0.1, 1.0, 10.0, 100.0, 1000.0), and 5 fold cross validation.
3. L1 regularisation using Lasso Regression with polynomial degree 1, alpha varied (0.001, 0.1, 1.0, 10.0, 100.0, 1000.0), and 5 fold cross validation.

**Results:**  
Linear Regression (1 degree polynomial):
- Train RMSE: 9922.3432108642
- Test RMSE: 9636.3801039762

Ridge Regression (1 degree polynomial, alpha = 10):
- Train RMSE: 9922.3432149006
- Test RMSE: 9636.3796371393

Lasso Regression (1 degree polynomial, alpha = 0.1):
- Train RMSE: 9922.3432197779
- Test RMSE: 9636.3805397717
  
With the lowest Test RMSE, Ridge Regession performed the best (marginally) while Lasso Regression has the highest Test RMSE and performed the worst (marginally). Therefore Ridge Regression is the selected model.
  
Based on the selected Ridge Regression model:
- Top 3 features contributing to higher prices: 'fuel_diesel', 'cylinders_8 cylinders', 'drive_4wd'
- Top 3 features contributing to lower prices: 'age', 'odometer', 'fuel_gas'

The similarity of the Train and Test RMSE across all models also showed that the models were not overfit. 

Additionally, the RMSEs were within 1 standard deviation (15000) of the car prices of the training data. This is reasonable for a start and the model can be tweaked further to improve the prediction.


## Deployment
Dear Client,
  
I have identified a car price prediction model that estimates car prices using data you provided, specifically, 'price', 'year', 'odometer', 'fuel', 'condition', 'transmission', 'drive', and 'cylinders'.
  
Please note that the model is only good for estimating car prices ranging between 1k and 300k, cars from year 1990, and cars with less than 500k clocked on the odometer.
  
The prediction error based on your sample data is 9636, which is within a standard deviation (15000) of the car prices.
  
Additionally, below are the key findings:
- Top 3 features contributing to higher prices: diesel cars, 4-wheel drives, cars with 8 cylinders
- Top 3 features contributing to lower prices: older cars, cars used for longer distances, cars with 4 cylinders.
  
We can test the prediction model and continue to assess and refine it as we gather more data.
  
Please let me know if you have questions.
  
Regards,  
Chee Siong



