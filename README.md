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
# bank client data:
1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
# social and economic context attributes
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

Output variable (desired target):
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

Exploratory Data Analysis (EDA) was performed on the data.  

## Data Preparation
1. Data Cleaning
  - All features other than the selected ones were dropped from the raw data.
  - Rows were filtered based on the numeric data assumptions (on 'price', 'year', 'odometer') listed above to remove outliers. Applying the filter removes NULL values for the selected numeric features.
  - For the remaining rows, NULL values for the following selected categorical features were filled based on the below:
    - 'fuel' - fill NULL with mode value i.e 'gas', which is reasonable as this is the case for majority of the cars
    - 'condition' - fill NULL with mode value i.e 'good', which is reasonable for a resale car
    - 'transmission' - fill NULL with mode value i.e 'automatic', which is reasonable as this is the case for majority of the cars
    - 'drive' - fill NULL with specific value 'fwd', which is reasonable as majority of the cars use front-wheel drive
    - 'cylinders' - fill NULL with specific value '4 cylinders', which is reasonable as majority of the cars run on 4 cylinders. Further, consolidate 3/5/10/12 cylinder counts to 'other' as these have relatively low counts.  
2. Data Transformation
  - The year that the car was manufactured was replaced with the age of the car (using current year as 2025).

**Cleaned Data**
Statistics of the car prices in the clean dataset is shown below. This would be helpful for comparison with the Root Mean Squared Errors (RMSE) results later.  
<img width="643" height="352" alt="image" src="https://github.com/user-attachments/assets/bf63d9ad-6451-4767-907c-85b2aa35e7ce" />

## Modelling
The clean data for modelling was shuffled before splitting into 80%/20% for train/test data. This prevents any ordering of the data from impacting the models.  

The following Column Transformations were applied:
- Numeric features: Polynomial Feature
- Categorical features: One Hot Encoding

**Model 1: Linear Regression**
- In this simple model, the polynomial degree was varied from 1 to 3 to assess if adding the polynomial features improved the model.
- Results: Polynomial degree 1 yielded the best results.
  - Polynomial Degree: [1, 2, 3]
  - Train RMSEs: ['9922.3432108642', '11437.6360049943', '12585.9282391401']
  - Test RMSEs: ['9636.3801039762', '11173.0485325417', '12354.9826787391']
- Polynomial degree 1 would be applied in subsequent models for simplicity.

**Model 2: Ridge Regression**
- In this L2 regularisation model, polynomial degree was kept at 1, while alpha was varied as [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0] through a GridSearch. 5-fold cross validation was also used.
- Results: Alpha = 10 yielded the best result. 
  - Ridge (GridSearch) Train RMSE: 9922.3432149006
  - Ridge (GridSearch) Test RMSE: 9636.3796371393
  - Ridge (GridSearch) Best Alpha: 10.0
- Under Ridge Regression, the Train RMSE is marginally higher than Linear Regression, but Test RMSE is marginally lower. Based on Test RMSE, it is a slightly better model than Linear Regression.
- Ridge coefficients are shown below:  
  - <img width="452" height="537" alt="image" src="https://github.com/user-attachments/assets/7fe56ed7-c8b2-43f8-a636-a688adcf3844" />
  - Top 3 features contributing to higher prices: 'fuel_diesel', 'cylinders_8 cylinders', 'drive_4wd'
  - Top 3 features contributing to lower prices: 'age', 'odometer', 'fuel_gas'


**Model 3: Lasso Regression**
- In this L1 regularisation model, the same hyperparameters were used as for Ridge Regression.
- Results: Alpha = 0.1 yielded the best result. 
  - Lasso (GridSearch) Train RMSE: 9922.3432197779
  - Lasso (GridSearch) Test RMSE: 9636.3805397717
  - Lasso (GridSearch) Best Alpha: 0.1
- Under Lasso Regression, both Train RMSE and Test RMSE are marginally higher than both earlier models, indicating it is a slightly poorer model.
  - Lasso coefficients are shown below:  
    - <img width="460" height="540" alt="image" src="https://github.com/user-attachments/assets/65f5abce-1d13-4d1c-b79a-a2e76025d2da" />
    - Top 3 features contributing to higher prices: 'fuel_diesel', 'drive_4wd', 'cylinders_8 cylinders'
    - Top 3 features contributing to lower prices: 'age', 'odometer', 'cylinders_4 cylinders'

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



