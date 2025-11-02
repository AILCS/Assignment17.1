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
1. age (numeric)  
2. job : type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')  
3. marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown'; note: 'divorced' means divorced or widowed)  
4. education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')  
5. default: has credit in default? (categorical: 'no', 'yes', 'unknown')  
6. housing: has housing loan? (categorical: 'no', 'yes', 'unknown')  
7. loan: has personal loan? (categorical: 'no', 'yes', 'unknown')  
Related with the last contact of the current campaign:  
8. contact: contact communication type (categorical: 'cellular', 'telephone')  
9. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')  
10. day_of_week: last contact day of the week (categorical: 'mon', 'tue', 'wed', 'thu', 'fri')  
11. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.  
Other attributes:  
12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)  
13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)  
14. previous: number of contacts performed before this campaign and for this client (numeric)  
15. poutcome: outcome of the previous marketing campaign (categorical: 'failure', 'nonexistent', 'success')  
Social and economic context attributes:  
16. emp.var.rate: employment variation rate - quarterly indicator (numeric)  
17. cons.price.idx: consumer price index - monthly indicator (numeric)  
18. cons.conf.idx: consumer confidence index - monthly indicator (numeric)  
19. euribor3m: euribor 3 month rate - daily indicator (numeric)  
20. nr.employed: number of employees - quarterly indicator (numeric)  
  
Output variable (target):  
21. y - has the client subscribed a term deposit? (binary: 'yes', 'no')  
  
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
- Four models (including the simple model) with default hyperparameters were trained on the dataset to assess which model performs best.
  1. Logistic Regression
  2. K Nearest Neighbors (KNN)
  3. Decision Tree
  4. Support Vector Machine (SVM)

**Stage 4: Model Improvement - Grid Search**  
- The hyperparameters for each model were further varied in this stage to enhance the model accuracy.
  1. Logistic Regression - 'C': 0.1, 1, 10
  2. K Nearest Neighbors (KNN) - 'n_neighbors': 3, 5, 7
  3. Decision Tree - 'max_depth': 5, 10, 15
  4. Support Vector Machine (SVM) - 'C': 0.1, 1, 10
- Grid Search with 5-fold cross validation was used.
- Note: Due to to large data size, Grid Search could only be performed on a limited number of hyperparameters and for a few hyperparameter values. Otherwise the model would take too long to train on Colab.

## Evaluation

**Results:**  
Stage 1: Baseline Model:
- Test Accuracy: 0.886502  
  
Stage 2: Baseline Model:
- Test Accuracy: 0.911022  
  
Stage 3: Model Comparison (default):  
<img width="487" height="112" alt="image" src="https://github.com/user-attachments/assets/6d58ae60-cbc0-4955-ba59-3d5e305179a2" />  
<img width="658" height="542" alt="image" src="https://github.com/user-attachments/assets/4bf01ae6-edfd-4e29-9f8f-91cea62b4d0c" />  
  
**Accuracy:**  
- All four models outperformed the DummyClassifier baseline.
- SVM achieved the highest accuracy, with Logistic Regression showing comparable results.
- KNN ranked third, while the Decision Tree performed the weakest.
- The Decision Tree model showed signs of overfitting, achieving 100% training accuracy but only baseline-level test accuracy.
  
**Train Time:**  
- SVM required the longest training time, consistent with its higher computational complexity.
- The other three models (KNN, Logistic Regression, and Decision Tree) trained significantly faster and had similar training durations.
    
Stage 4: Model Comparison (Grid Search):  
<img width="501" height="120" alt="image" src="https://github.com/user-attachments/assets/b3dce51e-d5e5-4757-af70-5588a3c8e7e1" />  
<img width="656" height="541" alt="image" src="https://github.com/user-attachments/assets/9ab7085d-cbb8-496d-b831-e56db61cb406" />  

**Accuracy:** 
- Both Decision Tree and KNN models showed improved test accuracy after applying GridSearch.
- The Decision Tree no longer achieved 100% training accuracy, indicating a reduction in overfitting.
- Logistic Regression and SVM achieved similar accuracy to their default hyperparameter results, suggesting minimal impact from tuning.
- Decision Tree performed the best overall.
- SVM and Logistic Regression had comparable performance, while KNN performed the weakest. 

**Train Time:** 
- Average training time increased for all models due to the GridSearch process evaluating multiple parameter combinations, some of which took longer.
- SVM remained the slowest to train, consistent with results from the default settings.
- The longer training time for SVM is expected, given its computational complexity and the large dataset size.



## Deployment
Dear Sir/Madam,
  
I have identified a prediction model that can estimate, with high accuracy based on the bank's dataset, whether a client will accept the bank's term deposit offer.
  
The prediction accuracy based on the dataset is 91%, which is reasonably high. We can test the prediction model and continue to assess and refine it as we gather more data.

A successful model will be a useful tool to achieve higher conversion and profitability rates, and enhance the bank's directed marketing campaign.
  
Please let me know if you have questions.
  
Regards,  
Chee Siong



