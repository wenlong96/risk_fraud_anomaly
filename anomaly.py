https://www.kaggle.com/code/gpreda/credit-card-fraud-detection-predictive-models/notebook

### Context

### It is important that credit card companies are able to recognize
### fraudulent credit card transactions so that customers are not charged for
### items that they did not purchase.


### Content

### The dataset contains transactions made by credit cards in September 2013 by
### European cardholders. This dataset presents transactions that occurred in
### two days, where we have 492 frauds out of 284,807 transactions.
### The dataset is highly unbalanced, the positive class (frauds) account
### for 0.172% of all transactions.
### It contains only numerical input variables which are the result of a PCA
### transformation. Unfortunately, due to confidentiality issues, we cannot
### provide the original features and more background information about the
### data. Features V1, V2, … V28 are the principal components obtained with
### PCA, the only features which have not been transformed with PCA are 'Time'
### and 'Amount'. Feature 'Time' contains the seconds elapsed between each
### transaction and the first transaction in the dataset. The feature 'Amount'
### is the transaction Amount, this feature can be used for example-dependant
### cost-sensitive learning. Feature 'Class' is the response variable and it
### takes value 1 in case of fraud and 0 otherwise.
### Given the class imbalance ratio, we recommend measuring the accuracy using
### the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix 
### accuracy is not meaningful for unbalanced classification.


### Introduction

### The datasets contains transactions made by credit cards in September 2013
### by european cardholders. This dataset presents transactions that occurred
### in two days, where we have 492 frauds out of 284,807 transactions.
### The dataset is highly unbalanced, the positive class (frauds) account for
### 0.172% of all transactions. It contains only numerical input variables
### which are the result of a PCA transformation. Due to confidentiality issues,
### there are no original features and more background information about the
### data. Features V1, V2, ... V28 are the principal components obtained with
### PCA; The only features which have not been transformed with PCA are Time
### and Amount. Feature Time contains the seconds elapsed between each
### transaction and the first transaction in the dataset. The feature Amount
### is the transaction Amount, this feature can be used for example-dependant
### cost-sensitive learning. Feature Class is the response variable and it
### takes value 1 in case of fraud and 0 otherwise.

import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier
import xgboost as xgb
import os
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import ADASYN

pd.set_option('display.max_columns', 100)


#CROSS-VALIDATION
NUMBER_KFOLDS = 5 #number of KFolds for cross-validation


MAX_ROUNDS = 1000 #lgb iterations
EARLY_STOP = 50 #lgb early stop 
OPT_ROUNDS = 1000  #To be adjusted based on best validation rounds
VERBOSE_EVAL = 50 #Print out metric result


data_df = pd.read_csv("C://Users//wlim129//Desktop//Work//Anomaly Detection//creditcard.csv")

print(data_df.shape)
# data has rows: 284807  columns: 31

data_df.head()

data_df.describe()

# Looking to the Time feature, we can confirm that the data contains 284,807
# transactions, during 2 consecutive days (or 172792 seconds).

# check missing data
total = data_df.isnull().sum().sort_values(ascending = False)
percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()

# data unbalance
temp = data_df["Class"].value_counts()
df = pd.DataFrame({'Class': temp.index,'values': temp.values})

trace = go.Bar(
    x = df['Class'],y = df['values'],
    name="Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)",
    marker=dict(color="Red"),
    text=df['values']
)
data = [trace]
layout = dict(title = 'Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1)',
          xaxis = dict(title = 'Class', showticklabels=True), 
          yaxis = dict(title = 'Number of transactions'),
          hovermode = 'closest',width=600
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='class')

# Only 492 (or 0.172%) of transaction are fraudulent. That means the data is highly
# unbalanced with respect with target variable Class.

# data exploration

class_0 = data_df.loc[data_df['Class'] == 0]["Time"]
class_1 = data_df.loc[data_df['Class'] == 1]["Time"]

hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
iplot(fig, filename='dist_only')


# Fraudulent transactions have a distribution more even than valid
# transactions - are equaly distributed in time, including the low real 
# transaction times, during night in Europe timezone.

# Let's look into more details to the time distribution of both classes
# transaction, as well as to aggregated values of transaction count and
# amount, per hour. We assume (based on observation of the time distribution
# of transactions) that the time unit is second.

# Fraudulent transactions have a distribution more even than valid
# transactions - are equaly distributed in time, including the low real
# transaction times, during night in Europe timezone.

# Let's look into more details to the time distribution of both classes
# transaction, as well as to aggregated values of transaction count and
# amount, per hour. We assume (based on observation of the time distribution
# of transactions) that the time unit is second.

data_df['Hour'] = data_df['Time'].apply(lambda x: np.floor(x / 3600))

tmp = data_df.groupby(['Hour', 'Class'])['Amount'].aggregate(['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()
df = pd.DataFrame(tmp)
df.columns = ['Hour', 'Class', 'Min', 'Max', 'Transactions', 'Sum', 'Mean', 'Median', 'Var']
df.head()

# total amount

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Sum", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Sum", data=df.loc[df.Class==1], color="red")
plt.suptitle("Total Amount")
plt.show();

# total transactions 
 
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Transactions", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Transactions", data=df.loc[df.Class==1], color="red")
plt.suptitle("Total Number of Transactions")
plt.show();

# average transactions

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Mean", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Mean", data=df.loc[df.Class==1], color="red")
plt.suptitle("Average Amount of Transactions")
plt.show();

# max amount of transactions

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Max", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Max", data=df.loc[df.Class==1], color="red")
plt.suptitle("Maximum Amount of Transactions")
plt.show();

# median amount of transactions

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Median", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Median", data=df.loc[df.Class==1], color="red")
plt.suptitle("Median Amount of Transactions")
plt.show();

# min amount of transactions
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Min", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Min", data=df.loc[df.Class==1], color="red")
plt.suptitle("Minimum Amount of Transactions")
plt.show();

# transactions amount
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=False)
plt.show();

tmp = data_df[['Amount','Class']].copy()
class_0 = tmp.loc[tmp['Class'] == 0]['Amount']
class_1 = tmp.loc[tmp['Class'] == 1]['Amount']
class_0.describe()

class_1.describe()

# using median as mean,
# The real transaction have a larger mean value, larger Q1,
# smaller Q3 and larger outliers; fraudulent transactions have a
# smaller Q1 and mean, larger Q3 and smaller outliers.

# Let's plot the fraudulent transactions (amount) against time.
# The time is shown is seconds from the start of the time period
# (totaly 48h, over 2 days).

fraud = data_df.loc[data_df['Class'] == 1]

trace = go.Scatter(
    x = fraud['Time'],y = fraud['Amount'],
    name="Amount",
     marker=dict(
                color='rgb(238,23,11)',
                line=dict(
                    color='red',
                    width=1),
                opacity=0.5,
            ),
    text= fraud['Amount'],
    mode = "markers"
)
data = [trace]
layout = dict(title = 'Amount of fraudulent transactions',
          xaxis = dict(title = 'Time [s]', showticklabels=True), 
          yaxis = dict(title = 'Amount'),
          hovermode='closest'
         )
fig = dict(data=data, layout=layout)
iplot(fig, filename='fraud-amount')

# features correlation

plt.figure(figsize = (14,14))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = data_df.corr()
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns,linewidths=.1,cmap="Reds")
plt.show()

# As expected, there is no notable correlation between features V1-V28.
# There are certain correlations between some of these features and Time
# (inverse correlation with V3) and Amount
# (direct correlation with V7 and V20, inverse correlation with V1 and V5).

# Let's plot the correlated and inverse correlated values on the same graph.

# Let's start with the direct correlated values: {V20;Amount} and {V7;Amount}.

s = sns.lmplot(x='V20', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='V7', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})
plt.show()

# We can confirm that the two couples of features are correlated
# (the regression lines for Class = 0 have a positive slope, whilst the
# regression line for Class = 1 have a smaller positive slope).
# generally, we do see that when 1 increase the other do as well.

# Let's plot now the inverse correlated values.

s = sns.lmplot(x='V2', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='V5', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})
plt.show()

# We can confirm that the two couples of features are inverse correlated
# (the regression lines for Class = 0 have a negative slope while the
# regression lines for Class = 1 have a very small negative slope).

# features density plot

var = data_df.columns.values

i = 0
t0 = data_df.loc[data_df['Class'] == 0]
t1 = data_df.loc[data_df['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplots(8,4,figsize=(16,28))

for feature in var:
    i += 1
    plt.subplot(8,4,i)
    sns.kdeplot(t0[feature], bw_method=0.5,label="Class = 0", warn_singular=False)
    sns.kdeplot(t1[feature], bw_method=0.5,label="Class = 1", warn_singular=False)
    plt.xlabel(feature, fontsize=12)
    locs, labels = plt.xticks()
    plt.tick_params(axis='both', which='major', labelsize=12)
plt.show();

# For some of the features we can observe a good selectivity in terms of
# distribution for the two values of Class: V4, V11 have clearly separated
# distributions for Class values 0 and 1, V12, V14, V18 are partially
# separated, V1, V2, V3, V10 have a quite distinct profile, whilst V25, V26,
# V28 have similar profiles for the two values of Class.

# In general, with just few exceptions (Time and Amount), the features
# distribution for legitimate transactions (values of Class = 0) is centered
# around 0, sometime with a long queue at one of the extremities. while
# the fraudulent transactions (values of Class = 1) have a skewed
# (asymmetric) distribution.

# predictive models

target = 'Class'

# Split data in train, test and validation set

#TRAIN/VALIDATION/TEST SPLIT
#VALIDATION
VALID_SIZE = 0.20 # simple validation using train_test_split
TEST_SIZE = 0.20 # test size using_train_test_split
RANDOM_STATE = 333

X = data_df.drop([target, 'Hour'],axis = 1)
Y = data_df[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True )
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )

# on scaling, What’s best for the credit card dataset:
# Max-min Normalization features will have a small Standard deviation
# compared to Standardization. Normalization will scale most of the data
# to a small interval, which means all features will have a small scale
# but do not handle outliers well.
# Whereas, Standardization is robust to outliers. They transform the
# probability distribution for an input variable to standard Gaussian
# Standardization and can become skewed or biased if the input variable
# contains outlier values.
# To overcome this, the median and Interquartile range can be used when
# standardizing numerical input variables which technique is referred to as
# robust scaling.
# Robust scaling uses percentile to scale numerical input variables that
# contain outliers by scaling numerical input variables using the median
# and interquartile range. It calculated the median, 25th, and 75th
# percentiles. The value of each variable is then subtracted with median
# and divided by Interquartile range (IQR)
# Value = (value- median) / (p75 — p25)
# This results in variable having mean to 0, median and standard
# deviation to 1

#RobustScaler is less prone to outliers.

rob_scaler_amount = RobustScaler()
rob_scaler_time = RobustScaler()
rob_scaler_amount.fit(X_train['Amount'].values.reshape(-1,1))
rob_scaler_time.fit(X_train['Time'].values.reshape(-1,1))
X_train['Time'] = rob_scaler_amount.transform(X_train['Amount'].values.reshape(-1,1))
X_train['Amount'] = rob_scaler_time.transform(X_train['Time'].values.reshape(-1,1))
X_train.head()


# class imbalance - oversampling

Y_train.value_counts()
Y_train.value_counts()/Y_train.value_counts().sum()

# since our train data is highly imbalance with way fewer fraudulent case
# (less than 0.2%)
# compared to non-fraudulent case, we can risk achieving high accuracy
# by consistently predicting the majority class but not the minority class.
# to counter this, we can apply oversampling technique, specifically choosing
# ADASYN instead of other techniques like
# SMOTE to balance the highly imbalance training data.

X_train, Y_train = ADASYN().fit_resample(X_train, Y_train)
Y_train.value_counts()

# modelling

# random forest classifier
# Define model parameters
# Let's set the parameters for the model.
# Let's run a model using the training set for training. Then, we will
# use the validation set for validation.
# We will use as validation criterion GINI, which formula is GINI =
# 2 * (AUC) - 1, where AUC is the Receiver Operating
# Characteristic - Area Under Curve (ROC-AUC). Number of estimators is set
# to 100 and number of parallel jobs is set to 4.

# We start by initializing the RandomForestClassifier.

RFC_METRIC = 'gini'  #metric used for RandomForestClassifier
NUM_ESTIMATORS = 100 #number of trees used for RandomForestClassifier
NO_JOBS = -1 #number of parallel jobs used for RandomForestClassifier, -1 uses all cores available

clf = RandomForestClassifier(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)

clf.fit(X_train, Y_train.values)

# Let's now predict the target values for the valid_df data, using predict
# function.

preds = clf.predict(X_valid)

# Let's also visualize the features importance. # split based importance vs
# gain based.

tmp = pd.DataFrame({'Feature': X_train.columns.to_numpy(), 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show() 

# The most important features are V4, V14, V17, V12, V10, V8.

# confusion matrix

cm = pd.crosstab(Y_valid.values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# Type I error and Type II error
# We need to clarify that confussion matrix are not a very good tool to represent the results in the case of largely unbalanced data, because we will actually need a different metrics that accounts in the same time for the selectivity and specificity of the method we are using, so that we minimize in the same time both Type I errors and Type II errors.

# Null Hypothesis (H0) - The transaction is not a fraud.
# Alternative Hypothesis (H1) - The transaction is a fraud.

# Type I error - You reject the null hypothesis when the null hypothesis is actually true.
# Type II error - You fail to reject the null hypothesis when the the alternative hypothesis is true.

# Cost of Type I error - You erroneously presume that the the transaction is a fraud, and a true transaction is rejected.
# Cost of Type II error - You erroneously presume that the transaction is not a fraud and a ffraudulent transaction is accepted.

# AUC-ROC

roc_auc_score(Y_valid.values, preds)


# The ROC-AUC score obtained with RandomForrestClassifier is 0.89.
