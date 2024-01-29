# Data Preprocessing

import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import gc
from datetime import datetime 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
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
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer

%matplotlib inline 
init_notebook_mode(connected=True)
pd.set_option('display.max_columns', 100)

data_df = pd.read_csv('creditcard.csv')

print(data_df.shape)
# data has rows: 284807  columns: 31

data_df.head()

data_df.describe()


# Exploratory Data Analysis
# check missing data

total = data_df.isnull().sum().sort_values(ascending = False)
percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending = False)
pd.concat([total, percent], axis=1, keys=['Total', 'Percent']).transpose()


# check for data imbalance

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


# time density plot for transactions

class_0 = data_df.loc[data_df['Class'] == 0]["Time"]
class_1 = data_df.loc[data_df['Class'] == 1]["Time"]

hist_data = [class_0, class_1]
group_labels = ['Not Fraud', 'Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
fig['layout'].update(title='Credit Card Transactions Time Density Plot', xaxis=dict(title='Time [s]'))
iplot(fig, filename='dist_only')


# statistical summaries of fraud and non-fraud transactions by hour

data_df['Hour'] = data_df['Time'].apply(lambda x: np.floor(x / 3600))

tmp = data_df.groupby(['Hour', 'Class'])['Amount'].aggregate(['min', 'max', 'count', 'sum', 'mean', 'median', 'var']).reset_index()
df = pd.DataFrame(tmp)
df.columns = ['Hour', 'Class', 'Min', 'Max', 'Transactions', 'Sum', 'Mean', 'Median', 'Var']
df.head()


# total amount by class, non-fraud is blue and fraud is red

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Sum", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Sum", data=df.loc[df.Class==1], color="red")
plt.suptitle("Total Amount")
plt.show();


# total number of transactions by class, non-fraud is blue and fraud is red

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Transactions", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Transactions", data=df.loc[df.Class==1], color="red")
plt.suptitle("Total Number of Transactions")
plt.show();


# average amount per transaction by class, non-fraud is blue and fraud is red

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Mean", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Mean", data=df.loc[df.Class==1], color="red")
plt.suptitle("Average Amount per Transaction")
plt.show();


# maximum amount of transaction by class, non-fraud is blue and fraud is red

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Mean", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Mean", data=df.loc[df.Class==1], color="red")
plt.suptitle("Maximum Amount of Transactions")
plt.show();


# median amount of transaction by class, non-fraud is blue and fraud is red

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Median", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Median", data=df.loc[df.Class==1], color="red")
plt.suptitle("Median Amount of Transactions")
plt.show();


# minimum amount of transaction by class, non-fraud is blue and fraud is red

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,6))
s = sns.lineplot(ax = ax1, x="Hour", y="Min", data=df.loc[df.Class==0])
s = sns.lineplot(ax = ax2, x="Hour", y="Min", data=df.loc[df.Class==1], color="red")
plt.suptitle("Minimum Amount of Transactions")
plt.show();


# summary statistics of both classes, with and without outliers

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
s = sns.boxplot(ax = ax1, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=True)
s = sns.boxplot(ax = ax2, x="Class", y="Amount", hue="Class",data=data_df, palette="PRGn",showfliers=False)
plt.show();

tmp = data_df[['Amount','Class']].copy()
class_0 = tmp.loc[tmp['Class'] == 0]['Amount']
class_1 = tmp.loc[tmp['Class'] == 1]['Amount']
class_0.describe()

class_1.describe()


# plot of fradulent transactions, amount against time in seconds

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


# regression line for Amount against V20 and V7

s = sns.lmplot(x='V20', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='V7', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})
plt.show()


# regression line for Amount against V2 and V5

s = sns.lmplot(x='V2', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})
s = sns.lmplot(x='V5', y='Amount',data=data_df, hue='Class', fit_reg=True,scatter_kws={'s':2})
plt.show()


# Features density plot

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


# Feature Engineering
# stratified Train/Validation/Test split

target = 'Class'

VALID_SIZE = 0.20 # simple validation using train_test_split
TEST_SIZE = 0.20 # test size using_train_test_split
RANDOM_STATE = 333

X = data_df.drop([target, 'Hour'],axis = 1)
Y = data_df[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y, random_state=RANDOM_STATE, shuffle=True)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=VALID_SIZE, stratify=Y_train, random_state=RANDOM_STATE, shuffle=True)


# robust scaling

rob_scaler_amount = RobustScaler()
rob_scaler_time = RobustScaler()
rob_scaler_amount.fit(X_train['Amount'].values.reshape(-1,1))
rob_scaler_time.fit(X_train['Time'].values.reshape(-1,1))
X_train['Time'] = rob_scaler_amount.transform(X_train['Amount'].values.reshape(-1,1))
X_train['Amount'] = rob_scaler_time.transform(X_train['Time'].values.reshape(-1,1))
X_valid['Time'] = rob_scaler_amount.transform(X_valid['Amount'].values.reshape(-1,1))
X_valid['Amount'] = rob_scaler_time.transform(X_valid['Time'].values.reshape(-1,1))
X_test['Time'] = rob_scaler_amount.transform(X_test['Amount'].values.reshape(-1,1))
X_test['Amount'] = rob_scaler_time.transform(X_test['Time'].values.reshape(-1,1))
X_train.head()
X_valid.head()
X_test.head()


# oversampling train data with ADASYN

X_train, Y_train = ADASYN(random_state=RANDOM_STATE).fit_resample(X_train, Y_train)
Y_train.value_counts()


# Modelling
# Random Forest Classifier
# fitting the RFC model

RFC_METRIC = 'gini'
NUM_ESTIMATORS = 100
NO_JOBS = -1

clf = RandomForestClassifier(n_jobs=NO_JOBS, 
                             random_state=RANDOM_STATE,
                             criterion=RFC_METRIC,
                             n_estimators=NUM_ESTIMATORS,
                             verbose=False)

clf.fit(X_train, Y_train.values)

preds = clf.predict(X_valid)


# feature importance

tmp = pd.DataFrame({'Feature': X_train.columns.to_numpy(), 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show() 


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


# AUC-ROC

roc_auc_score(Y_valid.values, preds)


# AdaBoost Classifier
# fitting the AdaBoost model

clf = AdaBoostClassifier(random_state=RANDOM_STATE,
                         algorithm='SAMME.R',
                         learning_rate=0.8,
                         n_estimators=NUM_ESTIMATORS)

clf.fit(X_train, Y_train.values)

preds = clf.predict(X_valid)


# feature importance

tmp = pd.DataFrame({'Feature': X_train.columns.to_numpy(), 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()   


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


# AUC-ROC

roc_auc_score(Y_valid.values, preds)


# CatBoost Classifier

VERBOSE_EVAL = 50

clf = CatBoostClassifier(iterations=500,
                         learning_rate=0.02,
                         depth=12,
                         eval_metric='AUC',
                         random_seed = RANDOM_STATE,
                         bagging_temperature = 0.2,
                         od_type='Iter',
                         metric_period = VERBOSE_EVAL,
                         od_wait=100)

clf.fit(X_train, Y_train.values, eval_set=[(X_train, Y_train), (X_valid, Y_valid)])

preds = clf.predict(X_test)


# feature importance

tmp = pd.DataFrame({'Feature': X_train.columns.to_numpy(), 'Feature importance': clf.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()  


# confusion matrix
cm = pd.crosstab(Y_test.values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# AUC-ROC

roc_auc_score(Y_test.values, preds)


# XGBoost
# prepare the train and valid datasets
dtrain = xgb.DMatrix(X_train, Y_train.values)
dvalid = xgb.DMatrix(X_valid, Y_valid.values)
dtest = xgb.DMatrix(X_test, Y_test.values)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


# set parameters
params = {}
params['objective'] = 'binary:logistic'
params['eta'] = 0.01
params['silent'] = True
params['max_depth'] = 2
params['subsample'] = 0.8
params['colsample_bytree'] = 0.9
params['eval_metric'] = 'auc'
params['random_state'] = RANDOM_STATE


# training the model

MAX_ROUNDS = 1000
EARLY_STOP = 50

model = xgb.train(params, 
                  dtrain, 
                  MAX_ROUNDS, 
                  watchlist, 
                  early_stopping_rounds=EARLY_STOP, 
                  maximize=True, 
                  verbose_eval=VERBOSE_EVAL)


# feature importance

fig, (ax) = plt.subplots(ncols=1, figsize=(8,5))
xgb.plot_importance(model, height=0.8, title="Features importance (XGBoost)", ax=ax, color="green") 
plt.show()


# confusion matrix
preds = model.predict(dtest)
preds = [round(value) for value in preds]

cm = pd.crosstab(Y_test.values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# AUC-ROC

roc_auc_score(Y_test.values, preds)


# LightGBM
# training the model

lgbm = LGBMClassifier(boosting_type= 'gbdt',
                      objective= 'binary',
                      metric='auc',
                      learning_rate= 0.01,
                      num_leaves= 7,  
                      max_depth= 4,  
                      min_child_samples= 100, 
                      max_bin= 100,  
                      subsample= 0.9, 
                      subsample_freq= 1, 
                      colsample_bytree= 0.7,  
                      min_child_weight= 0,  
                      min_split_gain= 0,  
                      n_jobs=os.cpu_count(),
                      verbose= 1,)


# fitting the model

lgbm.fit(X_train, Y_train.values, eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        eval_metric='auc')

preds = lgbm.predict(X_test)


# feature importance

tmp = pd.DataFrame({'Feature': X_train.columns.to_numpy(), 'Feature importance': lgbm.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()   


# confusion matrix

cm = pd.crosstab(Y_test.values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# AUC-ROC

roc_auc_score(Y_test.values, preds)


# Bayesian optimization, stratified K-Fold cross validation
# set initial parameters

X = data_df.drop([target, 'Hour'],axis = 1)
Y = data_df[target]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, stratify=Y, random_state=RANDOM_STATE, shuffle=True)

lgbm = LGBMClassifier(boosting_type='dart',
                      objective='binary',
                      metric='auc',
                      n_jobs=os.cpu_count(), 
                      verbose=-1,
                      random_state=RANDOM_STATE,
                      min_split_gain= 0, # or can regularize next step by searching L1/L2 
                      )


# specifying search space

search_spaces = {
    'learning_rate': Real(0.001, 0.01, 'log-uniform'),   
    'n_estimators' : Integer(50, 200),
    'num_leaves': Integer(10, 100),                    
    'max_depth': Integer(15, 100),                       
    'subsample': Real(0.1, 1.0, 'uniform'),           
    'subsample_freq': Integer(0, 10),                   
    'min_child_samples': Integer(10, 200),  
    #'reg_lambda': Real(1e-9, 100.0, 'log-uniform'),      # L2 regularization
    #'reg_alpha': Real(1e-9, 100.0, 'log-uniform'),       # L1 regularization
    }


# setting up the optimizer

opt = BayesSearchCV(estimator=lgbm,                                    
                    search_spaces=search_spaces,                                              
                    cv=5, # stratified K-Fold automatically used here (refer to documentation)
                    n_iter=30,                                   
                    n_points=3,                                       
                    n_jobs=-1,                                       
                    return_train_score=False,                         
                    refit=False,                                      
                    optimizer_kwargs={'base_estimator': 'GP'},       
                    random_state=RANDOM_STATE)


# running the optimizer

np.int = np.int64 # skopt uses np.int which was deprecated, so we change it to np.int64 manually
opt.fit(X_train, Y_train)


# results

best_param = opt.best_params_
opt.best_params_


# fitting the test set

lgbm = LGBMClassifier(boosting_type='dart',
                      objective='binary',
                      metric='auc',
                      n_jobs=os.cpu_count(), 
                      verbose=-1,
                      random_state=RANDOM_STATE,
                      min_split_gain= 0,
                      learning_rate=best_param['learning_rate'],
                      max_depth=best_param['max_depth'],
                      min_child_samples=best_param['min_child_samples'],
                      n_estimators=best_param['n_estimators'],
                      num_leaves=best_param['num_leaves'],
                      subsample=best_param['subsample'],
                      subsample_freq=best_param['subsample_freq'],
                      )

lgbm.fit(X_train, Y_train.values)

preds = lgbm.predict(X_test)


# feature importance

tmp = pd.DataFrame({'Feature': X_train.columns.to_numpy(), 'Feature importance': lgbm.feature_importances_})
tmp = tmp.sort_values(by='Feature importance',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Features importance',fontsize=14)
s = sns.barplot(x='Feature',y='Feature importance',data=tmp)
s.set_xticklabels(s.get_xticklabels(),rotation=90)
plt.show()  


# confusion matrix

cm = pd.crosstab(Y_test.values, preds, rownames=['Actual'], colnames=['Predicted'])
fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))
sns.heatmap(cm, 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            annot=True,ax=ax1,
            linewidths=.2,linecolor="Darkblue", cmap="Blues")
plt.title('Confusion Matrix', fontsize=14)
plt.show()


# AUC-ROC

roc_auc_score(Y_test.values, preds)