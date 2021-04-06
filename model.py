#!/usr/bin/env python
# coding: utf-8

# In[219]:


import pandas as pd
import numpy as np


# In[220]:


df = pd.read_csv("loan_train.csv")
df


# In[221]:


df.isna().sum()


# In[222]:


import seaborn as sns
sns.countplot(df['Gender'], hue = df['Loan_Status'])


# In[223]:


df['Gender'].value_counts()


# In[224]:


df['Gender'].fillna('Male', inplace = True)


# In[225]:


df['Married'].value_counts()
df['Married'].fillna('Yes', inplace = True)


# In[226]:


sns.countplot(df['Dependents'])


# In[227]:


df['Dependents'].fillna('0', inplace = True)


# In[228]:


sns.countplot(df['Self_Employed'])


# In[229]:


df['Self_Employed'].fillna('No', inplace = True)


# In[230]:


sns.kdeplot(df['LoanAmount'])


# In[231]:


df['LoanAmount'].describe()


# In[232]:


df['LoanAmount'].median()


# In[233]:


df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace = True)


# In[234]:


df['Loan_Amount_Term'].describe()


# In[235]:


df['Loan_Amount_Term'].median()


# In[236]:


df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace = True)


# In[237]:


sns.countplot(df['Credit_History'], hue = df['Loan_Status'])


# In[238]:


df['Credit_History'].mode()[0]


# In[239]:


df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace = True)


# In[240]:


df.isna().sum()


# In[241]:


df.head()


# In[242]:


df.info()


# In[243]:


loan_id = df['Loan_ID']
df.drop('Loan_ID',1,inplace = True)


# In[244]:


df['Loan_Status'] = df['Loan_Status'].replace(to_replace = ['Y','N'], value = [1,0])
df.head()


# In[245]:


df['Dependents'].value_counts()


# In[246]:


sns.countplot(df['Loan_Amount_Term'])


# In[247]:


df['Loan_Amount_Term'] = df['Loan_Amount_Term'].astype('object')
df['Credit_History'] = df['Credit_History'].astype('object')


# In[248]:


df.info()


# In[249]:


df = pd.get_dummies(df)
df.head()


# In[250]:


#df['ApplicantIncome'] = (df['ApplicantIncome'] - df['ApplicantIncome'].min())/(df['ApplicantIncome'].max() - df['ApplicantIncome'].min())
#df['CoapplicantIncome'] = (df['CoapplicantIncome'] - df['CoapplicantIncome'].min())/(df['CoapplicantIncome'].max() - df['CoapplicantIncome'].min())
#df['LoanAmount'] = (df['LoanAmount'] - df['LoanAmount'].min())/(df['LoanAmount'].max() - df['LoanAmount'].min())


# In[251]:


df.head()


# In[252]:


y = df['Loan_Status']
y.head()


# In[253]:


X =  df.drop(['Loan_Status'],1)
X.head()


# In[254]:


X.columns


# In[283]:


X = X.rename(columns={"Dependents_3+": "Dependents_3above"})


# In[297]:


X = X.rename(columns={"Education_Not Graduate": "Education_NotGraduate"})


# In[312]:


X = X.rename(columns={"Loan_Amount_Term_12.0": "Loan_Amount_Term_12", "Loan_Amount_Term_36.0": "Loan_Amount_Term_36",
                      "Loan_Amount_Term_60.0": "Loan_Amount_Term_60", "Loan_Amount_Term_84.0": "Loan_Amount_Term_84",
                     "Loan_Amount_Term_120.0": "Loan_Amount_Term_120", "Loan_Amount_Term_180.0": "Loan_Amount_Term_180",
                      "Loan_Amount_Term_240.0": "Loan_Amount_Term_240", "Loan_Amount_Term_300.0": "Loan_Amount_Term_300",
                     "Loan_Amount_Term_360.0": "Loan_Amount_Term_360", "Loan_Amount_Term_480.0": "Loan_Amount_Term_480"})


# In[314]:


X = X.rename(columns={"Credit_History_0.0": "Credit_History_0", "Credit_History_1.0": "Credit_History_1"})


# In[315]:


X.columns


# In[316]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42, stratify = y)


# In[317]:


print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[318]:


from sklearn.ensemble import RandomForestClassifier
randomforest_model = RandomForestClassifier(warm_start = True)


# In[319]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
param_grid = {
    'min_samples_split': range(5,25,5),
    'n_estimators': [100,1700,400] 
    }
# Create a based model
n_folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 4)
#rf = RandomForestClassifier(warm_start=True)
# Instantiate the grid search model
grid_search_rf= GridSearchCV(estimator = randomforest_model, param_grid = param_grid, 
                          cv = n_folds, n_jobs = -1,verbose = 2)


# In[320]:


grid_search_rf.fit(X_train, y_train)


# In[321]:


grid_search_rf.best_score_


# In[322]:


grid_search_rf.best_params_


# In[323]:


randomforest_model = RandomForestClassifier(min_samples_split=15, n_estimators=1700)
randomforest_model.fit(X_train, y_train)


# In[324]:


y_test_pred = randomforest_model.predict(X_test)
y_test_pred


# In[325]:


y_pred_test= randomforest_model.predict(X_test)
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, accuracy_score, recall_score, f1_score
print(" The Confusion matrix is - {}" .format(confusion_matrix(y_test, y_pred_test)))
print("The ROC_AUC score is - {}" .format(roc_auc_score(y_test, y_pred_test)))
print(" The accuracy score is - {}".format(accuracy_score(y_test, y_pred_test)))
print("The recall score is - {}" .format(recall_score(y_test.ravel().astype('int'), y_pred_test.ravel().astype('int'))))
print("The precision score is - {}".format(precision_score(y_test.ravel().astype('int'), y_pred_test.ravel().astype('int'))))
print("The f1-score is - {}".format(f1_score(y_test, y_pred_test)))


# In[340]:


import pickle 
loan_prediction_model_pkl = "loan_prediction_model.pkl"
with open(loan_prediction_model_pkl, 'wb') as file:
    pickle.dump(randomforest_model, file)


# In[327]:


X_test


# In[339]:


X_test.iloc[6]


# In[329]:


X_test.info()


# In[341]:


randomforest_model.predict([[3.523e+03, 3.230e+03, 1.520e+02, 0.000e+00, 1.000e+00, 0.000e+00,
       1.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       1.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00,
       0.000e+00, 1.000e+00, 0.000e+00, 1.000e+00, 0.000e+00, 0.000e+00]])


# In[338]:


X_test.iloc[0]


# In[345]:


if randomforest_model.predict([[4.191e+03, 0.000e+00, 1.200e+02, 0.000e+00, 1.000e+00, 1.000e+00,
       0.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00,
       0.000e+00, 1.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00,
       0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 1.000e+00,
       0.000e+00, 0.000e+00, 1.000e+00, 1.000e+00, 0.000e+00, 0.000e+00]])==1:
    print("Congrats")


# In[ ]:




