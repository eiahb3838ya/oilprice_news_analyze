#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn import svm
from sklearn.metrics import classification_report,confusion_matrix,consensus_score
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE,RandomOverSampler,ADASYN
from imblearn.under_sampling import AllKNN
from sklearn.feature_selection import SelectKBest,SelectPercentile,chi2,mutual_info_,f_classif
# ## Best Result 0.97

# In[3]:
DATA_ROOT = "../../../data/"

def run_predict():
    test_df = pd.read_csv(DATA_ROOT+"train_test_dataset/oilprice_cnbc_new_test.csv")
    train_df = pd.read_csv(DATA_ROOT+"train_test_dataset/oilprice_cnbc_new_train.csv")


    # In[4]:


    x_drop = ['tags','date']
    train_x = train_df.drop(x_drop,axis=1).values.astype(float)
    train_y = train_df['tags'].values.astype(float)
    test_x = test_df.drop(x_drop,axis=1).values.astype(float)
    test_y = test_df['tags'].values.astype(float)


    # In[5]:


    normalizer = Normalizer()
    train_x = normalizer.fit_transform(train_x)
    test_x = normalizer.transform(test_x)


    # In[6]:


    # Perform classification with SVM, kernel=linear 
    svc_model = svm.LinearSVC(C=100) 
    svc_model.fit(train_x, train_y) 
    prediction = svc_model.predict(test_x)

    print (classification_report(test_y, prediction))

test_df = pd.read_csv(DATA_ROOT+"train_test_dataset/oilprice_cnbc_new_test.csv")
train_df = pd.read_csv(DATA_ROOT+"train_test_dataset/oilprice_cnbc_new_train.csv")


# In[4]:


x_drop = ['tags','date']
train_x = train_df.drop(x_drop,axis=1).values.astype(float)
train_y = train_df['tags'].values.astype(float)
test_x = test_df.drop(x_drop,axis=1).values.astype(float)
test_y = test_df['tags'].values.astype(float)


# # In[5]:
# sm = SMOTE()
# train_x, train_y = sm.fit_resample(train_x,train_y)
# print("X_res.shape",train_x.shape)
# print("y_res.shape",train_y.shape)

# pca=PCA()
# pca.fit(train_x)
# train_x = pca.transform(train_x)
# test_x = pca.transform(test_x)


normalizer = Normalizer()
train_x = normalizer.fit_transform(train_x)
test_x = normalizer.transform(test_x)


# In[6]:

# selector=SelectPercentile(chi2, alpha)
# train_x=selector.fit_transform(train_x,train_y)
# test_x=selector.transform(test_x)


# Perform classification with SVM, kernel=linear 
svc_model = svm.LinearSVC(C=100) 
svc_model.fit(train_x, train_y) 
prediction = svc_model.predict(test_x)

print (classification_report(test_y, prediction))
print(confusion_matrix(test_y, prediction))
