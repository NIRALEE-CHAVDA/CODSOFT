#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[7]:


dataset = pd.read_csv("D:/Datasets/creditcard.csv")
dataset.head()


# In[8]:


dataset.info()


# In[9]:


dataset.describe().T #T transpose the table


# In[10]:


corr_matrix = dataset.corr()
corr_matrix


# In[11]:


print(dataset['Class'].unique())
array = dataset.values
X = array[:,0:30]
Y = array[:,30]


# In[12]:


target_corr = corr_matrix['Class']     # Select the correlations of the target variable
negative_corr = target_corr[target_corr < 0]   # Filter for negative correlations
negative_corr_features = negative_corr.index.tolist()  # Get the feature names as a list
print("\nFeatures with Negative Correlation:")
print(negative_corr_features)

# Prepare data for feature selection
X = dataset[negative_corr_features]
y = dataset['Class']


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=40)
rfc = RandomForestClassifier(n_estimators=3,max_depth=2,random_state=42)
# Fit RandomForestClassifier
rfc.fit(X_train, y_train)
# Predict the test set labels
y_pred = rfc.predict(X_test)


# In[14]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d').set_title('risks confusion matrix (0 = low risk, 1 = high risk)')
print(classification_report(y_test,y_pred))


# In[ ]:




