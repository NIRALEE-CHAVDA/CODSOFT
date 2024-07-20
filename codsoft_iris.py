#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# In[15]:


dataset = pd.read_csv("D:/Datasets/IRIS.csv")
dataset.reset_index(drop=True, inplace=True)
dataset.head()


# In[4]:


dataset.info()


# In[5]:


#HANDLING MISSING VALUES
dataset.dropna(axis=0,inplace=True)
dataset.shape
dataset.head()


# In[6]:


X = dataset[['sepal_length','sepal_width']]
Y = dataset['species']

# Train-and-Test -Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 4)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[23]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# In[24]:


# Evaluation and accuracy
print('Accuracy of the model is =',accuracy_score(y_test, y_pred))


# In[25]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize = (8, 5))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = "Greens")
plt.show()
print('The details for confusion matrix is =')

print (classification_report(y_test, y_pred))


# In[28]:


#USING RANDOM FOREST
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2,random_state=40)
rfc = RandomForestClassifier(n_estimators=3,max_depth=2,random_state=42)
# Fit RandomForestClassifier
rfc.fit(X_train, y_train)
# Predict the test set labels
y_pred = rfc.predict(X_test)
print('Accuracy of the model is =',accuracy_score(y_test, y_pred))


# In[29]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print(classification_report(y_test,y_pred))


# In[41]:


#USING decision tree classifier
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.4,random_state=42)
clf = DecisionTreeClassifier(random_state=42)
# Fit RandomForestClassifier
clf.fit(X_train, y_train)
# Predict the test set labels
y_pred = clf.predict(X_test)
print('Accuracy of the model is =',accuracy_score(y_test, y_pred))


# In[ ]:




