#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


dataset = pd.read_csv("D:/Datasets/Titanic-Dataset.csv")
dataset.head()


# In[3]:


dataset.info()


# In[4]:


#HANDLING MISSING VALUES
dataset.dropna(axis=0,inplace=True)
dataset.shape
dataset.head()


# In[16]:


X = dataset[['Pclass','Age','SibSp','Parch','Fare']]
Y = dataset['Survived']

# Train-and-Test -Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 4)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[15]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)


# In[18]:


# Evaluation and accuracy
print('Accuracy of the model is =',accuracy_score(y_test, y_pred))


# In[20]:


# Confusion matrix 
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data = cm, 
                           columns = ['Predicted:0', 'Predicted:1'], 
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (8, 5))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens")
plt.show()
print('The details for confusion matrix is =')
print (classification_report(y_test, y_pred))


# In[24]:


#USING RANDOM FOREST
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.2,random_state=40)
rfc = RandomForestClassifier(n_estimators=3,max_depth=2,random_state=42)
# Fit RandomForestClassifier
rfc.fit(X_train, y_train)
# Predict the test set labels
y_pred = rfc.predict(X_test)
print('Accuracy of the model is =',accuracy_score(y_test, y_pred))


# In[22]:


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
print(classification_report(y_test,y_pred))


# In[ ]:




