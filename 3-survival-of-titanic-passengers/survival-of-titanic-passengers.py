#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style="white", color_codes=True)
sns.set(font_scale=1.5)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics


# In[2]:


df_train = pd.read_csv("C:/Users/ASUS/train.csv")
df_train.shape


# In[3]:


df_train.Survived.value_counts()


# In[4]:


df_train.Sex.value_counts()


# In[5]:


df_train.Embarked.value_counts()


# In[6]:


df_train.isnull().sum()


# In[7]:


df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df_train.head()


# In[8]:


def age_approx(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[9]:


df_train.groupby(['Pclass']).mean()


# In[10]:


df_train['Age'] = df_train[['Age','Pclass']].apply(age_approx, axis=1)


# In[11]:


df_train.isnull().sum()


# In[12]:


df_train.dropna(inplace=True)
df_train.isnull().sum()


# In[13]:


df_train.dtypes


# In[14]:


df_train_dummied = pd.get_dummies(df_train, columns=["Sex"])


# In[15]:


df_train_dummied = pd.get_dummies(df_train_dummied, columns=["Embarked"])


# In[16]:


df_train_dummied.head()


# In[19]:


plt.figure(figsize=(6,4))
sns.heatmap(df_train_dummied.corr())


# In[20]:


used_features = ["Pclass", "Age", "SibSp", "Parch", "Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S"]

X = df_train_dummied[used_features].values
y = df_train_dummied['Survived']


# In[21]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)


# In[22]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[23]:


LogReg = LogisticRegression()


# In[24]:


LogReg.fit(X_train, y_train)


# In[25]:


y_pred = LogReg.predict(X_test)


# In[26]:


metrics.confusion_matrix(y_test,y_pred)


# In[27]:


metrics.accuracy_score(y_test,y_pred)


# In[28]:


len(X_test)


# In[29]:


print(classification_report(y_test,y_pred))


# In[30]:


LogReg.coef_


# In[31]:


LogReg.intercept_


# In[32]:


df_train_dummied[used_features].columns


# In[33]:


LogReg.predict_proba(X_test)


# In[ ]:




