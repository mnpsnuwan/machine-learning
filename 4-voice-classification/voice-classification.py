#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('C:/Users/ASUS/voice.csv')
df.head()


# In[3]:


# Dataset info
df.info()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


print("Shape of data: ", df.shape)
print("Total number of labels: {}".format(df.shape[0]))
print("Number of male: {}".format(df[df.label=='male'].shape[0]))
print("Number of female: {}".format(df[df.label=='female'].shape[0]))


# In[7]:


X = df.iloc[:,:-1]
print(df.shape)
print(X.shape)


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


y = df.iloc[:,-1]

gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
y


# In[10]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# In[12]:


from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix


# In[13]:


svc_model = SVC()
svc_model.fit(X_train,y_train)
y_pred = svc_model.predict(X_test)


# In[14]:


print('Accuracy Score: ')
print(metrics.accuracy_score(y_test,y_pred))


# In[15]:


print(confusion_matrix(y_test,y_pred))


# In[16]:


from sklearn.model_selection import GridSearchCV


# In[17]:


param_grid = {'C':[0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001]}


# In[18]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


# In[19]:


grid_predictions = grid.predict(X_test)


# In[20]:


print('Accuracy Score: ')
print(metrics.accuracy_score(y_test,grid_predictions))


# In[21]:


print(confusion_matrix(y_test,grid_predictions))


# In[22]:


print(classification_report(y_test,grid_predictions))


# In[ ]:




