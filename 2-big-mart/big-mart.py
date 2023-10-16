#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Load the train and test data set
train = pd.read_csv("C:/Users/ASUS/Desktop/train.csv")
test = pd.read_csv("C:/Users/ASUS/Desktop/test.csv")


# In[3]:


# Check number of rows and columns in train data set
train.shape


# In[4]:


# Print the name of columns in train data set
train.columns


# In[5]:


# Check number of rows and columns in test data set
test.shape


# In[6]:


# Print the name of columns in test data set
test.columns


# In[7]:


# Combine test and train into one file to perform EDA
train["source"] = "train"
test["source"] = "test"
data = pd.concat([train,test],ignore_index=True)
data.shape


# In[8]:


data.head()


# In[9]:


# Describe function for numerical data summary
data.describe()


# In[10]:


# Checking for missing values
data.isnull().sum()


# In[11]:


# The column Item_Weight has two thousand four hundred thirty nine (2439) missing values and Outlet_Size has around (4016).
# Item_Outlet_Sales has (5681) missing values, which we will predict using the model
# Print the unique values in the Item_Fat_Content column, where there are only two unique types of fat content in items: low fat and regular
data["Item_Fat_Content"].unique()


# In[12]:


# Print the unique values in the Outlet_Establishment_Year column, where the date ranges from 1985 to 2009
data["Outlet_Establishment_Year"].unique()


# In[13]:


# Calculate the Outlet_Age
data["Outlet_Age"] = 2018 - data["Outlet_Establishment_Year"]
data.head(2)


# In[14]:


# Unique values in Outlet_Size
data["Outlet_Size"].unique()


# In[15]:


# Printing the count value of Item_Fat_Content column
data["Item_Fat_Content"].value_counts()


# In[16]:


# Print the count value of Outlet_Size
data["Outlet_Size"].value_counts()


# In[17]:


# Use the mode function to find out the most common value in Outlet_Size
data["Outlet_Size"].mode()[0]


# In[18]:


# Two varialbles with missing values - Item_Weight and Outlet_Size
# Replasing missing values in Outlet_Size with the values "medium"
data["Outlet_Size"] = data["Outlet_Size"].fillna(data["Outlet_Size"].mode()[0])


# In[19]:


# Replace missing values in Item_Weight with the mean weight
data["Item_Weight"] = data["Item_Weight"].fillna(data["Item_Weight"].mean())


# In[20]:


# Plot a histogram to reveal the distribution of Item_Visibility column
data["Item_Visibility"].hist(bins=20)


# In[21]:


# Detecting outliers
# An outlier is a data point that lies outside the overall pattern in a distribution.
# A commonly used rule states that a data point is an outlier if it is more than 1.5*IQR above the third quartile or below the first quartile
# Using this, one can remove the outliers and output the resulting data in fill_data variable.
# Calculate first quantile for Item_Visibility
Q1 = data['Item_Visibility'].quantile(0.25)


# In[22]:


# Calculate third quantile for Item_Visibility 
Q3 = data['Item_Visibility'].quantile(0.75)


# In[23]:


# Calculate the inter quantile range (IQR) for Item_Visibility 
IQR = Q3 - Q1


# In[24]:


# Now that the IQR range is known, remove the outliers from the data
# The resulting data is stored in fill_data variale
fill_data = data.query('(@Q1 - 1.5 * @IQR) <= Item_Visibility <= (@Q3 + 1.5 * @IQR)')


# In[25]:


# Display the data
fill_data.head(2)


# In[26]:


# Check the shape of the resulting dataset without the outliers
fill_data.shape


# In[27]:


# Shape of the original dataset is (14204) rows and fourteen columns with outliers
data.shape


# In[28]:


# Assign fill_data dataset to DataFrame
data = fill_data


# In[29]:


data.shape


# In[30]:


# Modify Item_Visibility by converting the numerical values into the categories Low Visibility, Visibility and High Visibility
data['Item_Visibility_bins'] = pd.cut(data['Item_Visibility'], [0.000, 0.005, 0.13, 0.2], labels = ['Low Viz', 'Viz', 'High Viz'])


# In[31]:


# Print the count of Item_Visibility_bins
data['Item_Visibility_bins'].value_counts()


# In[32]:


# Replace null values with Low Visibility
data['Item_Visibility_bins'] = data['Item_Visibility_bins'].replace(np.nan, "Low Viz", regex = True)


# In[33]:


# We found types and differences in representation in categories of Item_Fat_Content variable
# This can be corrected using the code on screen
# Replace all representations of low fat with Low Fat 
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')


# In[34]:


# Replace all representations of reg with Regular
data['Item_Fat_Content'] = data["Item_Fat_Content"].replace("reg", "Regular")


# In[35]:


# Print unique fat count values
data["Item_Fat_Content"].unique()


# In[36]:


# Code all categorical variables as numeric using LabelEncoder from sklearn's preprocessing module
# Initialize the label encoder
le = LabelEncoder()


# In[37]:


# Transform Item_Fat_Content
data["Item_Fat_Content"] = le.fit_transform(data["Item_Fat_Content"])


# In[38]:


# Transform Item_Visibility_bins
data["Item_Visibility_bins"] = le.fit_transform(data["Item_Visibility_bins"])


# In[39]:


# Transform Outlet_Size
data["Outlet_Size"] = le.fit_transform(data["Outlet_Size"])


# In[40]:


# Transform Outlet_Location_Type
data["Outlet_Location_Type"] = le.fit_transform(data["Outlet_Location_Type"])


# In[41]:


# Print the uniqe values of Outlet_Type
data["Outlet_Type"].unique()


# In[42]:


# Create dummies for Outlet_Type
dummy = pd.get_dummies(data["Outlet_Type"])
dummy.head()


# In[43]:


# Explore the column Item_Identifier
data["Item_Identifier"]


# In[44]:


# As there are multiple values of Food, nonconsumable items, and drinks with different numbers, combine the item type.
data["Item_Identifier"].value_counts()


# In[45]:


# As multiple categories are present in Item_Identifier, reduce this by mapping
data["Item_Type_Combined"] = data["Item_Identifier"].apply(lambda x: x[0:2])
data["Item_Type_Combined"] = data["Item_Type_Combined"].map({'FD': 'Food',
                                                            'NC': 'Non-Consumable',
                                                            'DR': 'Drinks'})


# In[46]:


# Only three categories are present in an Item_Type_Combined column
data["Item_Type_Combined"].value_counts()


# In[47]:


data.shape


# In[48]:


# Perform one-hot encoding for all columns as the model works on neumerical values and not on categorical values
data = pd.get_dummies(data, columns=["Item_Fat_Content", "Outlet_Location_Type", "Outlet_Size", "Outlet_Type", "Item_Type_Combined"])


# In[49]:


data.dtypes


# In[50]:


import warnings
warnings.filterwarnings('ignore')

# Drop the colums which have been converted to different types
data.drop(["Item_Type", "Outlet_Establishment_Year"], axis=1, inplace=True)

# Devide the dataset created earlier into train and test datasets
train = data.loc[data["source"] == "train"]
test = data.loc[data["source"] == "test"]

# Drop unnecessary columns
test.drop(["Item_Outlet_Sales", "source"], axis=1, inplace=True)
train.drop(["source"], axis=1, inplace=True)

# Export modified versions of the files
train.to_csv("trained data/train_modified.csv", index=False)
test.to_csv("trained data/test_modified.csv", index=False)


# In[51]:


# read the train_modified.csv and test_modified.csv datasets
train2 = pd.read_csv("trained data/train_modified.csv")
test2 = pd.read_csv("trained data/test_modified.csv")


# In[52]:


# Print the data types of train2 column
train2.dtypes


# In[53]:


# Drop the irrelevant variables from train2 dataset
# Create the independent variable X_train and dependent variable y_train
X_train = train2.drop(["Item_Outlet_Sales", "Outlet_Identifier", "Item_Identifier"], axis=1)
y_train = train2.Item_Outlet_Sales


# In[54]:


# Drop those irrelevent variables from test2 dataset
X_test = test2.drop(["Outlet_Identifier", "Item_Identifier"], axis=1)


# In[55]:


X_test


# In[56]:


X_train.head(2)


# In[57]:


y_train.head(2)


# In[58]:


# Import sklearn libraries for model selection
from sklearn import model_selection
from sklearn.linear_model import LinearRegression


# In[59]:


# Create a train and test split
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(X_train, y_train, test_size=0.3,random_state=42)


# In[60]:


# Fit linear regression to the training dataset
lin = LinearRegression()


# In[61]:


lin.fit(xtrain, ytrain)


# In[62]:


# Find the coefficient and intercept of the line
# Use xtrain and ytrain for linear regression
print(lin.coef_)
lin.intercept_


# In[63]:


# Predict the test set results of training data
predictions = lin.predict(xtest)
predictions


# In[64]:


import math


# In[65]:


# Find the RMSE for the model
print(math.sqrt(mean_squared_error(ytest,predictions)))


# In[66]:


# A good RMSE for this problem is 1130. Here we can improve the RMSE by using algorithms like decision tree, random forest and XGboost.
# Next, we will predict the sales of each product at a particular store in test data.
# Predict the column Item_Outlet_Sales of test dataset
y_sales_pred = lin.predict(X_test)
y_sales_pred


# In[67]:


test_predictions = pd.DataFrame({
    'Item_Identifier': test2['Item_Identifier'],
    'Outlet_Identifier': test2['Outlet_Identifier'],
    'Item_Outlet_Sales': y_sales_pred
}, columns = ['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])


# In[68]:


test_predictions


# In[ ]:




