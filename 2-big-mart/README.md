# Big-Mart-Sales
# This is the DataSet of Hackathon held on Analytics Vidya 
link for this Hackathon : https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/

LeaderBoard RMSE value : 1158.29

Problem Statement:

The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. 
Also, certain attributes of each product and store have been defined. 
The aim is to build a predictive model and find out the sales of each product at a particular store.
Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.
Please note that the data may have missing values as some stores might not report all the data due to technical glitches. 
Hence, it will be required to treat them accordingly.

Data
We have train (8523) and test (5681) data set, train data set has both input and output variable(s). You need to predict the sales for test data set.

Variable :  Description

Item_Identifier :  Unique product ID

Item_Weight :  Weight of product

Item_Fat_Content : Whether the product is low fat or not

Item_Visibility : The % of total display area of all products in a store allocated to the particular product

Item_Type : The category to which the product belongs

Item_MRP : Maximum Retail Price (list price) of the product

Outlet_Identifier : Unique store ID

Outlet_Establishment_Year : The year in which store was established

Outlet_Size : The size of the store in terms of ground area covered

Outlet_Location_Type : The type of city in which the store is located

Outlet_Type : Whether the outlet is just a grocery store or some sort of supermarket

Item_Outlet_Sales : Sales of the product in the particulat store. This is the outcome variable to be predicted.

My approch:
after reading and analyzing data it is found that:

Item_Fat_Content has catagories ['Low Fat', 'reg', 'Regular', 'LF', 'low fat'] 
Corrected the misspeled catagories and converted them to  
'LF', 'low fat' => 'Low Fat'
'reg' => 'Regular'

There are some missing values in Outlet_size and Item_Weight
for  Item_Weight missing values are filled by mean of the column
and Outlet_size missing values are filled by mode of the column i.e 'Medium'

created new column called num_years
num_years indicate that how old the outlet is.

then applied different models for prediction
LinearRegression , SVM, RandomForestRegressor, XGBoost

Evaluated the model by checking RMSE value,
XGBoost gave the best result
that is RMSE = 1098.29


