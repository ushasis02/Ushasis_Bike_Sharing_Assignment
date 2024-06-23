#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the key libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#reading the dataset into a dataframe
df = pd.read_csv('day.csv')
df.head()


# In[3]:


#shape of the dataframe
df.shape


# In[4]:


#details on the data provided
df.info()


# In[5]:


#Confirm that there is no non-null data in the dataframe
df.isnull().sum()


# In[6]:


# data dictionary of day.csv for quick reference:
# instant: record index
# dteday : date
# season : season (1:spring, 2:summer, 3:fall, 4:winter)
# yr : year (0: 2018, 1:2019)
# mnth : month ( 1 to 12)
# holiday : weather day is a holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
# weekday : day of the week
# workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# weathersit : 
# - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# temp : temperature in Celsius
# atemp: feeling temperature in Celsius
# hum: humidity
# windspeed: wind speed
# casual: count of casual users
# registered: count of registered users
# cnt: count of total rental bikes including both casual and registered


# In[7]:


#Identifying Numeric and Catgeorical Variables based on the number of unique values
df.nunique()


# In[8]:


#separating the numerical columns and categorical columns based on the unique values and data dictionary 
num_cols=['instant','dteday','temp','atemp','hum','windspeed','casual','registered','cnt']
cat_cols=['season','yr','mnth','holiday','weekday','workingday','weathersit']


# In[9]:


#Reducing the no. of columns for analysis based on logical reasoning of the data dictionary and unique values
# Instant and dteday are unique for every record in the data, so we can drop them for the analysis, so dropping them from num_cols
# cnt is the target variable which is a sum of 'casual' and 'registered', so we can drop these 2 variables from num_col too
# not-dropping anything from the categorical data cat_cols
num_cols=['temp','atemp','hum','windspeed','cnt']


# In[10]:


# doing univariate analysis through histogram for numeric variables and countplot for categorical variables
for i in num_cols:
    sns.histplot(x=df[i])
    plt.show()
for j in cat_cols:
    sns.countplot(x=df[j])
    plt.show()


# In[11]:


# key output of the univariate analysis:
# 1) All the numerical variables seems to have a good representation in the data set and there are no causes of concern. 
# 2) Amongst numrical variables, 'temp' and 'atemp' data spread seems to be similar. Need to check the correlation in bivariate analysis.
# 3) All the categorical varaibles seems to be have a good representation in the data set except:
#      i) when 'holiday' = 1, and
#      ii) when 'weathersit' =3

# doing bivariate analysis for the numerical columns through a pairplot
sns.pairplot(df[num_cols])
plt.show()


# In[12]:


# key output of the bi-variate analysis:
# 1) We cab see that 'temp' and 'atemp' is absolutely correlated, so we can drop on one of these columns. 
# 2) The target variable 'cnt' seems to be linearly related with 'temp' and 'atemp'
# Plotting the numeric variables against target 'cnt' and plotting a heatmap to re-validate the conclusions above
for i in num_cols:
    sns.scatterplot(x=df[i], y=df['cnt'])
    plt.show()


# In[13]:


# there seems to be negative correlation between target variable 'cnt' and humidity 'hum'
# plotting the heatmap for multivariate analysis
plt.figure(figsize=(12,6))
sns.heatmap(df[num_cols].corr(), annot=True)
plt.show()


# In[14]:


# dropping 'atemp' from the numeric variables for analysis
num_cols=['temp','hum','windspeed','cnt']


# In[15]:


#one hot encoding - for categorical variables which have more than 2 categories we need to create a dummy variable
df[cat_cols].nunique()


# In[16]:


#num of dummies is n-1, create the dummy columns for 'season', 'mnth', 'weekday' and 'weathersit' variable as it has 4,12,7 and 3 types of categorical data
dum1=pd.get_dummies(df['season'], drop_first = True, dtype =int)
dum2=pd.get_dummies(df['mnth'], drop_first = True, dtype =int)
dum3=pd.get_dummies(df['weekday'], drop_first = True, dtype =int)
dum4=pd.get_dummies(df['weathersit'], drop_first = True, dtype =int)


# In[18]:


# getting the shape of the dummy column for 'season'
dum1.shape


# In[19]:


# getting the shape of the dummy column for 'mnth'
dum2.shape


# In[20]:


# getting the shape of the dummy column for 'weekday'
dum3.shape


# In[21]:


# getting the shape of the dummy column for 'weathersit'
dum4.shape


# In[22]:


dum1.head()


# In[23]:


dum2.head()


# In[24]:


dum3.head()


# In[25]:


dum4.head()


# In[27]:


#Need to rename the columns to make it relevant while concatenating in the main data frame
dum1.columns =['sn2','sn3','sn4']
dum1.head()


# In[28]:


#Renaming the columns in the other dummies
dum2.columns =['mn2','mn3','mn4','mn5','mn6','mn7','mn8','mn9','mn10','mn11','mn12']
dum2.head()


# In[29]:


#Renaming the columns in the other dummies
dum3.columns =['wk1','wk2','wk3','wk4','wk5','wk6']
dum3.head()


# In[30]:


#Renaming the columns in the other dummies
dum4.columns =['wth2','wth3']
dum4.head()


# In[34]:


# Cocatenating the dummy variables with the selected numeric variables to create the dataframe for analysis
df2=pd.concat([df[num_cols],dum1], axis=1)
df2.head()


# In[35]:


#Concatenating all the remaining dummy variables
df2=pd.concat([df2,dum2], axis=1)
df2=pd.concat([df2,dum3], axis=1)
df2=pd.concat([df2,dum4], axis=1)
df2.head()


# In[36]:


#checking the shape of new data frame
df2.shape


# In[37]:


#checking the columns of new dataframe
df2.columns


# In[38]:


# Dataset is ready
# Start of Modeling:
# Defining X and y:
X=df2.drop('cnt',axis=1)
y=df2['cnt']


# In[39]:


#Train-Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=100)


# In[40]:


X_train.head()


# In[41]:


#Getting the shape of the train and the test data
print(X_train.shape)
print(X_test.shape)


# In[42]:


#Scaling the data usinf standard scaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  #as test data is unseen data so we cannot run fit on it, only transform


# In[43]:


X_train_df=pd.DataFrame(X_train, columns=X.columns)
X_test_df=pd.DataFrame(X_test, columns=X.columns)


# In[44]:


X_train_df.head()


# In[45]:


#import RFE and Linear Regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[46]:


#Feature Selection
# Starting with 10 features
estimator=LinearRegression()
selector=RFE(estimator,n_features_to_select=10)


# In[47]:


#fitting the model for the 10 features on the data
selector = selector.fit(X_train_df, y_train)
selector.support_


# In[48]:


#Seeing the columns as per the selected features
selected_features=X_train_df.columns[selector.support_]
print(selected_features)


# In[49]:


# Now wil evaluate the model manually based on the selected features
X_train=X_train_df[selected_features]
X_test=X_test_df[selected_features]


# In[50]:


#importing stats model and adding the constant to X
import statsmodels.api as sm
X_train_sm=sm.add_constant(X_train)
X_test_sm=sm.add_constant(X_test)


# In[51]:


#Running the 1st version of the model and looking at its summary
model1=sm.OLS(np.array(y_train), X_train_sm)
res1 = model1.fit()
res1.summary()


# In[52]:


# The 1st model has F-stat less that 0.05, adj R2 is 0.59 but few variables like 'wk3' and 'wth2' have p value >0.05
# p-value should be less than 0.05
# VIF should be less than 5 
# Calculating VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data['Feature'] = X_train_sm.columns
vif_data['VIF']=[variance_inflation_factor(X_train_sm.values, i) for i in range(len(X_train_sm.columns))]
vif_data


# In[53]:


# VIF for all variables is less than 5, so no action
# Dropping variable 'wk3' which had the highest p value of 0.276
X_train_sm = X_train_sm.drop(['wk3'], axis =1)
X_test_sm = X_test_sm.drop(['wk3'], axis =1)


# In[54]:


# Developing the 2nd version of the model for best fit
model2=sm.OLS(np.array(y_train), X_train_sm)
res2 = model2.fit()
res2.summary()


# In[55]:


# The 2nd model has F-stat less that 0.05, adj R2 is 0.59 but one variable 'wth2' have p value >0.05
# p-value should be less than 0.05
# VIF should be less than 5 
# Calculating VIF again
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data['Feature'] = X_train_sm.columns
vif_data['VIF']=[variance_inflation_factor(X_train_sm.values, i) for i in range(len(X_train_sm.columns))]
vif_data


# In[56]:


# VIF for all variables is less than 5, so no action
# Dropping variable 'wth2' which had the highest p value of 0.105
X_train_sm = X_train_sm.drop(['wth2'], axis =1)
X_test_sm = X_test_sm.drop(['wth2'], axis =1)


# In[57]:


# Developing the 2nd version of the model for best fit
model3=sm.OLS(np.array(y_train), X_train_sm)
res3 = model3.fit()
res3.summary()


# In[58]:


# The 3rd model has F-stat less that 0.05, adj R2 is 0.588 and no variable has p value >0.05
# So, all p-values are less than 0.05
# VIF should be less than 5 
# Calculating VIF again
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data['Feature'] = X_train_sm.columns
vif_data['VIF']=[variance_inflation_factor(X_train_sm.values, i) for i in range(len(X_train_sm.columns))]
vif_data


# In[60]:


# All VIF values are less than 5
# All variable p values are less than 0.05
# So model 3 is the best fit model with an adjusted R-squared of 0.588 on training data set
# Linear regression of best fit is given below"
# Demand of Bike (cnt) = 4505.27 + 1359.76 * temp - 464.43 * hum - 306.69 * windspeed + 316.67 * season2 + 584.04 * season4 - 135.10 * month7 + 194.85 * month9 - 287.93 * weather3
#      where season2 means 'summer season' & season4 means 'winter season'
#      where month7 means 'July' & month 9 means 'September'
#      where weather3 means "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds"


# In[61]:


#Residual analysis of the train data
y_train_cnt = res3.predict(X_train_sm)


# In[62]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_cnt), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# In[63]:


# The error terms from the graph seems to be normally distributed around the mean of error equal to zero.
# So the residual analysis signify that the model obtained (model 3) is a valid linear regression model


# In[64]:


# Now we will use the model to make predictions on the test data set
# X_test_sm is already scaled, right columns selected and can be used in the obtained model for testing
# Making predictions
y_pred = res3.predict(X_test_sm)


# In[66]:


#Model evaluation on test data
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label
plt.show()


# In[67]:


# Predicting R-squared for the selected model on test data
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


# In[ ]:




