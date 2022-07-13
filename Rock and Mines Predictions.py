#!/usr/bin/env python
# coding: utf-8

# In[1]:


### Importing library 


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#### Importing Dataset


# In[4]:


data = pd.read_csv('sonar data.csv',header=None)


# In[5]:


### Check the first-five rows of dataset


# In[6]:


data.head()


# In[7]:


### Check the last-five rows of dataset


# In[8]:


data.tail()


# In[9]:


### Checking number of rows and columns in the dataset


# In[10]:


data.shape


# In[11]:


### Checking the info of the dataset


# In[12]:


data.info()


# In[13]:


### Statistical measures of the dataset


# In[14]:


data.describe()


# In[15]:


### checking the number of rock and mines in the dataset


# In[16]:


data[60].value_counts()


# # M = Mine
# 
# # R = Rock

# In[17]:


### Find the mean of each columns of rock and mine


# In[18]:


data.groupby(60).mean()


# In[19]:


### Spilt the data into X(feature variable) and Y(target variable)


# In[42]:


x = data.drop(columns=60,axis=1)

y = data[60]


# In[43]:


### Print x values

print(x)


# In[44]:


### Print y values

print(y)


# # Training and Test data

# In[45]:


from sklearn.model_selection import train_test_split 


# In[46]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,stratify=y,random_state=1)


# In[47]:


### Print the shape of x_train and x_test


# In[48]:


print(x.shape,x_train.shape,x_test.shape)


# In[49]:


### Print the shape of y_train and y_test


# In[50]:


print(y.shape,y_train.shape,y_test.shape)


# # Model training ---> Logistic Regression

# In[51]:


from sklearn.linear_model import LogisticRegression


# In[52]:


model = LogisticRegression()


# In[53]:


### Training the LogisticRegression model with training data


# In[54]:


model.fit(x_train,y_train)


# # Model Evalution 

# In[55]:


from sklearn.metrics import accuracy_score


# In[56]:


### Check the accuracy on trainig data


# In[57]:


x_train_pred = model.predict(x_train)


# In[58]:


training_data_accuracy = accuracy_score(x_train_pred,y_train)


# In[59]:


### Print the accuracy

training_data_accuracy


# In[60]:


### Check the accuracy on test data


# In[61]:


x_test_pred = model.predict(x_test)


# In[62]:


test_data_accuracy = accuracy_score(x_test_pred,y_test)


# In[63]:


### Print the accuracy

test_data_accuracy


# # Making a predictive System

# In[64]:


input_data = (0.02,0.0371,0.0428,0.0207,0.0954,0.0986,0.1539,0.1601,0.3109,0.2111,0.1609,0.1582,0.2238,0.0645,0.066,0.2273,0.31,0.2999,0.5078,0.4797,0.5783,0.5071,0.4328,0.555,0.6711,0.6415,0.7104,0.808,0.6791,0.3857,0.1307,0.2604,0.5121,0.7547,0.8537,0.8507,0.6692,0.6097,0.4943,0.2744,0.051,0.2834,0.2825,0.4256,0.2641,0.1386,0.1051,0.1343,0.0383,0.0324,0.0232,0.0027,0.0065,0.0159,0.0072,0.0167,0.018,0.0084,0.009,0.0032
)

### Changing the input_data into Numpy array

input_data_as_numpy_array = np.asarray(input_data)


# In[65]:


### Reshape the np array as we are predicting for one instance

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[67]:


prediction = model.predict(input_data_reshaped)

print(prediction)


if(prediction[0]=='R'):
    print('The object is a Rock')
else:
    print('The object is a Mine')


# In[ ]:




