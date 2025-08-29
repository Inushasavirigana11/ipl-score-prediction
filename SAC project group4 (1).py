#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:


ipl = pd.read_csv(r'C:/Users/savir/OneDrive/Desktop/ipl1.csv')
ipl.head()


# In[3]:


data = pd.read_csv('C:/Users/savir/OneDrive/Desktop/ipl2.csv')
data.head()


# In[4]:


ipl= ipl.drop(['Unnamed: 0','extras','match_id', 'runs_off_bat'],axis = 1)
new_ipl = pd.merge(ipl,data,left_on='striker',right_on='Player',how='left')
new_ipl.drop(['wicket_type', 'player_dismissed'],axis=1,inplace=True)
new_ipl.columns


# In[5]:


print(ipl)


# In[6]:


print(new_ipl)


# In[7]:


str_cols = new_ipl.columns[new_ipl.dtypes==object]
new_ipl[str_cols] = new_ipl[str_cols].fillna('.')


# In[8]:


import pandas as pd


# In[9]:


listf = []

for c in new_ipl.columns:
    if new_ipl[c].dtype == object:
        print(c, "->", new_ipl[c].dtype)
        listf.append(c)


# In[10]:


a1 = new_ipl['venue'].unique()
a2 = new_ipl['batting_team'].unique()
a3 = new_ipl['bowling_team'].unique()
a4 = new_ipl['striker'].unique()
a5 = new_ipl['bowler'].unique()

def labelEncoding(data):
	dataset = pd.DataFrame(new_ipl)
	feature_dict ={}
	
	for feature in dataset:
		if dataset[feature].dtype==object:
			le = preprocessing.LabelEncoder()
			fs = dataset[feature].unique()
			le.fit(fs)
			dataset[feature] = le.transform(dataset[feature])
			feature_dict[feature] = le
			
	return dataset

labelEncoding(new_ipl)


# In[11]:


ip_dataset = new_ipl[['venue','innings', 'batting_team',
					'bowling_team', 'striker', 'non_striker',
					'bowler']]

b1 = ip_dataset['venue'].unique()
b2 = ip_dataset['batting_team'].unique()
b3 = ip_dataset['bowling_team'].unique()
b4 = ip_dataset['striker'].unique()
b5 = ip_dataset['bowler'].unique()
new_ipl.fillna(0,inplace=True)

features={}

for i in range(len(a1)):
	features[a1[i]]=b1[i]
for i in range(len(a2)):
	features[a2[i]]=b2[i]
for i in range(len(a3)):
	features[a3[i]]=b3[i]
for i in range(len(a4)):
	features[a4[i]]=b4[i]
for i in range(len(a5)):
	features[a5[i]]=b5[i]
	
features


# In[12]:


X = new_ipl[['venue', 'innings','batting_team',
			'bowling_team', 'striker','bowler']].values
y = new_ipl['y'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=1)
print(y)


# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[14]:


#pip install tensorflow


# In[15]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping


# In[16]:


model = Sequential()
  
model.add(Dense(43, activation='relu'))
model.add(Dropout(0.5))
  
model.add(Dense(22, activation='relu'))
model.add(Dropout(0.5))
  
model.add(Dense(11, activation='relu'))
model.add(Dropout(0.5))
  
model.add(Dense(1))
  
model.compile(optimizer='adam', loss='mse')


# In[17]:


early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(x=X_train, y=y_train, epochs=400, 
          validation_data=(X_test,y_test),
          callbacks=[early_stop] )


# In[18]:


model_losses = pd.DataFrame(model.history.history)
model_losses.plot()


# In[19]:


predictions = model.predict(X_test)
sample = pd.DataFrame(predictions,columns=['Predict'])
sample['Actual']=y_test
sample.head(10)


# In[20]:


from sklearn.metrics import mean_absolute_error,mean_squared_error

mean_absolute_error(y_test,predictions)


# In[21]:


error=np.sqrt(mean_squared_error(y_test,predictions))
#print(error)


# In[22]:


from sklearn.metrics import mean_squared_error
import numpy as np

# Assuming you have imported y_test and predictions
mse = mean_squared_error(y_test, predictions)

# Calculate the total sum of squares (TSS), which represents the total variance in the true target values (y_test):
mean_y_test = np.mean(y_test)
tss = np.sum((y_test - mean_y_test) ** 2)

# Calculate R-squared
r_squared = 1 - (mse / tss)

#print("Mean Squared Error:", mse)
print("Accuracy-Score:", r_squared*100 ,"%")


# In[ ]:




