#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pickle


# In[2]:


df = pd.read_csv('crop_recommendation.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.size


# In[8]:


df.dtypes


# In[9]:


df['label'].value_counts()


# In[10]:


sns.heatmap(df.corr(),annot = True)


# In[11]:


features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target= df['label']
labels = df['label']


# In[12]:


#creating empty list to append all model and their name
acc = []
model = []


# In[13]:


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(features,target,test_size = 0.2, random_state = 2)


# # Decision Tree

# In[14]:


DecisionTree = DecisionTreeClassifier(criterion = 'entropy', random_state = 2, max_depth = 5)
DecisionTree.fit(X_train,Y_train)
predicted_value = DecisionTree.predict(X_test)
x = metrics.accuracy_score(Y_test,predicted_value)
acc.append(x)
model.append('Decision Tree')
print("DecisionTree's accuracy is :", x * 100)
print(classification_report(Y_test,predicted_value))


# In[15]:


score = cross_val_score(DecisionTree, features, target, cv=5)
score



# In[17]:


NaiveBayes = GaussianNB()

NaiveBayes.fit(X_train,Y_train)

predicted_value = NaiveBayes.predict(X_test)
x = metrics.accuracy_score(Y_test,predicted_value)

acc.append(x)
model.append('Naive Bayes')

print("Naive Bayes's accuracy is :",x*100)
print(classification_report(Y_test,predicted_value))


# In[18]:


score = cross_val_score(NaiveBayes, features, target,cv = 5)
score




# In[20]:


LogReg = LogisticRegression()
LogReg.fit(X_train,Y_train)
predicted_value = LogReg.predict(X_test)
x = metrics.accuracy_score(Y_test,predicted_value)
acc.append(x)
model.append("LogisticRegression")
print("LogisticRegression's accuracy is: ", x*100)
print(classification_report(Y_test,predicted_value))


# In[21]:


score = cross_val_score(LogReg, features, target, cv = 5)
score




# In[25]:


RF = RandomForestClassifier(n_estimators = 20, random_state = 0)
RF.fit(X_train,Y_train)
predicted_value = RF.predict(X_test)
x = metrics.accuracy_score(Y_test,predicted_value)
acc.append(x)
model.append('Random Forest')
print("Random Forest's accuracy is: ",x*100)
print(classification_report(Y_test,predicted_value))


# In[26]:


score = cross_val_score(RF, features, target, cv = 5)
score




# In[29]:


import xgboost as xgb
XB = xgb.XGBClassifier()
XB.fit(X_train,Y_train)
predicted_value = XB.predict(X_test)
x = metrics.accuracy_score(Y_test,predicted_value)
acc.append(x)
model.append('XGBoost')
print("XGBoost's accuracy is: ",x*100)
print(classification_report(Y_test,predicted_value))


# In[30]:


score = cross_val_score(RF, features, target, cv = 5)
score




# # Accuracy Comparison

# In[39]:


plt.figure(figsize = [10,5],dpi = 100)
plt.title('Accuracy Comparison')
plt.xlabel('Accuracy')
plt.ylabel('Algorithm')
sns.barplot(x = acc,y = model,palette='dark')


# In[44]:


accuracy_models = dict(zip(model, acc))
for k, v in accuracy_models.items():
    print (k, '-->', v)


# In[45]:


data = np.array([[19,78,16,20.6537,23.1053,5.9675,67.7176]])
prediction = RF.predict(data)
print(prediction)


# In[48]:


data = np.array([[36,59,46,34.28,93.61,6.72,127.25]])
prediction = RF.predict(data)
print(prediction)

# RF_pkl_filename = ('Random_forest.pkl')
# RF_Model_pkl = open(RF_pkl_filename,'wb')
# pickle.dump(RF,RF_Model_pkl)
# RF_Model_pkl.close()
pickle.dump(RF, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

print(model.predict([[19,78,16,20.6537,23.1053,5.9675,67.7176]]))
