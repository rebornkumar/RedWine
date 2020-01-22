
# coding: utf-8

# # redWine

# In[52]:


import numpy as np
import pandas as pd


# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


from sklearn import preprocessing


# In[55]:


from sklearn.ensemble import RandomForestRegressor


# In[56]:


from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


# In[57]:


from sklearn.metrics import mean_squared_error, r2_score


# In[58]:


from sklearn.externals import joblib


# In[59]:


data = pd.read_csv("winequality-red.csv")


# In[60]:


print data.head()


# In[61]:


data = pd.read_csv("winequality-red.csv",sep=';')


# In[62]:


data


# In[63]:


print data.head()


# In[64]:


data.describe()


# In[65]:


print data.shape


# In[84]:


Y=data.quality
X=data.drop('quality',axis=1)


# In[85]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123,stratify=Y)


# In[86]:


scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled=scaler.transform(X_train)


# In[87]:


print X_train_scaled.mean(axis=0)
print "\n"
print X_train_scaled.std(axis=0)


# In[88]:


X_test_scaled = scaler.transform(X_test)


# In[89]:


print X_test_scaled.mean(axis=0)
print "\n"
print X_test_scaled.std(axis=0)


# In[90]:


pipeline = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=100))


# In[93]:


hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}


# In[95]:


clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(X_train, Y_train)


# In[96]:


print clf.best_params_


# In[97]:


print clf.refit


# In[98]:


Y_pred = clf.predict(X_test)


# In[99]:


print r2_score(Y_test, Y_pred)


# In[100]:


print mean_squared_error(Y_test, Y_pred)


# In[101]:


joblib.dump(clf, 'rf_regressor.pkl')


# In[102]:


clf2 = joblib.load('rf_regressor.pkl')


# In[103]:


clf2.predict(X_test)

