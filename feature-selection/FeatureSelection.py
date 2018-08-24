#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 20:27:35 2018

@author: farzam
reference: Python Machine Learning by:
       Sebastian Raschka & Vahid Mirjalili
"""
from SBS import SBS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Visualization import Visualization as pdr
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

dataset=pd.read_csv('winequality-red.csv')
x=dataset.iloc[:,0:11].values
y=dataset.iloc[:,11].values

#data preprocessing
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,
                                               random_state=0)
sc=StandardScaler()
x_train_std=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#create an instance of feature selector object
knn=KNeighborsClassifier(n_neighbors=5)
fs=SBS(estimator=knn,k_features=1,test_size=0.2)
fs.fit(x_train_std,y_train)
scores=fs.scores_
subsets=fs.subsets_

#train model with all features
knn.fit(x_train_std,y_train)
y_pred=knn.predict(x_test)
accuracy1=accuracy_score(y_test,y_pred)

#train model with selected 3 features
knn2=KNeighborsClassifier(n_neighbors=5)
knn2.fit(x_train_std[:,[4,6,10]],y_train)
y_pred2=knn2.predict(x_test[:,[4,6,10]])
accuracy2=accuracy_score(y_test,y_pred2)

# visualize
plt.subplots(1,2)
plt.subplot(121)
plt.plot([len(x) for x in (fs.subsets_)], fs.scores_)
plt.show()

#RandomForest feature selection
labels=dataset.columns[1:]
forest=RandomForestClassifier(n_estimators=500,random_state=1)
forest.fit(x_train,y_train)
importance=forest.feature_importances_
indices=np.argsort(importance)[::-1]
for p in indices:
    print('%s: %s'%(p,importance[p])) 
plt.subplot(122)      
plt.bar(range(x_train.shape[1]),importance[indices])
plt.xticks(range(x_train.shape[1]),labels[indices],
           rotation=90)
plt.show()
