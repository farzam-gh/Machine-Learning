#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 19:10:39 2018

@author: farzam
reference: Python Machine Learning by:
       Sebastian Raschka & Vahid Mirjalili
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from itertools import combinations

class SBS(object):
    def __init__(self,estimator,k_features,test_size=0.2,
                 scoring=accuracy_score,random_state=0):
        self.estimator=estimator
        self.k_features=k_features
        self.test_size=test_size
        self.random_state=random_state
        self.scoring=scoring
       
        
    def fit(self,x,y):           
        x_train,x_test,y_train,y_test=train_test_split(x,y,
           test_size=self.test_size,random_state=self.random_state)
        dim=x.shape[1]
        print(dim)
        self.indices_=tuple(range(dim))
        self.subsets_=[self.indices_]
        score=self.calc_score(x_train,x_test,y_train,
                         y_test,self.indices_)
        self.scores_=[score]
        while(dim >self.k_features):
            scores=[]
            subsets=[]
            for p in combinations(self.indices_,r=dim-1):
                score=self.calc_score(x_train,x_test,y_train,
                                      y_test,p)       
                scores.append(score)
                subsets.append(p)
            best=np.argmax(scores) 
            self.indices_=subsets[best]
            self.subsets_.append(self.indices_)
            dim-=1
            self.scores_.append(scores[best])
        self.k_score=self.scores_[-1]
        return self


           
    def calc_score(self,x_train,x_test,y_train,y_test,indices):
        self.estimator.fit(x_train[:,indices],y_train) 
        y_pred=self.estimator.predict(x_test[:,indices])
        score=self.scoring(y_test,y_pred)
        return score   
           
           
           
