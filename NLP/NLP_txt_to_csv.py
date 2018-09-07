#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 15:57:56 2018

@author: farzam
reference: Python Machine Learning by:
       Sebastian Raschka & Vahid Mirjalili
"""
#importing libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

#importing dataset and split to test and train
dataset=pd.read_csv('movie-data.csv').iloc[:,0:2]
x_train=dataset.loc[:2500, 'review'].values
x_test=dataset.loc[2500:5000,'review'].values
y_train=dataset.loc[:2500,'sentiment'].values
y_test=dataset.loc[2500:5000,'sentiment'].values
 
#cleaning the text using regex      
import re
def preprocessor(text):     
       
    text=re.sub('<[^>]*>','',text)    
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)  
    text=(re.sub('[\W]+',' ',text.lower()+
                 ''.join(emoticons).replace('-','')))    
    return text

dataset['review']=dataset['review'].apply(preprocessor) 

#splitting text to words
def tokenizer(text):
    return text.split() 
      
from nltk.stem.porter import PorterStemmer
def tokenizer_porter(text):
    porter=PorterStemmer()   
    return [porter.stem(word) for word in tokenizer(text)]


#removing stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop=stopwords.words('english')

def remove_stopwords(text):
    return [word for word in text if word not in stop]   

 
#training model
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

tfidf=TfidfVectorizer(strip_accents=None,preprocessor=None,
                      lowercase=False)
param_grid=[{'vect__ngram_range':[(1,1)],'vect__stop_words':[stop,None],
                 'vect__tokenizer':[tokenizer,tokenizer_porter],
       'clf__penalty':['l1','l2'],'clf__C':[1.0,10.0,100.0]},
        {'vect__ngram_range':[(1,1)],'vect__stop_words':[stop,None],
                 'vect__tokenizer':[tokenizer,tokenizer_porter],
                 'vect__use_idf':[False],'vect__norm':[None],
       'clf__penalty':['l1','l2'],'clf__C':[1.0,10.0,100.0],
       
        }]
lr_tfidf=Pipeline([('vect',tfidf),
        ('clf',LogisticRegression(random_state=0))])
gs_lr_tfidf=GridSearchCV(lr_tfidf,param_grid,scoring='accuracy',
                         cv=5,n_jobs=-1,verbose=1)

gs_lr_tfidf.fit(x_train,y_train)
params=gs_lr_tfidf.best_params_
score=gs_lr_tfidf.best_score_
estimator=gs_lr_tfidf.best_estimator_     
accuracy=accuracy_score(y_test,estimator.predict(x_test))
