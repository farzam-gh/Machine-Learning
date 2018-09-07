#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 20:38:14 2018

@author: farzam
reference: Python Machine Learning by:
       Sebastian Raschka & Vahid Mirjalili
"""
from nltk.corpus import stopwords
import numpy as np
import re
stops=stopwords.words('english')
from nltk import PorterStemmer

#define tokenizer
def tokenizer(text):
    text=re.sub('<[*^>]','',text)  
    emoticons=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text=re.sub('[\W]+',' ',text.lower())+ \
    ''.join(emoticons).replace('-','')
    tokenized=[w for w in text.split() if w not in stops]           
    return tokenized

#define stream reader which read line by line
def stream_docs(path):
    with open (path,'r',encoding='utf-8') as csv:
        next(csv)
        for line in csv:              
            text,label=line[:-3],int(line[-2])            
            yield text,label

#define function for reading minibatches
def get_minibatch(doc_stream,size):     
    text,label=[],[]  
    for i in range(size):
        try:   
            t,l=next(doc_stream)            
            text.append(t)
            label.append(l)
        except StopIteration:
            break   
    return text,label


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

#converting texts to occurance matrix
vect=HashingVectorizer(n_features=2**21,decode_error='ignore',
                       preprocessor=None,
                       tokenizer=tokenizer)
clf=SGDClassifier(loss='log',random_state=1,n_iter=1) 
doc_stream=stream_docs('movie-data.csv')  


import pyprind
pbar=pyprind.ProgBar(45)     
classes=np.array([0,1])
for _ in range(45):
    x_train,y_train=get_minibatch(doc_stream,1000)
    if not x_train:
       break    
    x_train=vect.transform(x_train)
    clf.partial_fit(x_train,y_train,classes=classes)
    pbar.update()
x_test,y_test=get_minibatch(doc_stream,5000)
x_test=vect.transform(x_test)
print('accuracy: ',clf.score(x_test,y_test))    
y_pred=clf.predict_proba(
   vect.transform(np.array(['it was too long'])))    
    
    
    
    
