# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:42:59 2015

@author: Abhishek
"""
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn import  metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
    
def file_extractor(files_path, obj):
    files = os.listdir(files_path)
    files = [files_path+ '\\'+ file_name for file_name in files if obj in file_name]
    return files
    
files_train = file_extractor('TR_dst', 'TRAIN')
vec = CountVectorizer(input='filename',  encoding='ISO-8859-1', stop_words = 'english', 
                      token_pattern = '[A-Za-z]+',
                      ngram_range=(1,1))
                   
X = vec.fit_transform(files_train)
y = pd.read_csv('spam-mail.tr.label').Prediction

#clf1 = BernoulliNB(alpha=1, binarize=2)#, class_prior=[p,1-p]) #Laplace smoothing is 1
clf = MultinomialNB(alpha=1)



predictors = [clf]
#################
#Cross Validation
#################

kf = KFold(X.shape[0], 5, shuffle = True)

for clf in predictors:
    print clf
    for construct_idx, validate_idx in kf:
        X_construct, y_construct = X[construct_idx], y[construct_idx]
        X_validate, y_validate = X[validate_idx], y[validate_idx]       
        clf.fit(X_construct, y_construct)
        predictions = clf.predict(X_validate)
        predict_positive = sum(predictions)
        true_positve = sum(y_validate)
        print 'predict_positive:%d' % predict_positive
        print 'true_positve:%d' %true_positve
        print 'predict_negative:%d' % (len(y_validate)-predict_positive)
        print 'true_negative:%d' %(len(y_validate)-true_positve)
        print 'precision score', metrics.precision_score(y_validate, predictions)        
        print 'False-Positive:', sum(predictions[np.asarray(y_validate==0)])
        print 'False-Negative:', sum(predictions[np.asarray(y_validate==1)]==0)
        
        raw_input('should I proceed \n')
#
#print 'Spam ratio in y_test = %0.2f' %(1.0*sum(y_test==0)/len(y_test))
#
#print 'Bernoulli Precision Score %0.4f' % metrics.precision_score(y_test, clf.predict(X_test))
#print 'Multinomial Precision Score %0.4f' % metrics.precision_score(y_test, clf1.predict(X_test))
#print 'Dumb classifier %0.4f (Nothing is spam)' % metrics.precision_score(y_test, 
#np.ones_like(y_test))

