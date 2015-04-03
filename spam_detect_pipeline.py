# -*- coding: utf-8 -*-
"""
Created on Fri Apr 03 14:53:51 2015

@author: Abhishek
"""

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
    
#class file_loader(TransformerMixin):    
#    #Transformer to read the files relative path
#    def __init__(self):
#        pass      
#    def fit(self, X, y=None):
#        return self
#    def transform(self, X, y=None):
#        files_dir = X[0]; test_or_train = X[1]       
#        files = os.listdir(files_dir)
#        files = [files_dir+'\\'+ file_name for file_name in files if test_or_train in file_name]  
#        #print files[1:5]
#        return files
#
#
#        
#class vectorizer(TransformerMixin):
#    def __init__(self):
#        pass
#    def fit(self, file_names, y=None):
#        #print file_names[1:5]
#        self.vec = CountVectorizer(input='filename', token_pattern='[A-Za-z]+', decode_error='ignore')
#        self.vec.fit(file_names)
#        return self        
#    def transform(self, file_names, y=None):
#        return self.vec.transform(file_names)
#        
    

#y = pd.read_csv('spam-mail.tr.label').Prediction    
#multinomialnb = MultinomialNB(alpha=1.0)
#class_prior = [(x,1-x) for x in np.arange(0.1,1,0.1)]
#params = dict(class_prior=class_prior)
#clf = GridSearchCV(estimator=multinomialnb, param_grid=params, cv=3, verbose=5)
#pipe = Pipeline(steps=[
#    ('loader', file_loader()),
#    ('vec', vectorizer()), 
#    ('naivebayes_gridsearch', clf)])
#
#pipe.fit(['TR_dst', 'TRAIN'],y)
#pre = pipe.predict(['TT', 'TEST'])


#clf.fit(['TR_dst', 'TRAIN'],y)
#
#def tr_tt_split(file_dir,obj='TRAIN'):
#    f = file_loader()
#    vec = vectorizer()
#    files = f.transform([file_dir,obj]) 
#    X = vec.fit_transform(files)
#    y = pd.read_csv('spam-mail.tr.label').Prediction    
#    return train_test_split(X,y,test_size=0.2)
#    

    
#X_train, X_test, y_train, y_test = 
def file_extractor(files_path, obj):
    files = os.listdir(files_path)
    files = [files_path+ '\\'+ file_name for file_name in files if obj in file_name]
    return files
    
files_train = file_extractor('TR_dst', 'TRAIN')
##stop words notw working
##stop_words = stopwords.word(english)
#Making a count vectorizer
vec = CountVectorizer(input='filename',  decode_error='ignore',
                      ngram_range=(1,1))
#token_pattern='[A-Za-z]+'
#vec = TfidfVectorizer(input='filename',  decode_error='ignore',
#                      ngram_range=(1,1)) #token_pattern='[A-Za-z]+',
#features, predictors
                      
X = vec.fit_transform(files_train)
#response variables
y = pd.read_csv('spam-mail.tr.label').Prediction
#train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
#I am not going to use GridSearchCV because it would hide the propoerty of split from me
#training
#for p in np.arange(0.1,1,0.1):
clf = BernoulliNB(alpha=1, binarize=2)#, class_prior=[p,1-p]) #Laplace smoothing is 1
clf1 = MultinomialNB(alpha=1)
clf.fit(X_train, y_train)
clf1.fit(X_train,y_train)
#print 'Probabilities%0.3f' %p, 'score%0.3f'  %clf.score(X_test, y_test)
print 'Spam ratio in y_test = %0.2f' %(1.0*sum(y_test==0)/len(y_test))

print 'Bernoulli Precision Score %0.4f' % metrics.precision_score(y_test, clf.predict(X_test))
print 'Multinomial Precision Score %0.4f' % metrics.precision_score(y_test, clf1.predict(X_test))
print 'Dumb classifier %0.4f (Nothing is spam)' % metrics.precision_score(y_test, 
np.ones_like(y_test))

#clf.fit(X,y) 
#
##testing
#files_test_path = 'TT_dst'
#files_test = os.listdir(files_test_path)
#files_test = [files_test_path+ '\\'+ file_name for file_name in files_test if 'TEST' in file_name]
#X = vec.transform(files_test)
#tmp = clf.predict(X)
#
#def submit_csv(x):
#    #genrate a csv file in submission format from an numpy array
#    df = pd.DataFrame(data=x,columns=['Predictor'])
#    df.index = np.arange(1,len(x)+1,dtype=int)
#    df.index.name = 'ID'
#    df.to_csv(open('spam_detect.csv','w'))