#-------------------------------------------------------------------------------------------------------------
# import
import os
import pandas as pd
import numpy as np
from itertools import combinations 
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
# from xgboost import XGBClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn.feature_extraction.text import TfidfVectorizer

#-------------------------------------------------------------------------------------------------------------
# data-load

colnames = ['title','summary','bias']
data = pd.read_csv('data/KG3/knowledge_pre_new.tsv', names=colnames, sep='\t')

data = data.replace(np.nan, 'none')

#-------------------------------------------------------------------------------------------------------------
# feature-selection 
# since the number of features are less brute force is the best option

best_feats = []
best_score = 0
best_classifier = ''

classifiers = ['MultinomialNB()','LogisticRegression(max_iter=5000)','SVC()']
# classifiers = ['MultinomialNB()','LogisticRegression(max_iter=5000)','RandomForestClassifier()','AdaBoostClassifier()','SVC()','XGBClassifier()']

#title+cs
# comb = ['title','summary']

#cs
# comb = ['summary']

#title
comb = ['title']

feats = []
for j in comb: 
    tup = (j,TfidfVectorizer(min_df=0.1,max_df=0.4))
    feats.append(tup)  # combination of features

mapper = DataFrameMapper(feats) 

features = mapper.fit_transform(data)
categories = data.bias

x_tmp, x_test_tmp, y_tmp, y_test_tmp = train_test_split(features,categories,test_size=0.1, random_state = 0)

x, x_test, y, y_test = train_test_split(x_tmp,y_tmp,test_size=0.111, random_state = 0)

# selection of classifier is not that important but still

for classifier in classifiers:

    clf = eval(classifier).fit(x,y)
    predicted = clf.predict(x_test)

    score = classification_report(y_test, predicted)

    print(classifier + '  ' + str(score))