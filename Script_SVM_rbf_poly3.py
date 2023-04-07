# -*- coding: utf-8 -*-
"""
@author: mohammed.hammouti@igag.cnr.it

This script is mainly based on Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
"""


#Import libraries, e.g., sklearn, pandas, numpy, scipy.

import pandas as pd
import numpy as np
from scipy import io
import sklearn
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import from scipy import io
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

###### Set input data ######

#Directory path to load RO-AB profiles
rocs='\\path\\...'
 
#Directory path to load RO-VC profiles
rocl='\\path\\...'

#Total number of RO profiles
n_pr='number' 

#Directory path to save results
directory='\\path\\...' 
 
###########################

#Load data
x_cl1 = io.loadmat(rocl)
x_cl2=x_cl1[pr_VC] 
x_cl=x_cl2.T

x_no1 = io.loadmat(rocs)
x_no2=x_no1[pr_CS]
x_no=x_no2.T

#Labeling the RO-AB profiles with 0 and the RO-VC profiles with 1
y_cl_lab=np.ones(n_pr,dtype=int) #change
y_no_lab=np.zeros(n_pr,dtype=int) #change

X=np.append(x_cl, x_no, axis=0) 
y=np.append(y_cl_lab, y_no_lab, axis=0) 


#Split data in train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


### Choose a kernel to use in SVM:
''' 
#SVM with RBF Kernel
SVM_rbf = svm.SVC(kernel='rbf',random_state=1,cache_size=9000)
#Fit the model with training data
modelsvm = SVM_rbf.fit(X_train, y_train)
'''

#OR

'''
#SVM with 3rd degree polynomial kernel
SVM_poly3 = svm.SVC(kernel='poly',degree=3,random_state=1,cache_size=9000)
#Fit the model with training data
modelsvm = SVM_poly3.fit(X_train, y_train)
'''
### 

#Compute accuracy on testing data
score_test=modelsvm.score(X_test, y_test)
print('acc_test set',score_test)

#Compute accuracy on training data
score_train=modelsvm.score(X_train, y_train)
print('acc_train set',score_train)
 
#Compute confusion matrix and classification report from test subset
 
y_pred4 = modelsvm.predict(X_test)
cm = confusion_matrix(y_test, y_pred4)
print('confusion_matrix_test',cm)
 
classification_rep4=classification_report(y_test, y_pred4)
print('classification_report_test',classification_rep4)
 
#Compute confusion matrix from train subset
 
x_pred4 = modelsvm.predict(X_train)
cm_train = confusion_matrix(y_train, x_pred4)
print('confusion_matrix_train',cm_train)


#Confusion matrix parameters resulting from test subset
tn=cm[0,0]
tp=cm[1,1]
fp=cm[0,1]
fn=cm[1,0]

#Confusion matrix parameters resulting from train subset
tn1=cm_train[0,0]
tp1=cm_train[1,1]
fp1=cm_train[0,1]
fn1=cm_train[1,0]

#Save some results (e.g. to an Excel file)
arr_rbf=[score_test,tp,tn,fp,fn,score_train,tp1,tn1,fp1,fn1]
data=[arr_rbf] 
df = pd.DataFrame(data)
df.to_excel(directory+'name_file.xlsx')

