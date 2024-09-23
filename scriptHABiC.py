import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import matthews_corrcoef as MCC, accuracy_score as ACC
from tqdm import tqdm
import time
import os
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
from torch import FloatTensor, cuda, nn, mean, rand, ones
from torch.optim import LBFGS, SGD, Adam
from torch.autograd import Variable, grad

# GPU can be used to accelerate Wasserstein Neural Network fitting
if cuda.is_available(): dev = "cuda:0"
else: dev = "cpu"

# import function required for classification
from functionsHABiC import classification, performances



#######################################################
######################### MAIN ########################
#######################################################


#######################################################
##### load your data
to_load = 'data'

# load train dataset, the variable to be predicted (Y) separated from the variables allowing learning (X)
data_train = pd.read_csv(f'{to_load}/train.csv',header=0,index_col=0)
X = data_train.drop('Y',axis=1)
Y = data_train['Y'].copy()

# put your validation datasets into a list and give each one a name
data_valid = pd.read_csv(f'{to_load}/valid.csv',header=0,index_col=0)
Xval = [data_valid.drop('Y',axis=1)] 
Yval = [data_valid['Y'].copy()]
Nval = ['Val']


#######################################################
##### test all algorithms (you can change parameters)


# if naive.HABiC
params_naive = {'meth':'naive.HABiC'}

# if redPCA.HABiC
params_redPCA = {'meth':'redPCA.HABiC', 'DimRed':100}

# if redPLS.HABiC
params_redPLS = {'meth':'redPLS.HABiC', 'DimRed':100}

# if bagSTD.HABiC
params_bagSTD = {'meth':'bagSTD.HABiC', 'NbTrees':50}

# if bagRF.HABiC
params_bagRF = {'meth':'bagRF.HABiC', 'NbTrees':50, 'NbVarImp':3}

# if bagPLS.HABiC
params_bagPLS = {'meth':'bagPLS.HABiC', 'NbTrees':50, 'NbVarImp':3}

# if Wass-NN
params_WassNN = {'meth':'Wass-NN', 'struct':{'hidden_layer_sizes':(300,300,300), \
          'activation':'relu', 'solver':'adam', 'batch_size':64, \
          'learning_rate_init':0.0001, 'max_iter':10, 'lambd':10}}



#######################################################
##### performances

# choose a metric between these :
## - 'MCC' (Matthews Correlation Coefficient)
## - 'ACC' (Accuracy score)
## - 'AUC' (Area Under the Curve)
metr = 'MCC'

# choose the number of splits fro cross-validation
nb_CV = 3

# list of parameters of all algorithms to test
params = ['params_naive', 'params_redPCA', 'params_redPLS', 'params_bagSTD', \
           'params_bagRF', 'params_bagPLS', 'params_WassNN']

# create en empty dataframe to save the results there
results = pd.DataFrame(index=pd.MultiIndex.from_product([params,[f'CV{cv}' for cv in range(1,nb_CV+1)]]),columns=['Train','Test']+Nval)

# create splits for cross-validation
sss = StratifiedShuffleSplit(n_splits=nb_CV, test_size=0.3, random_state=0)
for fold, (train_index, test_index) in enumerate(sss.split(X,Y)):
    print('\nFOLD',fold+1)
    # for each split, create train/test dataset 
    xtrain_cv, xtest_cv = X.iloc[train_index,:], X.iloc[test_index,:]
    ytrain_cv, ytest_cv = Y.iloc[train_index], Y.iloc[test_index]

    # then, run all classifiers and assess their prediction performance
    for param in params :
        print('--->',param)
        pred = classification(xtrain_cv, ytrain_cv, [xtest_cv]+Xval, [ytest_cv]+Yval, ['Test']+Nval, param=eval(param))
        for y_true,samp in zip([ytrain_cv,ytest_cv]+Yval,['Train','Test']+Nval):
            perf = performances(y_true,pred[samp], metr=metr)
            results.loc[(param,f'CV{fold+1}'),samp] = perf

# print results
print('\nMean',results.groupby(level=0).mean(),'\nStd',results.groupby(level=0).std(),'\n',sep='\n')













