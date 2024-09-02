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
from functionsHABiC import classification



#######################################################
######################### MAIN ########################
#######################################################


#######################################################
##### load your data
to_load = '/chiaraco/HABiC/data'
data = "synthetique_sklearn"

# load train dataset, the variable to be predicted (Y) separated from the variables allowing learning (X)
data_train = pd.read_csv(f'{to_load}/{data}/train.csv',header=0,index_col=0)
X = data_train.drop('Y',axis=1)
Y = data_train['Y'].copy()

# put your validation datasets into a list and give each one a name
data_valid = pd.read_csv(f'{to_load}/{data}/valid.csv',header=0,index_col=0)
Xval = [data_valid.drop('Y',axis=1)] 
Yval = [data_valid['Y'].copy()]
Nval = ['Val']


#######################################################
##### choose your algorihtm and change its parameters if you want

# if naive.HABiC
params = {'meth':'naive.HABiC'}

# if redPCA.HABiC
params = {'meth':'redPCA.HABiC', 'DimRed':100}

# if redPLS.HABiC
params = {'meth':'redPLS.HABiC', 'DimRed':100}

# if bagSTD.HABiC
params = {'meth':'bagSTD.HABiC', 'NbTrees':50}

# if bagRF.HABiC
params = {'meth':'bagRF.HABiC', 'NbTrees':50, 'NbVarImp':3}

# if bagPLS.HABiC
params = {'meth':'bagPLS.HABiC', 'NbTrees':50, 'NbVarImp':3}

# if Wass-NN
params = {'meth':'Wass-NN', 'struct':{'hidden_layer_sizes':(300,300,300), \
          'activation':'relu', 'solver':'adam', 'batch_size':64, \
          'learning_rate_init':0.0001, 'max_iter':10, 'lambd':10}}


#######################################################
##### its performances

# you can choose as metric :
## - 'MCC' (Matthews Correlation Coefficient)
## - 'ACC' (accuracy score)

perf = classification(X, Y, Xval, Yval, Nval, param=params, metr='MCC')
print(perf)



















