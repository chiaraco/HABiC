import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import matthews_corrcoef as MCC, accuracy_score as ACC
from sklearn.metrics import roc_curve, auc
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


if cuda.is_available(): dev = "cuda:0"
else: dev = "cpu"



#######################################################
###################### FUNCTIONS ######################
#######################################################

def classification(X,Y,Xval=[],Yval=[],Nval=[],param={'meth':'naive.HABiC'},mult=10**2):

    pred = {}
    
    # resampling for balanced data
    X,Y = balancedData(X,Y)

    ## naive HABiC
    if param['meth'] == 'naive.HABiC' :

        scores = naiveHABiC(X,Y,Xval,Yval,Nval,mult)

        threshold = scores[0].mean()

        # prediction performances
        for y,n,sc in zip([Y]+Yval,['Train']+Nval,scores):
            pred[n] = predictions(y,sc,threshold)
            

    ## dimensionality reduction + HABiC
    # PCA
    elif param['meth'] == 'redPCA.HABiC' :

        # dimensionality reduction by PCA
        X_red,Xval_red = reduction(X,Y,Xval,'PCA',param['DimRed'])

        scores = naiveHABiC(X_red,Y,Xval_red,Yval,Nval,mult)

        threshold = scores[0].mean()

        # prediction performances
        for y,n,sc in zip([Y]+Yval,['Train']+Nval,scores):
            pred[n] = predictions(y,sc,threshold)


    # PLS
    elif param['meth'] == 'redPLS.HABiC' :

        # dimensionality reduction by PLS-DA
        X_red,Xval_red = reduction(X,Y,Xval,'PLS',param['DimRed'])

        scores = naiveHABiC(X_red,Y,Xval_red,Yval,Nval,mult)

        threshold = scores[0].mean()

        # prediction performances
        for y,n,sc in zip([Y]+Yval,['Train']+Nval,scores):
            pred[n] = predictions(y,sc,threshold)


    ## bagging + HABiC
    # Standard bagging
    elif param['meth'] == 'bagSTD.HABiC' :

        # standard bagging
        scores = bagging(X,Y,Xval,Yval,Nval,'STD',param['NbTrees'],mult=mult)

        threshold = 0.5

        # prediction performances
        for y,n,sc in zip([Y]+Yval,['Train']+Nval,scores):
            pred[n] = predictions(y,sc,threshold)
 

    # Random Forest bagging
    elif param['meth'] == 'bagRF.HABiC' :

        # standard bagging
        scores = bagging(X,Y,Xval,Yval,Nval,'RF',param['NbTrees'],param['NbVarImp'],mult=mult)

        threshold = 0.5

        # prediction performances
        for y,n,sc in zip([Y]+Yval,['Train']+Nval,scores):
            pred[n] = predictions(y,sc,threshold)


    # PLS-DA bagging
    elif param['meth'] == 'bagPLS.HABiC' :

        # standard bagging
        scores = bagging(X,Y,Xval,Yval,Nval,'PLS',param['NbTrees'],param['NbVarImp'],mult=mult)

        threshold = 0.5

        # prediction performances
        for y,n,sc in zip([Y]+Yval,['Train']+Nval,scores):
            pred[n] = predictions(y,sc,threshold)


    ## Deep Learning method
    elif param['meth'] == 'Wass-NN' :

        # Wasserstein Neural Network
        clf = WassNNClassifier(v=X.shape[1],**param['struct']).to(dev)
        clf.fit(X,Y)

        # prediction performances
        for x,y,n in zip([X]+Xval,[Y]+Yval,['Train']+Nval):
            sc = clf.score_prediction(x)
            if n == 'Train' : threshold = sc.mean()
            pred[n] = predictions(y,sc,threshold)


    return perf


def balancedData(X,Y):
    if Y.value_counts()[0] != Y.value_counts()[1]:
        class_sup = Y.value_counts().index[np.argmax(Y.value_counts())]
        to_drop = Y[Y == class_sup].sample(abs(Y.value_counts()[0] - Y.value_counts()[1])).index
        X.drop(to_drop, inplace=True)
        Y.drop(to_drop, inplace=True)
    return X,Y

def CostMatrix(X0,X1):
    return manhattan_distances(X0,X1)

class HungarianAlgo: #### inspired by https://gist.github.com/KartikTalwar/3158534

    def __init__(self, weights):
        self.weights = weights
        self.dim = len(self.weights)
        self.U = range(self.dim)
        self.V = range(self.dim)
        self.phi = np.array([min([self.weights[u][v] for v in self.V]) for u in self.U])                    
        self.psi = np.array([min([self.weights[u][v] - self.phi[u] for u in self.U]) for v in self.V])
        self.Mu = {}                         
        self.Mv = {}
        self.minSlack = None

    def __improveLabels(self, val, S, T):

        for u in S:
            self.phi[u] += val
        for v in self.V:
            if v in T:
                self.psi[v] -= val
            else:
                self.minSlack[v][0] -= val

    def __improveMatching(self, v, T):

        u = T[v]
        if u in self.Mu:
            self.__improveMatching(self.Mu[u], T)
        self.Mu[u] = v
        self.Mv[v] = u

    def __slack(self, u, v): 
        return self.weights[u][v] - self.phi[u] - self.psi[v]

    def __augment(self, S, T):

        while True:
            # select edge (u,v) with u in S, v not in T and min slack
            ((val, u), v) = min([(self.minSlack[v], v) for v in self.V if v not in T])
            assert u in S
            if val > 0:        
                self.__improveLabels(val, S, T)
            # now we are sure that (u,v) is saturated
            assert self.__slack(u, v) == 0
            T[v] = u
            if v in self.Mv:
                u1 = self.Mv[v]
                assert not u1 in S
                S[u1] = True
                for v in self.V:
                    if not v in T and self.minSlack[v][0] > self.__slack(u1, v):
                        self.minSlack[v] = [self.__slack(u1 ,v), u1]
            else:
                self.__improveMatching(v, T)
                return

    def minWeightMatching(self, init_M = True):

        if init_M:
            self.init_assign()
        while len(self.Mu) < self.dim:
            free = [u for u in self.V if u not in self.Mu]
            u0 = free[0]
            S = {u0: True}
            T = {}
            self.minSlack = [[self.__slack(u0, v), u0] for v in self.V]
            self.__augment(S, T)
        val = sum(self.phi) + sum(self.psi)
        return (self.Mu, self.Mv, val)

    def getPhiPsi(self):

        return (self.phi, self.psi)

    def __col_already_assigned(self, mat, col):

        for i in range(self.dim):
            if mat[i][col] == 1: return True
        return False

    def init_assign(self):

        assign_mat = [[0 for i in range(self.dim)] for j in range(self.dim)]
        for i in range(self.dim):
            for j in range(self.dim):
                if (not self.__slack(i, j) and not self.__col_already_assigned(assign_mat, j)):
                    assign_mat[i][j] = 1
                    self.Mu[i] = j
                    self.Mv[j] = i
                    break
        return assign_mat




def f_train(phi,psi,Y):
    f_X = Y.copy()
    i,j=0,0
    for nb in range(len(f_X)):
        if f_X.iloc[nb] == 0 :
            f_X.iloc[nb]=-phi[i]
            i+=1
        else :
            f_X.iloc[nb]=psi[j]
            j+=1
    return f_X


def f_test(Xtest,X,Y,phi,psi):
    Cval = CostMatrix(Xtest,X[Y==1])
    Cval2 = CostMatrix(Xtest,X[Y==0])
    f_Xval = pd.Series(index=Xtest.index,dtype='float64')
    f_Xval2 = pd.Series(index=Xtest.index,dtype='float64')
    for i in range(len(f_Xval)):
        f_Xval.iloc[i]=max(psi-Cval[i])
        f_Xval2.iloc[i]=-max(phi-Cval2[i])

    ff=(f_Xval+f_Xval2)/2

    return ff


def naiveHABiC(X,Y,Xval=[],Yval=[],Nval=[],mult=10**2):

    # cost matrix calculation
    C = CostMatrix(X[Y==0],X[Y==1])

    # Hungarian Algorithm
    HA = HungarianAlgo((C*mult).astype(int))
    HA.minWeightMatching()
    phi_int,psi_int = HA.getPhiPsi()
    phi,psi=phi_int/mult,psi_int/mult

    # scores
    scores=[]
    scores.append(f_train(phi,psi,Y))
    scores.extend([f_test(val,X,Y,phi,psi) for val in Xval])

    return scores


def predictions(y,sc,threshold):
    cond = sc>=threshold
    sc[cond]=1
    sc[~cond]=0
    return sc

def performances(y,sc,metr):
    if metr=='MCC' : p = MCC(y,sc)
    elif metr=='ACC': p = ACC(y,sc)
    elif metr=='AUC' : 
        fpr, tpr, thresholds = roc_curve(y,sc)
        p = auc(fpr,tpr)
    return p

def reduction(X,Y,Xval,RedMeth,DimRed):
    if RedMeth == 'PLS' : transfo = PLSRegression(n_components=DimRed, scale=False)
    elif RedMeth == 'PCA': transfo = PCA(n_components=DimRed)

    transfo.fit(X,Y)
    X_red = pd.DataFrame(transfo.transform(X),index=X.index)
    Xval_red = []
    for i,val in enumerate(Xval):
        Xval_red.append(pd.DataFrame(transfo.transform(val),index=val.index))

    return X_red,Xval_red



def bagging(X,Y,Xval,Yval,Nval,BagMeth,NbTrees,NbVarImp=None,mult=10**2):
    df_scores = [pd.DataFrame(index=data.index,columns=['Tree'+str(i+1) for i in range(NbTrees)]) for data in [X]+Xval]
    nb_p = int(X.shape[1]**(1/2))
    pct_obs = 50

    if BagMeth == 'RF':
        param_grid = {'max_depth' : [2,5,10,None], 'criterion' :['gini', 'entropy'], 'min_samples_leaf': [2,5]}
        grf = GridSearchCV(RFC(), param_grid, cv=3)
        grf.fit(X, Y)

    var_imp_total = {}
    for var in X.columns: var_imp_total[var] = 0

    for T in tqdm(range(NbTrees)):

        if BagMeth in ['RF','PLS']:
            x1 = X.sample(nb_p,axis=1)
            x = x1.groupby(Y).sample(x1.shape[0]*pct_obs//100,replace=True)
            y = Y[x.index]

            if BagMeth == 'RF':
                dt = DTC(max_depth=grf.best_params_['max_depth'],criterion=grf.best_params_['criterion'],min_samples_leaf=grf.best_params_['min_samples_leaf'])
                dt.fit(x,y)
                imp = pd.DataFrame(data=dt.feature_importances_,index=x.columns)

            elif BagMeth == 'PLS':
                pls = PLSRegression(scale=False)
                pls.fit(x, y)
                imp = pd.DataFrame(data=abs(pls.coef_).reshape(-1),index=x.columns)

            var_imp = imp.sort_values(by=0,ascending=False).index[:NbVarImp].to_list()

        elif BagMeth == 'STD':
            var_imp = X.sample(nb_p,axis=1).columns.to_list()

        x1b = X[var_imp]
        xb = x1b.groupby(Y).sample(x1b.shape[0]*pct_obs//100,replace=True)
        yb = Y[xb.index]

        scores = naiveHABiC(xb,yb,[data[var_imp] for data in [X]+Xval],[Y]+Yval,['Train']+Nval,mult)

        threshold = scores[0].mean()

        for i,(x,y,n,sc) in enumerate(zip([X]+Xval,[Y]+Yval,['Train']+Nval,scores[1:])):
            cond = sc>=threshold
            sc[cond]=1
            sc[~cond]=0

            df_scores[i]['Tree'+str(T+1)].loc[x.index] = sc

    return [sc.mean(axis=1) for sc in df_scores]


class WassNNClassifier(nn.Module, BaseEstimator, ClassifierMixin):

    def __init__(self,v,hidden_layer_sizes=(300,300,300), activation='relu', solver='adam', batch_size=64, learning_rate_init=0.0001, max_iter=1000, lambd=10):
        super(WassNNClassifier,self).__init__()

        self.v=v
        self.hidden_layer_sizes=hidden_layer_sizes
        self.activation=activation
        self.solver=solver
        self.batch_size=batch_size
        self.learning_rate_init=learning_rate_init
        self.max_iter=max_iter
        self.lambd=lambd

        if self.activation == 'relu' : act = nn.ReLU(inplace=True)
        elif self.activation == 'logistic' : act = nn.Sigmoid()
        elif self.activation == 'tanh' : act = nn.Tanh()

        if len(self.hidden_layer_sizes)!=0 :
            layers = [nn.Linear(self.v,self.hidden_layer_sizes[0])]
            layers.append(act)

            for i in range(1,len(self.hidden_layer_sizes)):
                layers += [nn.Linear(self.hidden_layer_sizes[i-1],self.hidden_layer_sizes[i])]
                layers.append(act)

            layers += [nn.Linear(self.hidden_layer_sizes[-1],1)]

        else :
            layers = [nn.Linear(self.v,1)]

        self.reseau = nn.Sequential(*layers).to(dev)

        if self.solver == 'adam': self.optimizer = Adam(self.parameters(), lr=self.learning_rate_init)
        elif self.solver == 'sgd': self.optimizer = SGD(self.parameters(), lr=self.learning_rate_init)
        elif self.solver == 'lbfgs': self.optimizer = LBFGS(self.parameters(), lr=self.learning_rate_init)

        self.grad_norm=[]
        self.score0 = []
        self.score1 = []
        self.crit = [] # Wasserstein Distance

    def forward(self, x):
        return self.reseau(x)

    def fit(self,X,Y):

        self.reseau.train()
        for i in tqdm(range(self.max_iter)):

            X = X.sample(frac=1)
            data0 = X[Y[X.index]==0]
            data1 = X[Y[X.index]==1]

            sc0 = []
            sc1 = []
            critB = []

            ## batch
            nb_batch = min(data0.shape[0],data1.shape[0])//self.batch_size
            # print(nb_batch, data0.shape[0],data1.shape[0])
            for b in range(nb_batch):
                self.optimizer.zero_grad()

                sample0 = data0.iloc[(b*self.batch_size):((b+1)*self.batch_size)]
                sample1 = data1.iloc[(b*self.batch_size):((b+1)*self.batch_size)]
                # print(sample0.shape,sample1.shape)
                sample_0 = Variable(FloatTensor(sample0.values)).to(dev)
                sample_1 = Variable(FloatTensor(sample1.values)).to(dev)

                score_0 = mean(self.forward(sample_0))
                score_1 = mean(self.forward(sample_1))
                sc0.append(float(score_0))
                sc1.append(float(score_1))

                loss = score_0 - score_1
                critB.append(float(loss))

                eps = rand(self.batch_size,1)
                eps = eps.expand_as(sample_0).to(dev)
                interpolation = eps*sample_0 + (1-eps)*sample_1
                interpolation = Variable(interpolation, requires_grad=True)
                score_interp = self.forward(interpolation)
                gradients = grad(outputs=score_interp, inputs=interpolation,
                                 grad_outputs=ones(score_interp.size()).to(dev),
                                 create_graph=True, retain_graph=True)[0]
                gradients = gradients.view(self.batch_size, -1)
                gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
                self.grad_norm.append(gradients_norm.mean().item())
                GP = self.lambd * ((gradients_norm - 1.)**2).mean()

                loss = loss + GP

                loss.backward()
                self.optimizer.step()

            self.score0.append(np.array(sc0).mean())
            self.score1.append(np.array(sc1).mean())
            self.crit.append(np.array(critB).mean())

        self.reseau.eval()
        self.s = self.seuil(X,Y)

    def score_prediction(self,X):
        return self.forward(torch.Tensor(X.values).to(dev)).detach().cpu().reshape(-1)

    def seuil(self,X,Y):
        pred_sc = self.score_prediction(X)
        threshold = pred_sc.numpy().mean()
        return threshold

    def predict(self,X):
        sc = self.score_prediction(X)
        pred = sc.clone()
        pred[sc >= self.s] = 1
        pred[sc < self.s] = 0
        return pred.numpy()

    def score(self,X,Y):
        return ACC(Y,self.predict(X))
