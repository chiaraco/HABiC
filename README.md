# Implementation of HABiC and Wass-NN

## Installation
Before using HABiC, it is necessary to install requirements presented in [`requirements.yml`](requirements.yml) file.

If you are using `conda`, you can install an environment called `HABiCenv`. First, clone this repository ([Cloning a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)), or download ([Downloading a repository](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github)) and unpack it. 

Then, open an Anaconda prompt and run the following command :
```bash
> conda env create -f path\to\the\folder\requirements.yml  # path to the folder where the requirements.yml file is
```
( The installation can take more than 15 min)

The IDE is not included in the environment, so you can install one if necessary.
```bash
> conda activate HABiCenv
#if needed:
> conda install spyder  
> spyder 
```


## Usage 

The algorithm can be used as follows:

```python
#######################################################
##### import required functions

# You have to be in the folder where functionsHABiC.py is.

import pandas
from sklearn.metrics import matthews_corrcoef as MCC
from functionsHABiC import classification
from functionsHABiC import performances 

#######################################################
##### load your data

# With your own datasets
#------------------------

# Train dataset
X = pandas.read_csv(...) # pandas DataFrame with observations in row, genes in column
Y = pandas.read_csv(...) # pandas Series with class to predict

# External dataset(s) - Y for external dataset can be loaded if available, to realize performance tests.
Xext1 = pandas.read_csv(...) # pandas DataFrame with observations in row, genes in column for external dataset 1
Xext2 = pandas.read_csv(...) # obs/gene dataframe for external dataset 2
# Optional:
Yext1 = pandas.read_csv(...) # pandas Series with class to predict for external dataset 1
Yext2 = pandas.read_csv(...) # class to predict for external dataset 2

# Categorical variables can be included with OneHotEncoder
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html


# With the included datasets (in the same folder than the one with functionsHABiC.py file)
#-------------------------------------------------------------------------------------------------

to_load = 'data'

data_train = pandas.read_csv(f'{to_load}/train.csv',header=0,index_col=0)
X = data_train.drop('Y',axis=1)
Y = data_train['Y'].copy()

data_valid = pandas.read_csv(f'{to_load}/valid.csv',header=0,index_col=0)
Xval1 = data_valid.drop('Y',axis=1)
Yval1 = data_valid['Y'].copy() # Y for validation is not required, but can be loaded in order to allow perfomance evaluation


#######################################################
##### choose HABiC parameters (example)

# if naive.HABiC
params_naive = {'meth':'naive.HABiC'}  

# if bagPLS.HABiC
params_bagPLS = {'meth':'bagPLS.HABiC', 'NbTrees':50, 'NbVarImp':3}

#see all implemented methods and their associated parameters in the table below

```


**Implemented methods:**
| Method 						                    | Key 					| Parameters 										                                                                            |
|:--------------------------------------------------|:----------------------|:------------------------------------------------------------------------------------------------------------------------------|
| HABiC (naive approach)	                        | "naive.HABiC" 		|                                         			                                                                            |
| HABiC after reduction by PCA					    | "redPCA.HABiC" 		| 'DimRed' : dimension reduction to select            	                                                                                |
| HABiC after reduction by PLS-DA				    | "redPLS.HABiC" 		| 'DimRed' : dimension reduction to select    					                                                                        |
| HABiC with standard bagging				        | "bagSTD.HABiC" 		| 'NbTrees' : number of sub-algorithms	to select  			                                                                            |
| HABiC with bagging and RF feature selection 		| "bagRF.HABiC" 		| 'NbTrees' : number of sub-algorithms, 'NbVarImp' : number of features to select	                                            |
| HABiC with bagging and PLS-DA feature selection	| "bagPLS.HABiC" 	    | 'NbTrees' : number of sub-algorithms, 'NbVarImp' : number of features to select	                                            |
| Wasserstein Neural Netwrok 	                    | "Wass-NN" 	        | 'struct' : net architecture ('hidden_layer_sizes', 'activation', 'solver', 'batch_size', 'learning_rate_init', 'max_iter', 'lambd') |

```python

#######################################################
##### run the selected classifier

# With your own datasets
#------------------------

# first, use 'classification' function that allows to train the classifier and perform predictions at the same time:

pred = classification(X, Y, [Xext1,Xext2], ['ExtSet.1','ExtSet.2'], param=params_naive)
         # X, Y # train dataset
         # [Xext1,Xext2]  # all dataframes for external datasets, with variables in column and observations in row
         # ['ExtSet.1','ExtSet.2'] # output names to choose for the results table, 'Train' is automatically included
         # (here, it will be 'Train', 'ExtSet.1','ExtSet.2')

# 'pred' will returns a dictionary with the names of the predicted datasets in keys,
# and the class predictions for each observation in values.
# example : you can access to the predictions of 'ExtSet.1'with:
pred['ExtSet.1']

# then, 'performances' function allows to evaluate prediction performance if the true class is known:

perf = performances(Yext1,pred['ExtSet.1'], metr='MCC')
# available metrics : 'MCC' (Matthews correlation coefficient), 'ACC' (accuracy score), 'AUC' (area under the ROC curve)


# With the included datasets (loaded above)
#-------------------------------------------------------------------------------------------------

# Classifier training and prediction:
pred = classification(X, Y, [Xval1], ['Valid.1'], param=params_naive)

# prediction performance (# available metrics : 'MCC', 'AUC', 'ACC')
perf = performances(Yval1,pred['Valid.1'], metr='MCC') 


```

## Run an example with synthetic data testing all methods with cross validation 

For a complete running example on synthetic dataset, please see [scriptHABiC.py](scriptHABiC.py).
The code generates two DataFrames with prediction performances (mean and standard deviation) of all presented algorithms. 

To run the example code, activate the conda environment and execute the code from the root of the project:
```bash
> conda activate HABiCenv
> python scriptHABiC.py
```


## Transcriptomics data preprocessing
When using transcriptomics data, validation datasets could be homogenised with the train dataset using MatchMixeR algorithm (https://doi.org/10.1093/bioinformatics/btz974).
The script for homogenisation is available in [preprocessing.R](preprocessing.R) (with MatchMixer, the reference dataset (here, train dataset) is not modified, only the validation one).

Train and external validation datasets (including class to predict and standard prognostic variables) are available on Zenodo: [10.5281/zenodo.15091117](https://doi.org/10.5281/zenodo.15091117)

After homogenisation  : <br/>
METABRIC dataset was initially downloaded from https://www.cbioportal.org/study/summary?id=brca_metabric (data_expression_median.txt).
Buffa dataset was initially downloaded from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22219 (GSE22219_non-normalized_data_mRNA.csv).
Hatzis dataset was initally downloaded from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE25055  (.CEL download then mas5 normalization).
METABRIC was used as train dataset. Buffa and Hatzis were used as validation datasets.

Without homogenisation (two experiments) : <br/>
1/ Broad train dataset was initially downloaded from https://singlecell.broadinstitute.org/single_cell/study/SCP1039 and the external validation Xu from https://zenodo.org/records/10672250. <br/>
2/ Zhang train dataset was initially downloaded from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40791, and the other two external validations Hou from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE19188 and Seo from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40419.

# License

   Copyright 2024 INSTITUT DE CANCEROLOGIE DE L'OUEST (ICO) and UNIVERSITE ANGERS

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
