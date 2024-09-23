# Implementation of HABiC and Wass-NN

## Installation
To install requirements, refer to the [`requirements.yml`](requirements.yml) file.

If you are using `conda`, you can install an environment called `HABiCenv`. First, clone this repository ([Cloning a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)), or download ([Downloading a repository](https://docs.github.com/en/get-started/start-your-journey/downloading-files-from-github)) and unpack it. 

Then, open an Anaconda prompt and run the following command :
```bash
> conda env create -f path\to\the\folder\requirements.yml  # path to the folder where the requirements.yml file is.
```
(Installation can take more than 15 min)


## Usage 

The code can be used as follows:

```python
#######################################################
##### import required functions

import pandas
from sklearn.metrics import matthews_corrcoef as MCC
from functionsHABiC import classification  # you should be in the folder where functionsHABiC.py is.

#######################################################
##### load your data

# training data
X = pandas.read_csv(...) # pandas DataFrame with observations in row, genes (and categorical variables) in column
Y = pandas.read_csv(...) # pandas Series with class to predict

# validation data
Xval1, Yval1  # external validation 1
Xval2, Yval2  # external validation 2

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
##### Run a classifier

pred = classification(X, Y, [Xval1,Xval2], [Yval,Yval2], ['Valid.1','Valid.2'], param=params_naive)
# X, Y # train dataset
# Xval1, Yval1  # external validation 1
# Xval2, Yval2  # external validation 2

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
When using transcriptomics data, validation datasets are prealably homogenised with the train dataset using MatchMixeR algorithm (https://doi.org/10.1093/bioinformatics/btz974).
The script for homogenisation is available in [preprocessing.R](preprocessing.R).


# License

   Copyright 2024 INSTITUT DE CANCEROLOGIE DE L'OUEST and UNIVERSITE ANGERS

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
