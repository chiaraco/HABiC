# Implementation of HABiC and Wass-NN

## Installation
To install requirements, refer to the [`requirements.yml`](requirements.yml)
file.

If you use `conda`, then you can install an environment called `HABiCenv` by
executing the following command: 

```bash
> conda env create -f requirements.yml
```

## Usage 

The code can be used as follows:

```python
#######################################################
##### import required functions
from sklearn.metrics import matthews_corrcoef as MCC
from functionsHABiC import classification

#######################################################
##### load your data
X, Y, Xval1, Yval1, Xval2, Yval2 = ...

#######################################################
##### test an algorihtm
# if naive.HABiC
params_naive = {'meth':'naive.HABiC'}

#######################################################
##### performances
perf = classification(X, Y, [Xval1,Xval2], [Yval,Yval2], ['Valid.1','Valid.2'], param=params_naive, metr='MCC')
```


**Implemented method:**
| Method 						                    | Key 					| Parameters 										                                                                            |
|:--------------------------------------------------|:----------------------|:------------------------------------------------------------------------------------------------------------------------------|
| HABiC (naive approach)	                        | "naive.HABiC" 		|                                         			                                                                            |
| HABiC after reduction by PCA					    | "redPCA.HABiC" 		| 'DimRed' : reduction dimension            	                                                                                |
| HABiC after reduction by PLS-DA				    | "redPLS.HABiC" 		| 'DimRed' : reduction dimension    					                                                                        |
| HABiC with standard bagging				        | "bagSTD.HABiC" 		| 'NbTrees' : number of sub-algorithms				                                                                            |
| HABiC with bagging and RF feature selection 		| "bagRF.HABiC" 		| 'NbTrees' : number of sub-algorithms, 'NbVarImp' : number of features to select	                                            |
| HABiC with bagging and PLS-DA feature selection	| "bagPLS.HABiC" 	    | 'NbTrees' : number of sub-algorithms, 'NbVarImp' : number of features to select	                                            |
| Wasserstein Neural Netwrok 	                    | "Wass-NN" 	        | 'struct' : net architecture ('hidden_layer_sizes','activation','solver','batch_size','learning_rate_init','max_iter','lambd') |




## Synthetic data
For a complete running example, please see [scriptHABiC.py](scriptHABiC.py).
The code generates two DataFrames with prediction performances (mean and std) of all presented algorithms. 

To run the example code, simply activate the conda environment and execute the code from the root of the project:
```bash
> conda activate HABiCenv
> python scriptHABiC.py
```
