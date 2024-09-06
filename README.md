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
X, Y, Xval, Yval = ...

#######################################################
##### test an algorihtm
# if naive.HABiC
params_naive = {'meth':'naive.HABiC'}

#######################################################
##### performances
perf = classification(X, Y, [Xval], [Yval], ['Valid.'], param=params_naive, metr='MCC')
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
For a complete running example, please see [examples/explain_mnist.py](examples/explain_mnist.py).
The code generates this plot: 
<img src="examples/plots/mnist_explanations.png" style="max-width: 500px;"/>

To run the example code, simply activate the conda environment and execute the code from the root of the project:
```bash
> conda activate torchlrp
> python examples/explain_mnist.py
```


## References
[1] Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.R. and Samek, W., 2015. On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. PloS one, 10(7), p.e0130140.  
[2] Kindermans, P.J., Schütt, K.T., Alber, M., Müller, K.R., Erhan, D., Kim, B. and Dähne, S., 2017. Learning how to explain neural networks: Patternnet and patternattribution. arXiv preprint arXiv:1705.05598.  
[3] Montavon, G., Binder, A., Lapuschkin, S., Samek, W. and Müller, K.R., 2019. Layer-wise relevance propagation: an overview. In Explainable AI: interpreting, explaining and visualizing deep learning (pp. 193-209). Springer, Cham.  






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
