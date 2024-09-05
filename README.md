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
import torch
import lrp

model = Sequential(
    lrp.Conv2d(1, 32, 3, 1, 1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2, 2),
    torch.nn.Flatten(),
    lrp.Linear(14*14*32, 10)
)

x = ... # business as usual
y_hat = model.forward(x, explain=True, rule="alpha2beta1")
y_hat = y_hat[torch.arange(batch_size), y_hat.max(1)[1]] # Choose maximizing output neuron
y_hat = y_hat.sum()

# Backward pass (do explanation)
y_hat.backward()
explanation = x.grad
```


**Implemented rules:**
|Rule 							|Key 					| Note 												|
|:------------------------------|:----------------------|:--------------------------------------------------|
|epsilon-rule					| "epsilon" 			| Implemented but epsilon fixed to `1e-1` 			|
|gamma-rule						| "gamma" 				| Implemented but gamma fixed to `1e-1`				|
|epsilon-rule					| "epsilon" 			| gamma and epsilon fixed to `1e-1`					|
|alpha=1 beta=0 				| "alpha1beta0" 		| 													|
|alpha=2 beta=1 				| "alpha2beta1" 		| 													|
|PatternAttribution (all) 		| "patternattribution" 	| Use additional argument `pattern=patterns_all` 	|
|PatternAttribution (positive) 	| "patternattribution" 	| Use additional argument `pattern=patterns_pos` 	|
|PatternNet (all) 				| "patternnet" 			| Use additional argument `pattern=patterns_all` 	|
|PatternNet (positive) 			| "patternnet" 			| Use additional argument `pattern=patterns_pos` 	|



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
