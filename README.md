# Continuous-exploratory Discrete Langevin Sampler

This repository contains code for the paper
[Continuous-exploratory Discrete Langevin Sampler](../Continuous_exploratory_Discrete_Langevin_Sampler.pdf).

# Introduction
We propose Continuous-exploratory Discrete Langevin Sampler (cDLS), a hybrid gradient-based sampler for discrete distributions. cDLS exploits the gradient information to explore the continuous space then samples. We theoretically prove the efficiency of cDLS by showing that without a Metropolis-Hastings correction, the asymptotic bias of cDLS is zero for log-quadratic distributions, and is small for distributions that are close to being log-quadratic. We also provide a non-asymptotic convergence and inference guarantees for general discrete distribution. With cDLS, we develop several variants of sampling algorithms, including unadjusted, Metropolis-adjusted versions, indicating the general applicability for different scenarios. We demonstrate the effectiveness of our proposed algorithm on several experiments, including the Ising model, restricted Boltzmann machines, deep energy-based model, and binary Bayesian neural network.


# Dependencies
* [PyTorch 1.9.1](http://pytorch.org/) 
* [torchvision 0.10.1](https://github.com/pytorch/vision/)

# Usage
## Sampling From Ising Models
Please run
```
python ising_sample.py
```
## Sampling From Restricted Boltzmann Machines
Please run
```
python rbm_sample.py
```
## Learning Ising Models
Run ``bash generate_data.sh`` to generate the data, then learn the Ising model by running
```
python pcd.py --sampler=<SAMPLER>
```
* ```SAMPLER``` &mdash; Specify which sampler to use. \
                        ``cdmala``: continuous-exploratory discrete Metropolis-adjusted Langevin algorithm; \
                        ``cdula``: continuous-exploratory discrete unadjusted Langevin algorithm 

Use ``plt_pcd`` to plot the results of log RMSE with respect to the number of iterations and the runtime.

## Learning Deep EBMs
The datasets can be found [here](https://github.com/jmtomczak/vae_vampprior/tree/master/datasets).

To learn the EBM, run ``bash ebm.sh`` and to evaluate the learned EBM using AIS, run ``ais.sh``.


## Binary Bayesian Neural Networks
See 
```
./BinaryBNN
```

# References
* This repo is built upon the [GWG repo](https://github.com/wgrathwohl/GWG_release) 
