# BONAS

Experiment run of proposed algorithm BONAS in search space of DARTS

## Background

Neural Architecture Search (NAS) has shown great potentials in finding a better neural network design than human design. Sample-based NAS is the most fundamental method aiming at exploring the search space and evaluating the most promising architecture. However, few works have focused on improving the sampling efficiency for a multi-objective NAS. Inspired by the nature of the graph structure of a neural network, we propose BOGCN-NAS, a NAS algorithm using Bayesian Optimization with Graph Convolutional Network (GCN) predictor. Specifically, we apply GCN as a surrogate model to adaptively discover and incorporate nodes structure to approximate the performance of the architecture.

### Prerequisites

Dependency files that have to be downloaded before running the program
```
pip install numpy
pip install scipy
pip install pytorch
```
Download cifar10 datset and place it into ./data

### Run experiment script

A log file recording experiment result will be generated after executing the script.
```
cd BONAS
python bogcn_distributed_singlecard.py
```
All experiment settings are specified in settings.py, one can change them accordingly

## Related Work

* [Scalable Bayesian Optimization Using Deep Neural Networks](https://arxiv.org/pdf/1502.05700.pdf)
* [Semi-Supervised Classification with Graph Convolutional Networks](https://openreview.net/pdf?id=SJU4ayYgl)
* [scalable global optimization via local bayesian optimization](https://arxiv.org/abs/1910.01739)
