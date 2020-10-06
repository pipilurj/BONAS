# BONAS

Experiment run of proposed algorithm BONAS in search space of DARTS

## Background
Neural Architecture Search (NAS) has shown great potentials in finding  better neural network designs. Sample-based NAS is the most reliable approach which aims at exploring the search space and evaluating the most promising architectures. However, it is computationally very costly.  As a remedy, the one-shot approach has emerged as a popular technique for accelerating NAS using weight-sharing. However, due to the weight-sharing of vastly different networks, the one-shot approach is less reliable than the sample-based approach. In this work, we propose BONAS (Bayesian Optimized Neural Architecture Search), a sample-based NAS framework which is accelerated using weight-sharing to evaluate multiple related architectures simultaneously.
Specifically, we apply Graph Convolutional Network predictor as a surrogate model for Bayesian Optimization to select multiple related candidate models in each iteration. We then apply weight-sharing to train multiple candidate models simultaneously. This approach not only accelerates the traditional sample-based approach significantly, but also keeps its reliability. This is because weight-sharing among related architectures are more reliable than those in the one-shot approach. 
Extensive experiments are conducted to verify the effectiveness of our method over many competing algorithms.

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
* [Scalable global optimization via local bayesian optimization](https://arxiv.org/abs/1910.01739)
* [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)

