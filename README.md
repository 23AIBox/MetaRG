# MetaRG

## Overview of the framework
MetaRG is a computational framework for predicting disease risk genes and identifying shared genetic candidates underlying comorbid disorders. It combines protein sequence embeddings from a pretrained ESM2 model with protein–protein interaction network information through a graph convolutional network and a multilayer perceptron. Its three-stage learning strategy comprises self-supervised pretraining, network propagation for pseudo-label generation, and weakly supervised meta-learning to exploit limited known risk genes. Applied separately to Parkinson’s disease and major depressive disorder, MetaRG generates gene-level risk rankings and prioritizes shared candidates. 
## Installation & Dependencies
The code is written in Python 3 and was mainly tested on Python 3.9.20 and a Linux OS. The package development version is tested on Linux and Windows 10 operating systems. The developmental version of the package has been tested on the following systems:

* Linux: Ubuntu 22.04.3 LTS
* Windows: 10

RL-GeneTrans has the following dependencies:

* networkx
* torch
* torch-cluster
* torch-geometric
* torch-scatter
* torch-sparse
* torch-spline-conv
* scikit-learn
* pandas
* numpy

The details of Python dependencies used in experiments can be found in requirements.txt. 



## Running MetaRG

To clone this repository, users can use:
```
git clone https://github.com/23AIBox/MetaRG.git
```
Set up the required environment using `requirements.txt` with Python. While in the project directory, run:
```
pip install -r requirements.txt
```
It takes about 35 minutes to set up the environment. 
We also provided a conda environment file (from Linux). Users can build the environment by running:
```
conda env create -f environment.yaml
```

Due to the large size of the data files, we have made the data available for download on the https://zenodo.org/records/23234394 public website.

We upload a trained model for risk gene identification. To run this model, you can use the command line instructions:
```
python predict.py --disease both
```
