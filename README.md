# NNacceleration4EARCG

This repository contains code that leverages the Density Functional Toolkit (DFTK), the energy-adaptive conjugate gradient method and PyTorch to generate training data, train a neural network, and analyze its performance. The code was written in equal parts by Jonas Püschel [@jonas-pueschel](https://github.com/jonas-pueschel) and Kilian Rueß [@kilianar](https://github.com/kilianar).

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Training Data Generation](#training-data-generation)
- [Training the Neural Network](#training-the-neural-network)
- [Evaluating the Trained Model](#evaluating-the-trained-model)
- [Play around with enhanced EARCG](#play-around-with-enhanced-earcg)


## Overview
The goal of this project is to enhance the performance of nonlinear solvers using a neural network. The workflow consists of three main steps:
1. **Generating training data** from partial optimization results using DFTK.
2. **Training a neural network** on the generated dataset using PyTorch.
3. **Evaluating the trained model** by comparing its performance with a classical algorithm.


## Dependencies

### External Software and Data

After cloning this repository, the following steps need to be done to run the code.
- clone the repository of the energy-adaptive conjugate gradient method for DFTK [RCG_DFTK](https://github.com/jonas-pueschel/RCG_DFTK) and manually set the path to `rcg.jl` in `julia/dftk_setup.jl`.
- (optional) get the training and validation data of the paper from [Zenodo](https://zenodo.org/records/15791260). It can be found in `data.zip` and needs to be unpacked to `./data/`.
- (optional) get the trained model from [Zenodo](https://zenodo.org/records/15791260). It can be foound in `model_paper.pth` and needs to be put at `./models/model_paper.pth`
- (optional) get the performance benchmark data of the paper from [Zenodo](https://zenodo.org/records/15791260). It can be found in `comp_paper.zip` and needs to be unpacked to `./comparisons/comp-paper/`.

### Python
Python dependencies are listed in `requirements.txt` and can be installed with:

```shell
pip install -r python/requirements.txt
```

### Julia
Julia dependencies are specified in `Project.toml`. To install them, use:

```shell
julia -e 'using Pkg; Pkg.instantiate()'
```

## Training Data Generation
The training data is generated using the Julia script `data_generation.jl`. Execute the script from the project root with:

```shell
julia julia/data_generation.jl
```

This script produces a dataset containing:
- **Partial optimization results**
- **Energy-adaptive gradient**
- **Converged optimization results**

The output files are stored in the `./data/` directory:
- `final_x.npy`
- `partial_x_y.npy`
- `grad_x_y.npy`

where:
- `x` denotes the index of the data generation run.
- `y` represents the index of the sample for run `x`.

### Splitting the Data
To divide the dataset into training and evaluation sets, run:

```shell
python ./python/split.py
```

By default, 90% of the data is allocated for training and 10% for validation. The split datasets are saved in `./data/training/` and `./data/evaluation/`.

## Training the Neural Network
Train the neural network by executing:

```shell
python ./python/train.py
```

Ensure that the `./data/` directory contains the subfolders `training/` and `evaluation/` before starting the training.

### Training Outputs
During training, all outputs are stored in `result/unet_${date}-${time}/`, including:
- **Best checkpoint** (`model.pth`), selected based on validation accuracy.
- **Loss values** (`losses.json`).
- **Loss plots** (`losses.pdf`).
- **Epoch-wise visualizations** of network outputs on training and validation data (`epoch_${n}_training.png`).

If a GPU is available, the model will train on it automatically; otherwise, the CPU will be used. 

## Evaluating the Trained Model
To compare the classical algorithm with the neural-augmented approach, run:

```shell
julia ./julia/compare_algorithms.jl
```

This script:
- Generates random initial data.
- Executes the classical, the neural-random and neural-augmented algorithms.
- Stores performance data in `comparisons/comp-${date}-${time}/`.

Each test case is stored in `example-${n}/`, containing:
- **Neural network output visualization** (`neural_progression.png`).
- **Performance statistics** (`statistics.csv`).

### Performance Analysis
To analyze the results, run:

```shell
julia ./julia/plot_compare_algorithms.jl
```

This script:
- Parses `statistics.csv` for each test case.
- Generates histograms of iteration differences, percentual iteration differences, runtime differenence and density approximation improvement between the classical, the neural-random and the neural-augmented algorithms.
- Computes and mean and median differences.
- Identifies cases where the neural-augmented approach over- and underperformed.
- Generates all the `tikz` figure plots also found in the paper

## Play around with enhanced EARCG

The script `test_gp.jl` allows to use the enhanced EARCG in a comprehensible enviroment. It can be run using
```shell
julia ./julia/test_gp.jl
```
and optionally can use data from a comparison example, when the path is provided in the script. 
