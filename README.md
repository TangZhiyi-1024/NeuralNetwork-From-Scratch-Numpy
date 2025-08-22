README.md
Deep Learning Exercise 3 — Neural Networks from Scratch (Regularization, RNN, LeNet)
Overview

This repository contains a from-scratch (NumPy-based) neural network mini-framework and exercises. You’ll implement core layers and training logic, add regularization and optimizers, and train classic models such as LeNet. A simple RNN is included to introduce recurrent computation and BPTT.

Highlights

Modular layers: Fully Connected, Conv, Pooling, Flatten, common activations

Regularization: L2 weight decay, Dropout, Batch Normalization

Recurrent layer: basic RNN

Optimizers and losses

Unit tests and a LeNet training script

Minimal dependencies (NumPy + Matplotlib)

Project layout
src_to_implement/
├─ Data/
├─ Layers/
│  ├─ Base.py
│  ├─ BatchNormalization.py
│  ├─ Conv.py
│  ├─ Dropout.py
│  ├─ Flatten.py
│  ├─ FullyConnected.py
│  ├─ Helpers.py
│  ├─ Initializers.py
│  ├─ Pooling.py
│  ├─ RNN.py
│  ├─ ReLU.py
│  ├─ Sigmoid.py
│  ├─ SoftMax.py
│  ├─ TanH.py
│  └─ __init__.py
├─ Models/
│  └─ __init__.py
├─ Optimization/
│  ├─ Constraints.py
│  ├─ Loss.py
│  ├─ Optimizers.py
│  └─ __init__.py
├─ metplotlib/             # (typo) consider renaming to 'matplotlib'
├─ NeuralNetwork.py
├─ NeuralNetworkTests.py
├─ TrainLeNet.py
├─ log.txt
├─ 3_1_Regularization.pdf
├─ 3_2_Recurrent.pdf
└─ dispatch.py

Installation
python -V          # 3.8+ recommended
pip install numpy matplotlib


If your dataset loader or plotting utilities require anything else, install as needed.

Quick start
1) Run the unit tests

Make sure layer shapes, forward/backward passes, and gradients behave as expected.

python NeuralNetworkTests.py

2) Train LeNet (example)
python TrainLeNet.py


The script typically: loads data → builds LeNet → sets loss/optimizer → trains for N epochs → logs to log.txt and/or plots with Matplotlib.

Check available flags (if any) with:

python TrainLeNet.py -h

3) (Optional) Exercise dispatcher

If dispatch.py is provided to run specific subtasks:

python dispatch.py

What you implement

Each layer/module follows a simple contract:

forward(x) returns outputs and caches what’s needed for backprop.

backward(dout) returns gradients w.r.t. inputs and accumulates param grads.

Train/Eval modes (BatchNorm/Dropout) must be handled correctly.

Focus areas:

Layers: FullyConnected, Conv (padding/stride/output size), Pooling, Flatten

Activations: ReLU, Sigmoid, TanH

Output: SoftMax (use log-sum-exp tricks for numerical stability)

Regularization: BatchNormalization, Dropout

Recurrent: RNN forward over sequences and BPTT

Optimization/Loss: implementations in Optimization/Loss.py and Optimizers.py (e.g., SGD, Momentum, Adam; cross-entropy, MSE, etc.)

Tips & troubleshooting

Start small: a tiny subset of data and a few iterations—ensure loss goes down.

Verify shapes at every layer; mismatched NCHW/NHWC will break conv/pool.

For Softmax+CrossEntropy use stable computations (subtract max, log-sum-exp).

BatchNorm/Dropout: different behavior in train vs. eval.

Use the tests and (if available) numerical gradient checks before full training.

Dataset note

The Data/ module loads the dataset configured by the scripts (commonly MNIST for LeNet). If you need to switch datasets, adjust the loader and model input shapes accordingly.

Naming & logging

Consider renaming metplotlib/ → matplotlib/ for clarity.

Training output/metrics may be written to log.txt.

