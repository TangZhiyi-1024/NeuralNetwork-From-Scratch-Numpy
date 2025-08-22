# Neural Networks From Scratch (NumPy)

A lightweight, NumPy-based mini framework for training neural networks from first principles. It includes modular layers (FC/Conv/Pooling/Activations), regularization (BatchNorm/Dropout/L2), a simple RNN, losses and optimizers, unit tests, and a LeNet training script.

## Features

* Modular layers: Fully Connected, Convolution, Pooling, Flatten, common activations
* Regularization & stabilization: Batch Normalization, Dropout, L2 weight decay
* Recurrent layer: basic RNN with backpropagation through time
* Optimizers & losses: configurable training loop
* Unit tests and a self-contained LeNet example
* Minimal dependencies (NumPy + Matplotlib)

## Project layout

```
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
├─ metplotlib/              # consider renaming to `matplotlib/`
├─ NeuralNetwork.py
├─ NeuralNetworkTests.py
├─ TrainLeNet.py
├─ log.txt
├─ 3_1_Regularization.pdf
├─ 3_2_Recurrent.pdf
└─ dispatch.py
```

## Quick start

### 1) Run tests

Validate forward/backward passes, tensor shapes, and gradient flow:

```bash
python NeuralNetworkTests.py
```

### 2) Train LeNet

```bash
python TrainLeNet.py
```

Typical flow: load data → build LeNet → set loss/optimizer → train for N epochs → write logs/plots.
Check available flags (if provided) with:

```bash
python TrainLeNet.py -h
```

### 3) Task dispatcher (optional)

If present, this can run predefined tasks:

```bash
python dispatch.py
```

## Implementation guide

**Layer contract**

* `forward(x)`: returns output and caches intermediates needed for backprop.
* `backward(dout)`: returns `dx` and accumulates parameter gradients.
* Switch train/eval modes where needed (BatchNorm/Dropout).

**Convolution/Pooling**

* Handle padding/stride/output size carefully.
* Keep tensor layout consistent across layers.

**Softmax + Cross-Entropy**

* Use numerically stable computations (subtract max, log-sum-exp).

**BatchNorm/Dropout**

* Different behavior in training vs. evaluation; manage running stats and masks.

**RNN**

* Sequence forward, hidden-state carryover, and BPTT (truncated if desired).

**Optimizers/Losses**

* Implement/choose in `Optimization/` (e.g., SGD, Momentum, Adam; CE, MSE).

## Tips & troubleshooting

* Start small (tiny dataset subset, few iterations) and confirm the loss decreases.
* Print/assert shapes at boundaries; NCHW/NHWC mismatches are common.
* Add complexity gradually: baseline → BN/Dropout/weight decay.
* Use numerical gradient checks (where available) before long trainings.
* Logs and figures: see `log.txt` and the plotting utilities.


