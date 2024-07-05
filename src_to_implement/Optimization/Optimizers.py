import numpy as np
from .Constraints import *

class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)
        return self._calculate_update(weight_tensor, gradient_tensor)

    def _calculate_update(self, weight_tensor, gradient_tensor):
        raise NotImplementedError


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            gradient_tensor += self.regularizer.calculate_gradient(weight_tensor)
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        if self.regularizer:
            if isinstance(self.regularizer, L2_Regularizer):
                weights_updated = (1 - self.learning_rate * self.regularizer.alpha) * weight_tensor + self.v
            elif isinstance(self.regularizer, L1_Regularizer):
                weights_updated = weight_tensor - self.learning_rate * self.regularizer.alpha * np.sign(weight_tensor) + self.v
            else:
                weights_updated = weight_tensor + self.v
        else:
            weights_updated = weight_tensor + self.v

        return weights_updated



class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.lr = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.t = 1
        self.eps = 1e-8

    def calculate_update(self, weights, grads):
        self.v = self.mu * self.v + (1 - self.mu) * grads
        self.r = self.rho * self.r + (1 - self.rho) * (grads ** 2)
        v_hat = self.v / (1 - self.mu ** self.t)
        r_hat = self.r / (1 - self.rho ** self.t)
        self.t += 1

        update = self.lr * v_hat / (np.sqrt(r_hat) + self.eps)

        if self.regularizer:
            if isinstance(self.regularizer, L2_Regularizer):
                weights = (1 - self.lr * self.regularizer.alpha) * weights - update
            else:
                weights = weights - self.lr * self.regularizer.alpha * np.sign(weights) - update
        else:
            weights -= update

        return weights