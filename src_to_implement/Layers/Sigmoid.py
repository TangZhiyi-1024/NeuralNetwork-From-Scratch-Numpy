import numpy as np
from .Base import BaseLayer


class Sigmoid(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.activations = None

    def forward(self, input_tensor):
        self.activations = 1 / (1 + np.exp(-1 * input_tensor))
        return self.activations

    def backward(self, error_tensor):   # gradient of the error
        return error_tensor * (self.activations * (1 - self.activations))
