
import copy
from Layers.BatchNormalization import *

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer=None, bias_initializer=None):
        self.bias_initializer = bias_initializer
        self.weights_initializer = weights_initializer
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self._phase = None  # 新增：phase属性
        self.regularization_loss = 0

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value
        for layer in self.layers:
            if hasattr(layer, 'phase'):
                layer.phase = value

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        reg_loss = 0
        for layer in self.layers:
            if hasattr(layer, 'weights') and self.optimizer.regularizer:
                reg_loss += self.optimizer.regularizer.norm(layer.weights)
            input_tensor = layer.forward(input_tensor)
        self.regularization_loss = reg_loss
        return self.loss_layer.forward(input_tensor, label_tensor) + reg_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.loss_layer.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            if isinstance(layer, BatchNormalization):
                layer.initialize()
            else:
                layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        self.phase = 'train'  # 设置phase为train
        for _ in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        self.phase = 'test'  # 设置phase为test
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
