from Layers.Base import BaseLayer
import numpy as np

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        # 包括偏置项的权重矩阵，大小为 (input_size + 1, output_size)
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self._optimizer = None  # 初始化优化器为 None
        self._gradient_weights = None  # 初始化权重梯度为 None
        self.bias = np.random.uniform(0, 1, (output_size,))  # 初始化偏置

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, self.input_size + 1, self.output_size)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.output_size)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        # 增加偏置项到输入张量中
        input_tensor = np.hstack((input_tensor, np.ones((batch_size, 1))))
        self.input_tensor = input_tensor
        return np.dot(input_tensor, self.weights)

    def backward(self, error_tensor):
        # 计算权重梯度
        self._gradient_weights = np.dot(self.input_tensor.T, error_tensor)

        # 如果有优化器，更新权重
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        # 计算并返回传递给前一层的误差张量（去掉偏置项）
        return np.dot(error_tensor, self.weights[:-1].T)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights
