import numpy as np
from .Base import BaseLayer
class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.probabilities = None  # 用于存储前向传播的softmax结果

    def forward(self, input_tensor):
        # 通过减去每个样本的最大值来避免数值上的溢出Avoid numerical overflow
        input_shifted = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exps = np.exp(input_shifted)
        self.probabilities = exps / np.sum(exps, axis=1, keepdims=True)
        return self.probabilities

    def backward(self, error_tensor):
        # 计算每个元素的梯度，使用前向传播时计算的概率值
        temp = self.probabilities * error_tensor
        sum_temp = np.sum(temp, axis=1, keepdims=True)
        d_input = temp - self.probabilities * sum_temp
        return d_input
