import numpy as np

class Dropout:
    def __init__(self, probability: float):
        """
        初始化 Dropout 层

        参数:
        probability (float): 保留单位的比例
        """
        self.probability = probability
        self.mask = None
        self.phase = 'train'  # 默认情况下处于训练阶段
        self.trainable = False  # Dropout 层没有可训练的参数

    def forward(self, input_tensor: np.ndarray) -> np.ndarray:
        """
        前向传播

        参数:
        input_tensor (np.ndarray): 输入张量

        返回:
        np.ndarray: 经过 dropout 处理后的张量
        """
        if self.phase == 'train':
            self.mask = (np.random.rand(*input_tensor.shape) < self.probability) / self.probability
            return input_tensor * self.mask
        else:
            return input_tensor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        """
        反向传播

        参数:
        error_tensor (np.ndarray): 误差张量

        返回:
        np.ndarray: 经过 dropout 处理后的误差张量
        """
        return error_tensor * self.mask
    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

    @property
    def testing_phase(self):
        return self._phase == 'test'

    @testing_phase.setter
    def testing_phase(self, value):
        self._phase = 'test' if value else 'train'
