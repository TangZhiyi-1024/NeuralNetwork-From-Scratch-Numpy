import numpy as np

class BatchNormalization:
    def __init__(self, channels: int):
        self.channels = channels
        self.trainable = True
        self.testing_phase = False
        self.image_shape = None

        self.gamma = np.ones(channels)
        self.beta = np.zeros(channels)  # This is equivalent to bias in batch normalization
        # self.learning_rate = learning_rate  # 添加 learning_rate 属性
        self.optimizer = None  # 添加优化器属性

        self.moving_mean = np.zeros(channels)
        self.moving_variance = np.ones(channels)
        self._phase = 'train'


        self.initialize()

    def initialize(self):
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)

    def forward(self, input_tensor: np.ndarray, epsilon=1e-10) -> np.ndarray:
        if input_tensor.ndim == 4:
            self.image_shape = input_tensor.shape
        if self.testing_phase:
            mean = self.moving_mean
            variance = self.moving_variance
        else:
            if input_tensor.ndim == 4:  # 如果输入是卷积层的输出（4D张量）
                mean = np.mean(input_tensor, axis=(0, 2, 3), keepdims=True)
                variance = np.var(input_tensor, axis=(0, 2, 3), keepdims=True)
                # 更新移动平均
                self.moving_mean = mean
                self.moving_variance = variance
            else:  # 如果输入是全连接层的输出（2D张量）
                mean = np.mean(input_tensor, axis=0)
                variance = np.var(input_tensor, axis=0)
                # 更新移动平均
                self.moving_mean = mean
                self.moving_variance = variance

        if self.testing_phase:
            if input_tensor.ndim == 4:
                mean = self.moving_mean.reshape(1, self.channels, 1, 1)
                variance = self.moving_variance.reshape(1, self.channels, 1, 1)
            else:
                mean = self.moving_mean
                variance = self.moving_variance
        else:
            if input_tensor.ndim == 4:
                mean = mean
                variance = variance
            else:
                mean = mean
                variance = variance

        normalized_tensor = (input_tensor - mean) / np.sqrt(variance + epsilon)
        if input_tensor.ndim == 4:
            output_tensor = self.gamma.reshape(1, self.channels, 1, 1) * normalized_tensor + self.beta.reshape(1, self.channels, 1, 1)
        else:
            output_tensor = self.gamma * normalized_tensor + self.beta

        self.normalized_tensor = normalized_tensor
        self.input_tensor = input_tensor
        self.mean = mean
        self.variance = variance
        self.epsilon = epsilon

        return output_tensor

    def backward(self, error_tensor: np.ndarray) -> np.ndarray:
        batch_size = error_tensor.shape[0]

        # 计算损失函数对gamma和beta的梯度
        if error_tensor.ndim == 4:
            grad_gamma = np.sum(error_tensor * self.normalized_tensor, axis=(0, 2, 3))
            grad_beta = np.sum(error_tensor, axis=(0, 2, 3))
        else:
            grad_gamma = np.sum(error_tensor * self.normalized_tensor, axis=0)
            grad_beta = np.sum(error_tensor, axis=0)
        self.grad_gamma = grad_gamma
        self.grad_beta = grad_beta
        # self.grad_gamma = grad_gamma
        # self.grad_beta = grad_beta
        if self.optimizer:
            self.gamma = self.optimizer.calculate_update(self.gamma, self.grad_gamma)
            self.beta = self.optimizer.calculate_update(self.beta, self.grad_beta)
        # 计算损失函数对标准化输入的梯度 (∂L/∂Ỹ)
        if error_tensor.ndim == 4:
            grad_normalized = error_tensor * self.gamma.reshape(1, self.channels, 1, 1)
        else:
            grad_normalized = error_tensor * self.gamma

        # 计算损失函数对方差的梯度 (∂L/∂σ²_B)
        if error_tensor.ndim == 4:
            grad_variance = np.sum(
                grad_normalized * (self.input_tensor - self.mean.reshape(1, self.channels, 1, 1)) *
                -0.5 * np.power(self.variance.reshape(1, self.channels, 1, 1) + self.epsilon, -1.5),
                axis=(0, 2, 3)
            )
        else:
            grad_variance = np.sum(
                grad_normalized * (self.input_tensor - self.mean) *
                -0.5 * np.power(self.variance + self.epsilon, -1.5),
                axis=0
            )

        # 计算损失函数对均值的梯度 (∂L/∂μ_B)
        if error_tensor.ndim == 4:
            grad_mean = np.sum(
                grad_normalized * -1 / np.sqrt(self.variance.reshape(1, self.channels, 1, 1) + self.epsilon),
                axis=(0, 2, 3)
            )
        else:
            grad_mean = np.sum(grad_normalized * -1 / np.sqrt(self.variance + self.epsilon), axis=0)

        # 计算损失函数对输入数据的梯度 (∂L/∂X)
        if error_tensor.ndim == 4:
            grad_input = (
                    grad_normalized / np.sqrt(self.variance.reshape(1, self.channels, 1, 1) + self.epsilon) +
                    grad_variance.reshape(1, self.channels, 1, 1) * 2 * (
                                self.input_tensor - self.mean.reshape(1, self.channels, 1, 1)) / (
                                batch_size * self.input_tensor.shape[2] * self.input_tensor.shape[3]) +
                    grad_mean.reshape(1, self.channels, 1, 1) / (
                                batch_size * self.input_tensor.shape[2] * self.input_tensor.shape[3])
            )
        else:
            grad_input = (
                    grad_normalized / np.sqrt(self.variance + self.epsilon) +
                    grad_variance * 2 * (self.input_tensor - self.mean) / batch_size +
                    grad_mean / batch_size
            )

        return grad_input


    def reformat(self, tensor):
        if tensor.ndim == 4:  # 图像到向量
            batch_size, _, height, width = tensor.shape
            return tensor.reshape(batch_size, self.channels, height * width).transpose(0, 2, 1).reshape(-1, self.channels)
        elif tensor.ndim == 2:  # 向量到图像
            if self.image_shape is None:
                raise ValueError("Image shape is not set. Call forward with a 4D tensor first.")
            batch_size, _, height, width = self.image_shape
            return tensor.reshape(batch_size, height * width, self.channels).transpose(0, 2, 1).reshape(self.image_shape)
        else:
            raise ValueError("Input tensor must be 2D or 4D.")

    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, value):
        self.gamma = value

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, value):
        self.beta = value

    @property
    def gradient_weights(self):
        return self.grad_gamma

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

    @property
    def gradient_bias(self):
        return self.grad_beta