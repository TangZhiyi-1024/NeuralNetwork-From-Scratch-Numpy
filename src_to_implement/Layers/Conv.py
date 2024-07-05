import copy

import numpy as np
from scipy.signal import correlate, correlate2d

class Conv:
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        if isinstance(stride_shape, list):
            self.stride_shape = tuple(stride_shape)
        elif isinstance(stride_shape, int):
            self.stride_shape = (stride_shape,)
        else:
            self.stride_shape = stride_shape

        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True

        if len(convolution_shape) == 2:  # 1D convolution
            c, m = convolution_shape
            self.weights = np.random.uniform(0, 1, (num_kernels, c, m))
            self.bias = np.random.uniform(0, 1, (num_kernels,))
        elif len(convolution_shape) == 3:  # 2D convolution
            c, m, n = convolution_shape
            self.weights = np.random.uniform(0, 1, (num_kernels, c, m, n))
            self.bias = np.random.uniform(0, 1, (num_kernels,))
        self._optimizer = None
        self._optimizer_bias = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(self.convolution_shape) == 2:  # 1D convolution
            b, c, y = input_tensor.shape
            num_kernels, _, kernel_size = self.weights.shape
            stride_y = self.stride_shape[0]
            out_y = (y - 1) // stride_y + 1  # Calculate the output length
            output_tensor = np.zeros((b, num_kernels, out_y))

            for i in range(num_kernels):
                for j in range(b):
                    for k in range(c):
                        output_tensor[j, i, :] += correlate(input_tensor[j, k, :], self.weights[i, k, :], mode='same')[::stride_y]
                    output_tensor[j, i, :] += self.bias[i]

        elif len(self.convolution_shape) == 3:  # 2D convolution
            b, c, y, x = input_tensor.shape
            num_kernels, _, kernel_y, kernel_x = self.weights.shape
            stride_y, stride_x = self.stride_shape
            out_y = (y - 1) // stride_y + 1
            out_x = (x - 1) // stride_x + 1
            output_tensor = np.zeros((b, num_kernels, out_y, out_x))

            for i in range(num_kernels):
                for j in range(b):
                    for ch in range(c):
                        output_tensor[j, i, :, :] += correlate2d(input_tensor[j, ch, :, :], self.weights[i, ch, :, :], mode='same')[::stride_y, ::stride_x]
                    output_tensor[j, i, :, :] += self.bias[i]

        return output_tensor

    def backward(self, error_tensor):
        b, c, *spatial_dims = self.input_tensor.shape

        if len(spatial_dims) == 1:  # 1D convolution
            y = spatial_dims[0]
            m_kernel = self.weights.shape[2]
            pad_y = (m_kernel - 1) // 2
            y_out = (y + 2 * pad_y - m_kernel) // self.stride_shape[0] + 1
            padded_input = np.pad(self.input_tensor, ((0, 0), (0, 0), (pad_y, pad_y)), mode='constant')

            self._gradient_weights = np.zeros_like(self.weights)
            self._gradient_bias = np.zeros_like(self.bias)
            grad_input = np.zeros_like(padded_input)

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    self._gradient_bias[kernel] += np.sum(error_tensor[batch, kernel, :])
                    for i in range(y_out):
                        start = i * self.stride_shape[0]
                        end = start + m_kernel
                        if end <= y + 2 * pad_y:  # Ensure we do not exceed padded_input dimensions
                            self._gradient_weights[kernel] += error_tensor[batch, kernel, i] * padded_input[batch, :,
                                                                                               start:end]
                            grad_input[batch, :, start:end] += error_tensor[batch, kernel, i] * self.weights[kernel]

            if m_kernel > 1:
                grad_input = grad_input[:, :, pad_y:-pad_y]

        else:  # 2D convolution
            y, x = spatial_dims
            m_kernel, n_kernel = self.weights.shape[2:4]
            pad_y = (m_kernel - 1) // 2
            pad_x = (n_kernel - 1) // 2
            y_out, x_out = error_tensor.shape[2:4]
            padded_input = np.pad(self.input_tensor, ((0, 0), (0, 0), (pad_y, pad_y), (pad_x, pad_x)), mode='constant')

            self._gradient_weights = np.zeros_like(self.weights)
            self._gradient_bias = np.zeros_like(self.bias)
            grad_input = np.zeros_like(padded_input)

            for batch in range(b):
                for kernel in range(self.num_kernels):
                    self._gradient_bias[kernel] += np.sum(error_tensor[batch, kernel, :, :])
                    for i in range(y_out):
                        for j in range(x_out):
                            y_start = i * self.stride_shape[0]
                            y_end = y_start + m_kernel
                            x_start = j * self.stride_shape[1]
                            x_end = x_start + n_kernel
                            if y_end <= y + 2 * pad_y and x_end <= x + 2 * pad_x:  # Ensure we do not exceed padded_input dimensions
                                self._gradient_weights[kernel] += error_tensor[batch, kernel, i, j] * padded_input[
                                                                                                      batch, :,
                                                                                                      y_start:y_end,
                                                                                                      x_start:x_end]
                                grad_input[batch, :, y_start:y_end, x_start:x_end] += error_tensor[
                                                                                          batch, kernel, i, j] * \
                                                                                      self.weights[kernel]

            if m_kernel > 1 or n_kernel > 1:
                grad_input = grad_input[:, :, pad_y:-pad_y, pad_x:-pad_x]

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return grad_input

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_bias = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)
