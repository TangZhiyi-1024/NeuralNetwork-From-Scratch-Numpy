import numpy as np
from Layers.Base import BaseLayer

# reduce the dimensionality of the input and therefore also decrease memory consumption
# reduce overfitting by introducing a degree of scale and translation invariance


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()      # call base class constructor
        self.trainable = False
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        b, c, y, x = input_tensor.shape
        y_out = (y - self.pooling_shape[0]) // self.stride_shape[0] + 1
        x_out = (x - self.pooling_shape[1]) // self.stride_shape[1] + 1
        output_data = np.zeros((b, c, y_out, x_out))

        for batch in range(b):
            for channel in range(c):
                for i in range(y_out):
                    for j in range(x_out):
                        data_slice = input_tensor[batch, channel,
                                     i * self.stride_shape[0]:(i * self.stride_shape[0] + self.pooling_shape[0]),
                                     j * self.stride_shape[1]:(j * self.stride_shape[1] + self.pooling_shape[1])]
                        output_data[batch, channel, i, j] = np.max(data_slice)

        return output_data

    def backward(self, error_tensor):
        b, c, y_out, x_out = error_tensor.shape
        output_error = np.zeros(self.input_tensor.shape)

        for batch in range(b):
            for channel in range(c):
                for i in range(y_out):
                    for j in range(x_out):
                        data_slice = self.input_tensor[batch, channel,
                                     i * self.stride_shape[0]:(i * self.stride_shape[0] + self.pooling_shape[0]),
                                     j * self.stride_shape[1]:(j * self.stride_shape[1] + self.pooling_shape[1])]
                        indices = np.where(data_slice == np.max(data_slice))
                        output_error[batch, channel,
                                     i * self.stride_shape[0] + indices[0],
                                     j * self.stride_shape[1] + indices[1]] += error_tensor[batch, channel, i, j]

        return output_error
