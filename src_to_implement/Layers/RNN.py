from .Base import BaseLayer
import numpy as np
from .Sigmoid import Sigmoid
from .TanH import TanH
from .FullyConnected import FullyConnected


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = [np.zeros((1, self.hidden_size))]
        self._memorize = False
        self._optimizer = None
        self._gradient_weights = None

        self.sigma = Sigmoid()
        self.tanh = TanH()
        self.fcl_1 = FullyConnected(input_size + hidden_size, hidden_size)
        self.fcl_2 = FullyConnected(hidden_size, output_size)

    def initialize(self, weights_initializer, bias_initializer):
        self.fcl_1.initialize(weights_initializer, bias_initializer)
        self.fcl_2.initialize(weights_initializer, bias_initializer)
        self.weights = self.fcl_1.weights
        self.weights_2 = self.fcl_2.weights

    def forward(self, input_tensor):
        if not self.memorize:
            self.hidden_state = [np.zeros((1, self.hidden_size))]

        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]
        self.output = np.zeros((self.batch_size, self.output_size))

        for i in range(self.batch_size):
            x_tilde = np.concatenate((input_tensor[i, :], self.hidden_state[-1]), axis=None).reshape(1, -1)
            u = self.fcl_1.forward(x_tilde)
            new_hidden_state = self.tanh.forward(u)
            self.hidden_state.append(new_hidden_state)
            o = self.fcl_2.forward(new_hidden_state)  # Compute the output for the current time step
            self.output[i] = self.sigma.forward(o)

        return self.output

    def backward(self, error_tensor):
        self._gradient_weights = np.zeros(self.fcl_1.weights.shape)
        self.gradient_weights_2 = np.zeros(self.fcl_2.weights.shape)
        output_error = np.zeros((self.batch_size, self.input_size))
        hidden_error = np.zeros((1, self.hidden_size))

        # Backpropagation through time
        for t in reversed(range(error_tensor.shape[0])):    # reverse traversal every time step
            x_tilde = np.concatenate((self.input_tensor[t, :], self.hidden_state[t]), axis=None).reshape(1, -1)
            u = self.fcl_1.forward(x_tilde)
            h = self.tanh.forward(u)
            o = self.fcl_2.forward(h)
            output = self.sigma.forward(o)

            grad_output = self.sigma.backward(error_tensor[t, :])
            grad = self.fcl_2.backward(grad_output) + hidden_error  # hidden_output
            self.gradient_weights_2 += self.fcl_2.gradient_weights

            grad = self.tanh.backward(grad)     # hidden
            grad = self.fcl_1.backward(grad)    # input_hidden
            self._gradient_weights += self.fcl_1.gradient_weights

            output_error[t, :] = grad[:, :self.input_size]  # Accumulate output error for input tensor
            hidden_error = grad[:, self.input_size:]    # pass the hidden error to next time step

        if self.optimizer:
            self.fcl_1.weights = self.optimizer.calculate_update(
                self.fcl_1.weights, self._gradient_weights)
            self.fcl_2.weights = self.optimizer.calculate_update(
                self.fcl_2.weights, self.gradient_weights_2)

        self.weights = self.fcl_1.weights
        self.weights_2 = self.fcl_2.weights

        return output_error

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def weights(self):
        return self.fcl_1.weights

    @weights.setter
    def weights(self, weights):
        self.fcl_1.weights = weights

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, new_weights):
        self.fcl_1._gradient_weights = new_weights

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
