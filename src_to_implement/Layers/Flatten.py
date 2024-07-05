class Flatten:
    def __init__(self):
        # Indicate that this layer does not have parameters to train
        self.original_shape = None
        self.trainable = False

    def forward(self, input_tensor):
        # Store the original shape to use it in the backward pass
        self.original_shape = input_tensor.shape
        # Flatten the input tensor to a one-dimensional feature vector
        return input_tensor.reshape(self.original_shape[0], -1)

    def backward(self, error_tensor):
        # Reshape the error tensor back to the original input tensor shape
        return error_tensor.reshape(self.original_shape)