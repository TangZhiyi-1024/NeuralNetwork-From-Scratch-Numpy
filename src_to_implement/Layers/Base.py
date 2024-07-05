class BaseLayer:
    def __init__(self, input_size=None, output_size=None):
        self.trainable = False  # initialize a boolean member trainable with False
        self.testing_phase = False
