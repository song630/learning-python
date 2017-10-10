import numpy as np
import sys


def sigmoid(x):
    return 1.0 / (1 + np.exp(x))


class FullConnectedLayer:
    def __init__(self, num_inputs, num_outputs):
        self.inputs = np.zeros((num_inputs, 1))  # n * 1 matrix
        self.outputs = np.zeros((num_outputs, 1))  # n * 1 matrix
        self.W = np.random.uniform(-0.1, 0.1, (num_outputs, num_inputs))  # weights
        self.bias = np.random.uniform(-0.1, 0.1, (num_outputs, 1))
        self.delta = np.zeros((num_outputs, 1))

    def calc_output(self, inputs):  # inputs: get from up layer, a matrix
        if self.W.shape[1] != inputs.shape[0]:  # cannot perform a dot
            print('illegal parameter inputs {}.'.format(inputs))
            sys.exit()
        self.outputs = sigmoid(np.dot(self.W, inputs) + self.bias)

    def calc_delta_output_layer(self, target):  # target: label
        self.delta = self.outputs * (1 - self.outputs) * (target - self.outputs)

    def calc_delta_hidden_layer(self, delta_next_layer):
        self.delta = self.outputs * (1 - self.outputs) * np.dot(self.W.T, delta_next_layer)

    def weight_bias_update(self, rate):
        self.W += rate * np.dot(self.delta, self.inputs.T)
        self.bias += rate * self.delta
