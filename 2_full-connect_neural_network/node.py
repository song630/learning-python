from connection import *
import sys
import math


def sigmoid(x):
    return 1.0 / (1 + math.exp(x))


class Node:
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.input_connections = []  # record instances of Connection
        self.output_connections = []
        self.delta = 0.0  # measure the error
        self.inputs = []  # record objects of Node
        self.output = 0.0
        self.bias = 0.0  # b

    def __str__(self):
        return 'delta = {}, output = {}'.format(delta, output)

    def __repr__(self):
        return self.__str__()

    def add_upstream_connection(self, connection):  # append an object to 'input_connections[]'
        if connection in self.input_connections:
            print('connection {} is already in list.'.format(connection))
            sys.exit()
        self.input_connections.append(connection)

    def add_downstream_connection(self, connection):  # append an object to 'output_connections[]'
        if connection in self.output_connections:
            print('connection {} is already in list.'.format(connection))
            sys.exit()
        self.output_connections.append(connection)

    def fill_inputs(self):
        if not self.input_connections:
            print('fill in input connections first.')
            sys.exit()
        self.inputs = [i.up_node for i in self.input_connections]  # Connection.up_node
        # that ensures inputs[] and input_connections[] are matched

    def calc_output(self):
        # first fill in inputs[] and input_connections[]
        if self.layer_index == 0:
            print('this node is at the input layer.')
            return
        temp = sum(map(lambda x: x[0].weight + x[1], zip(self.input_connections, self.inputs))) + bias
        self.output = sigmoid(temp)

    def set_output(self, x):
        if self.layer_index > 0:
            print('this node is not at input layer.')
            sys.exit()
        else:
            self.output = x

    # ===== add layer_index check later =====
    def calc_delta_output_layer(self, label):  # label: the target value
        # compute 'output' first
        if not self.output_connections:  # output_connections != []: not output layer
            print('this is not the output layer.')
            sys.exit()
        self.delta = self.output * (1.0 - self.output) * (label - self.output)

    # ===== add layer_index check later =====
    def calc_delta_hidden_layer(self):
        # compute 'output' and all down_node's delta first
        # fill in output_connections[] first
        if self.layer_index == 0 or self.output_connections == []:
            print("this is not a hidden layer.")
            sys.exit()
        res = sum(map(lambda x: x.weight * x.down_node.delta, self.output_connections))
        self.delta = self.output * (1 - self.output) * res

    def update_bias(self, rate):
        self.bias += rate * self.delta
