# from connection import Connection
import sys
import math
import random


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
        self.bias = random.uniform(-0.1, 0.1)  # b  # ===== ? =====

    def __str__(self):
        return 'node_index = {}, delta = {:.5f}, output = {:.5f}, ' \
               'bias = {:.5f}'.format(self.node_index, self.delta, self.output, self.bias)

    def __repr__(self):
        return self.__str__()

    def __cmp__(self, other):
        return cmp(self.node_index, other.node_index)

    def add_upstream_connection(self, connection):  # append an object to 'input_connections[]'
        self.input_connections.append(connection)

    def add_downstream_connection(self, connection):  # append an object to 'output_connections[]'
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
        # ===== error: x[0].weight + x[1] =====
        temp = sum(map(lambda x: x[0].weight * x[1].output, zip(self.input_connections, self.inputs))) + self.bias
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
        # ===== error: if len(output_connections): =====
        # ===== error: if output_connections: =====
        if not self.output_connections:  # the list is empty
            self.delta = self.output * (1.0 - self.output) * (label - self.output)
        else:  # output_connections != []: not output layer
            print('this is not the output layer.')
            sys.exit()

    # ===== add layer_index check later =====
    def calc_delta_hidden_layer(self):
        # compute 'output' and all down_node's delta first
        # fill in output_connections[] first
        if self.layer_index == 0 or self.output_connections == []:
            print("this is not a hidden layer.")
            sys.exit()
        res = sum(map(lambda x: x.weight * x.down_node.delta, self.output_connections))
        self.delta = self.output * (1 - self.output) * res
        #  print('index = {}, delta = {:.5f}'.format(self.node_index, self.delta))

    def update_bias(self, rate):
        if self.layer_index == 0:  # at the input layer
            print('cannot update bias of a node at input layer')
            sys.exit()
        self.bias += rate * self.delta
