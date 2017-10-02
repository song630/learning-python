from node import *
from connection import *


class Layer:
    def __init__(self, layer_index, n_nodes):
        self.layer_index = layer_index  # input layer if 'layer_index' == 0
        self.n_nodes = n_nodes
        self.nodes = []

    def add_nodes(self, node_indexes):
        """
        create Node objects and append them in.
        :param node_indexes: a list
        """
        self.nodes = [Node(self.layer_index, i) for i in node_indexes]
        # sort nodes[] according to their node_index:
        # ===== cmp function not supported in Python 3 =====
        sorted(self.nodes, key=lambda x: x.node_index)

    def calc_outputs(self):
        """
        computes all nodes' output value at a layer
        """
        if self.layer_index == 0:
            print('wrong function to call.')
            return
        else:
            for i in self.nodes:
                i.calc_output()

    def set_output(self, x):  # for input layer. x[] is a list
        if self.n_nodes != len(x):
            print('number of nodes does not match the list {}.'.format(x))
            sys.exit()
        for i, j in enumerate(self.nodes):  # i: index
            j.set_output(x[i])

    def calc_delta_output_layer(self, labels):
        # the same function in Node object will automatically check layer
        if self.n_nodes != len(labels):
            print('number of nodes does not match the list {}.'.format(labels))
            sys.exit()
        for i, j in enumerate(self.nodes):  # i: index
            j.calc_delta_output_layer(labels[i])

    def calc_delta_hidden_layer(self):
        # the same function in Node object will automatically check layer
        for i in self.nodes:  # i: index
            i.calc_delta_hidden_layer()

    def update_weights(self, total_layers, rate):  # update weights of connections connecting this level with next
        if self.layer_index == total_layers - 1:  # output layer
            print('cannot update weights for connections starting from output layer.')
            sys.exit()
        for i in self.nodes:  # traverse every node at this layer
            for j in i.output_connections:
                j.update_weight(rate)

    def update_bias(self, rate):
        # after computing delta
        if self.layer_index == 0:
            print('cannot update bias for input layer.')
            sys.exit()
        for i in self.nodes:
            i.update_bias(rate)
