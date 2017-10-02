from node import *
from connection import *
from connections import *
from layer import *


class Network:
    def __init__(self, layers):
        """
        :param layers: a 2-dimension list, stores indexes of input layer in layers[0], etc.
        """
        self.connections = Connections()  # call Connections.__init__(). it is an object, not a list.
        self.layers = []  # stores Layer objects
        for i, j, in enumerate(layers):  # j: a list storing all node indexes at a layer
            # first build a layer:
            temp_layer = Layer(i, len(j))
            temp_layer.add_nodes(j)
            self.layers.append(temp_layer)
            # then fill connections:
            if i == 0:  # the input layer
                continue
            else:
                conns = [Connection(m, n) for m in self.layers[i - 1].nodes for n in self.layers[i].nodes]
                for m in conns:  # call functions of Node:
                    self.connections.add_connection(m)
                    m.up_node.add_downstream_connection(m)
                    m.down_node.add_upstream_connection(m)
            # ===== error: forget to fill inputs of nodes: =====
            for m in self.layers[i].nodes:
                m.fill_inputs()
        self.n_layers = len(layers)
        self.trained = False

    def train(self, groups, labels, iterations, rate):
        """
        :param groups: 2-dimension list
        :param labels: 2-dimension list
        :param iterations:
        :param rate:
        """
        while iterations > 0:
            for i, j in enumerate(groups):  # traverse every group of sample
                # first compute output:
                self.layers[0].set_output(j)  # input layer
                for m in self.layers[1:]:  # every node at each level computes its output
                    m.calc_outputs()
                # then compute delta:
                for m, n in enumerate(reversed(self.layers)):
                    if m == 0:  # output layer
                        n.calc_delta_output_layer(labels[i])  # output layer: update delta
                    elif m == self.n_layers - 1:  # input layer
                        continue
                    else:  # hidden layers
                        n.calc_delta_hidden_layer()  # hidden layers: update delta
                # update weights and bias:
                # ===== error: not in reversed order =====
                for m, n in enumerate(self.layers):
                    if m == self.n_layers - 1:  # output layer
                        n.update_bias(rate)  # output layer: update bias
                    elif m == 0:  # input layer
                        n.update_weights(self.n_layers, rate)  # update weights of connections between levels
                    else:  # hidden layers
                        n.update_weights(self.n_layers, rate)  # update weights of connections between levels
                        n.update_bias(rate)  # hidden layer: update bias
            iterations -= 1
        self.trained = True

    # ================================
    def predict(self, sample):  # +++++ used for gradient check +++++
        self.layers[0].set_output(sample)  # input layer
        for j in self.layers[1:]:  # every node at each level compute its output
            j.calc_outputs()
        return list(map(lambda x: x.output, self.layers[-1].nodes))  # get outputs of output layer

    def calc_delta(self, label):  # +++++ used for gradient check +++++
        self.layers[-1].calc_delta_output_layer(label)  # output layer
        for i in reversed(self.layers[1:-1]):
            i.calc_delta_hidden_layer()  # hidden layers

    def calc_gradient(self):  # +++++ used for gradient check +++++
        for i in self.layers[:-1]:
            for j in i.nodes:
                for k in j.output_connections:
                    k.calc_gradient()
    # ================================

    def __str__(self):
        for i, j in enumerate(self.layers):  # traverse every layer
            for m in j.nodes:  # traverse every node at a layer
                print(m)  # print a node
                if i == self.n_layers - 1:  # output layer
                    continue
                else:
                    for n in m.output_connections:
                        print('\t', n)  # print a connection
        return ''
