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
            self.layers.append(Layer(i, len(j)).add_nodes(j))
            # then fill connections:
            if i == 0:  # the input layer
                continue
            else:
                temp_conn = [Connection(m, n) for m in self.layers[i].nodes for n in self.layers[i - 1].nodes]
                self.connections.add_connections(temp_conn)
                for m in temp_conn:  # call functions of Node:
                    m.up_node.add_downstream_connection(i)
                    m.down_node.add_upstream_connection(i)
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
            for i in groups:  # traverse every group of sample
                self.layers[0].set_output(i)  # input layer
                for j in self.layers:  # every node at each level compute its output
                    j.calc_outputs()
                # then compute delta:
                for m, n in enumerate(reversed(self.layers)):
                    if m == 0:  # output layer
                        n.calc_delta_output_layer(labels[i])  # output layer: update delta
                    elif m == self.n_layers - 1:  # input layer
                        continue
                    else:  # hidden layers
                        n.calc_delta_hidden_layer()  # hidden layers: update delta
                # update weights and bias:
                for m, n in enumerate(reversed(self.layers)):
                    if m == 0:  # output layer
                        n.update_bias(rate)  # output layer: update bias
                    elif m == self.n_layers - 1:  # input layer
                        n.update_weights(self.n_layers, rate)  # update weights of connections between levels
                    else:  # hidden layers
                        n.update_weights(self.n_layers, rate)  # update weights of connections between levels
                        n.update_bias(rate)  # hidden layer: update bias
            iterations -= 1
        self.trained = True

    def predict(self, sample):
        self.layers[0].set_output(i)  # input layer
        for j in self.layers:  # every node at each level compute its output
            j.calc_outputs()
        return list(map(lambda x: x.output, self.layers[-1].nodes))  # get outputs of output layer

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
