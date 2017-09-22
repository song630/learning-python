import sys


class Perceptron:
    """
    defines a single perceptron.
    private members:
    __vectors[], __n_inputs, __n_groups, __labels[], __rate
    """
    def __init__(self, num_inputs, num_groups, activator):
        """
        set bias and all weights 0.
        :param num_inputs: num of 'x's
        :param num_groups: num of training groups
        """
        self.n_inputs = num_inputs
        self.n_groups = num_groups
        self.weights = [0 for i in range(num_inputs)]  # also [0] * num_inputs
        self.bias = 0  # b
        self.trained = False
        self.activator = activator  # activation function

    def __str__(self):
        return "weights: {}\nbias: {}".format(list(self.weights), self.bias)

    def update(self, delta_weights, delta_b):  # update bias and weights[]
        self.weights = list(map(lambda x: x[0] + x[1], zip(self.weights, delta_weights)))  # add the two lists
        self.bias += delta_b

    def training(self, vectors, labels, iterations=10, rate=0.1):
        while iterations > 0:
            for i in range(self.n_groups):  # deal with all groups of inputs and y
                # x[0], x[1]: 0 and 1 denotes the order of parameters in zip()
                # compute weights * vectors[i]
                y = sum(map(lambda x: x[0] * x[1], zip(self.weights, vectors[i]))) + self.bias
                y = self.activator(y)
                # formula: delta_weights[i] = rate * (label - y) * x[i]
                delta_weights = [rate * (labels[i] - y) * j for j in vectors[i]]
                delta_b = rate * (labels[i] - y)
                self.update(delta_weights, delta_b)  # update
            iterations -= 1
        self.trained = True

    def predict(self, inputs):  # after being trained, compute y of the given 'x's
        if not self.trained:  # has not been trained
            sys.exit()
        y = sum(map(lambda x: x[0] * x[1], zip(self.weights, inputs))) + self.bias
        print(self.activator(y))
