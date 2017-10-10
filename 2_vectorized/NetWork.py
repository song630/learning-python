import FullConnectedLayer


class NetWork:
    def __init__(self, layers, rate):  # layers: a 2-dimension array
        self.layers = []
        self.rate = rate
        # ===== do not construct input layer =====
        for i, j in enumerate(layers[1:]):
            self.layers.append(FullConnectedLayer(len(layers[i]), len(j)))

    def train(self, samples, labels, iterations):  # call the 3 functions below
        for i in iterations:
            for j in range(len(samples)):
                self.predict(samples[j])
                self.calc_delta(labels[j])
                self.update()

    def predict(self, sample):  # sample should be a matrix
        next_input = self.layers[0].calc_output(sample)  # the first hidden layer
        for i in self.layers[1:]:
            next_input = i.calc_output(next_input)
        return next_input

    def calc_delta(self, label):  # reversly
        self.layers[-1].calc_delta_output_layer(label)
        delta = self.layers[-1].delta
        for i in reversed(self.layers[1:-1]):
            i.calc_delta_hidden_layer(delta)
            delta = i.delta  # update

    def update(self):
        for i in self.layers:
            i.weight_bias_update(self.rate)
