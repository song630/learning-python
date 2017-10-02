from network import Network


class GradientCheck:
    def __init__(self, network):
        self.expected_grad = []
        self.actual_grad = []
        self.network = network

    def calc_error(self, sample, label):
        """
        if not self.network.trained:
            print('training undone.')
            sys.exit()
        """
        # preparations:
        self.network.predict(sample)
        self.network.calc_delta(label)
        self.network.calc_gradient()
        # preparations done.
        print('preparations done.')
        i = 0
        for conn in self.network.connections.connections:  # traverse every connection
            self.actual_grad.append(conn.get_gradient())
            epsilon = 0.0001
            conn.weight += epsilon
            error_inc = 0.5 * sum(map(lambda x: (x[0] - x[1]) * (x[0] - x[1]),
                                      zip(self.network.predict(sample), label)))
            if not i:
                print('error_inc = {}'.format(error_inc))
                print(self.network)
            conn.weight -= 2 * epsilon
            error_dec = 0.5 * sum(map(lambda x: (x[0] - x[1]) * (x[0] - x[1]),
                                      zip(self.network.predict(sample), label)))
            if not i:
                print('error_dec = {}'.format(error_dec))
                print(self.network)
            self.expected_grad.append((error_dec - error_inc) / (2.0 * epsilon))
            i += 1
        print('expected gradient:')
        print(self.expected_grad)
        print('actual gradient:')
        print(self.actual_grad)
