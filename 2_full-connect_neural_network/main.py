from network import Network
from gradient_check import GradientCheck


if __name__ == '__main__':
    layers = [[1, 2, 3], [4, 5], [6]]  # 3 layers
    net = Network(layers)  # build up a network
    groups = [[0, 1, 0],
              [1, 1, 1],
              [0, 1, 1],
              [0, 0, 0],
              [1, 1, 0]]
    labels = [[0.3], [0.7], [0.5], [0.0], [0.5]]
    iterations = 20
    rate = 0.1
    net.train(groups, labels, iterations, rate)
    rst = net.predict([1, 0, 1])
    print('result:')
    print(rst)
    print('info on network just built:')
    print(net)

    # ===== begin gradient check:
    # layers = [[1, 2, 3], [4, 5], [6]]  # 3 layers
    # net = Network(layers)  # build up a network
    # check = GradientCheck(net)
    # check.calc_error([0.2, 0.5, 0.3], [0.4])
