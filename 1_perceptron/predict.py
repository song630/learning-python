from Perceptron import Perceptron
import sys


def step_fun(value):  # define an activation function
    return 1 if value > 0 else 0

if __name__ == '__main__':
    n_inputs = int(input('enter num of inputs: '))  # num of 'x's
    rate = float(input('enter the learning rate: '))  # eta
    num_training = int(input('enter num of trainings: '))
    vectors = []  # 2-dimension list
    labels = []  # the relative y value
    for i in range(num_training):  # traverse every training group
        vectors.append([])  # append a group
        print('enter values of inputs: ', end='')
        vectors[i] = [float(j) for j in input().split(' ')]
        if len(vectors[i]) != n_inputs:
            print('Invalid input.')
            sys.exit()
    print('enter the labels: ', end='')
    labels = [float(i) for i in input().split(' ')]  # input the relative y values of every group
    if len(labels) != num_training:
        print('Invalid input.')
        sys.exit()
    # input done.

    P = Perceptron(n_inputs, num_training, step_fun)
    P.training(vectors, labels)
    print('Results: ')
    print(P)
