import sys
sys.path.append(r'D:\code_blocks\python\2_full-connect_neural_network')

# import the loader classes:
from Loader import Loader
from ImageLoader import ImageLoader
from LabelLoader import LabelLoader

# import network classes:
from network import Network


def get_training_data_set():
    image_loader = ImageLoader(r'D:\code_blocks\python\train-images.idx3-ubyte', 60000)
    label_loader = LabelLoader(r'D:\code_blocks\python\train-labels.idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()


def get_test_data_set():
    image_loader = ImageLoader(r'D:\code_blocks\python\t10k-images.idx3-ubyte', 10000)
    label_loader = LabelLoader(r'D:\code_blocks\python\t10k-labels.idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()


def get_result(output):  # output: get from network's output layer
    return output.index(max(output))  # 0-9


def error_evaluate(network, test_samples, test_labels):
    error = 0.0
    for i in range(10000):  # altogether 10000 test samples
        actual_rst = get_result(network.predict(test_samples[i]))  # 0-9
        standard_rst = get_result(test_labels[i])
        if actual_rst != standard_rst:
            error += 1
    return error / 10000.0


def train_and_evaluate(network, train_samples, train_labels, test_samples, test_labels):
    print('begin with training.')
    error1 = 1.0
    counter = 0
    while True:
        network.train(train_samples, train_labels, 10, 0.3)
        counter += 10
        print('trained for {} rounds.'.format(counter))
        error2 = error_evaluate(network, test_samples, test_labels)
        print('result (error) of evaluation: {}'.format(error2))
        if error2 < error1:
            break
        else:
            error1 = error2  # update
            continue


if __name__ == '__main__':
    print('program begins.')
    input_layer = [i for i in range(784)]
    hidden_layer = [(i + 784) for i in range(300)]
    output_layer = [(i + 1084) for i in range(10)]
    network = Network([input_layer, hidden_layer, output_layer])
    print('finished building network.')
    del input_layer, hidden_layer, output_layer
    training = get_training_data_set()  # return a 2-elements tuple
    print('finished loading training data set.')
    test = get_test_data_set()  # return a 2-elements tuple
    print('finished loading test data set.')
    train_and_evaluate(network, training[0], training[1], test[0], test[1])
