# -*- coding: utf-8 -*-
import random

import numpy
import numpy as np

from NeuralNetwork import NeuralNetwork
from Topology import Topology
from consts import *
from Kohonen import *
import matplotlib.pyplot as plt
import pandas
import sys
import consts

def get_data_set(rows_to_skip, rows_count):
    data_frame = pandas.read_csv('exchange_rates.csv', delimiter=',', skiprows=rows_to_skip, nrows=rows_count,
                                 usecols=['Time Period', 'RXI$US_N.B.EU', 'RXI$US_N.B.UK'], names=['Time Period', 'RXI$US_N.B.EU', 'RXI$US_N.B.UK'])
    array = numpy.array(data_frame['RXI$US_N.B.UK'].dropna())
    #plt.plot(array)
  #  plt.show()
   # print(array)
    return array


def normalize_and_divide(data, learn_percentage=LEARN_PERCENTAGE):
  #  data_list = {'min': min(data), 'max': max(data)}
   # data_list['data'] = [(item - data_list['min']) / (data_list['max'] - data_list['min']) for item in data]
    last_learn_item = int(len(data) * learn_percentage)

    return data, {
        'learn': data[:last_learn_item],
        'test': data[last_learn_item:]
    }


def learn_hybrid_kohonen_network(kohonen_layer_network, mlp_network, learn_data, epochs=EPOCHS):
    print('Kohonen train:')
    kohonen_layer_network.learn(EPOCHS_KOHONEN)
    print('Normalized outputs from kohonen layer for train:')
    kohonen_layer_learn_outputs = kohonen_layer_network.get_normalize_outputs(learn_data)
    print(kohonen_layer_learn_outputs)

    print('MLP train')
    learn_error, rmse_arr, a, b = mlp_network.learn(kohonen_layer_learn_outputs, epochs=epochs)
    plt.plot(a, 'b')
    plt.plot(b, 'r')
    plt.show()
    print('Train error: {}'.format(learn_error))
    plt.plot(rmse_arr)
    plt.show()
    return kohonen_layer_learn_outputs


def test_hybrid_kohonen_network(kohonen_layer_network, mlp_network, test_data, data_list, window_size=WINDOW_SIZE):
    print('Normalized outputs from kohonen layer for test:')
    kohonen_layer_test_outputs = kohonen_layer_network.test(test_data)
    print('Outputs: ' + str(kohonen_layer_test_outputs))
    rmse, prediction, real = mlp_network.test(kohonen_layer_test_outputs, data_list, window_size=window_size)
    print('Test error: %f' % rmse)
    # print(prediction)
    # print(real)
    plt.plot(real)
    plt.plot(prediction)
    plt.show()
    return kohonen_layer_test_outputs, rmse

def train_and_test(learn_percentage=LEARN_PERCENTAGE, alpha_kohonen=ALPHA_KOHONEN,
                   neurons_count_kohonen=NEURONS_COUNT_KOHONEN, epochs=EPOCHS, window_size=WINDOW_SIZE, hidden_neurons_count=NEURONS_COUNT_HIDDEN):
    data_list, data = normalize_and_divide(get_data_set(ROWS_TO_SKIP, ROWS_COUNT), learn_percentage)
    kohonen_layer_network = Kohonen(data['learn'], alpha_kohonen, neurons_count_kohonen, window_size)
    NN_LAYERS = [neurons_count_kohonen, hidden_neurons_count, 3]
    mlp_network = NeuralNetwork(Topology(NN_LAYERS))
    kohonen_layer_learn_outputs = learn_hybrid_kohonen_network(kohonen_layer_network, mlp_network, data['learn'], epochs)
    kohonen_layer_test_output, rmse = test_hybrid_kohonen_network(kohonen_layer_network, mlp_network, data['test'], data_list,
                                                                  window_size=window_size)
    return rmse

def test_alpha():
    alpha_min = 0.05
    alpha_max = 0.51
    step = 0.05
    error_arr = []
    alpha_arr = np.arange(alpha_min, alpha_max, step)
    #orig = consts.ALPHA_KOHONEN
    for a in alpha_arr:
       # consts.ALPHA_KOHONEN = a
        test_rmse = train_and_test(alpha_kohonen=a)
      #  print(learn_error)
        print(test_rmse)
        error_arr.append(test_rmse)
    #consts.ALPHA_KOHONEN = orig
    return alpha_arr, error_arr

def test_kohonen_neurons_count():
    kohonen_neurons_count_min = 2
    kohonen_neurons_count_max = 11
    step = 1
    error_arr = []
    neurons_count_arr = np.arange(kohonen_neurons_count_min, kohonen_neurons_count_max, step)
    #orig = consts.NEURONS_COUNT_KOHONEN
    for a in neurons_count_arr:
       # consts.NEURONS_COUNT_KOHONEN = a
        test_error = train_and_test(neurons_count_kohonen=a, window_size=a)
        print(test_error)
        error_arr.append(test_error)
  #  consts.NEURONS_COUNT_KOHONEN = orig
    return neurons_count_arr, error_arr

def test_hidden_neurons_count():
    kohonen_neurons_count_min = 1
    kohonen_neurons_count_max = 10
    step = 1
    error_arr = []
    neurons_count_arr = np.arange(kohonen_neurons_count_min, kohonen_neurons_count_max, step)
    #orig = consts.NEURONS_COUNT_KOHONEN
    for a in neurons_count_arr:
       # consts.NEURONS_COUNT_KOHONEN = a
        test_error = train_and_test(hidden_neurons_count=a)
        print(test_error)
        error_arr.append(test_error)
  #  consts.NEURONS_COUNT_KOHONEN = orig
    return neurons_count_arr, error_arr

def test_learn_size():
    learn_size_min = 0.55
    learn_size_max = 0.96
    step = 0.05
    error_arr = []
    learns_size_arr = np.arange(learn_size_min, learn_size_max, step)

    #orig = consts.LEARN_PERCENTAGE
    for a in learns_size_arr:
       # consts.LEARN_PERCENTAGE = a
        print(a)
        test_error = train_and_test(learn_percentage=a)
        print(test_error)
        error_arr.append(test_error)
   # consts.LEARN_PERCENTAGE = orig
    return learns_size_arr, error_arr

def test_epochs_count():
    epochs_min = 20
    epochs_max = 201
    step = 20
    error_arr = []
    epochs_arr = np.arange(epochs_min, epochs_max, step)

    orig = consts.EPOCHS
    for a in epochs_arr:
        consts.EPOCHS = a
        test_error = train_and_test(epochs=a)
        print(test_error)
        error_arr.append(test_error)
    consts.EPOCHS = orig
    return epochs_arr, error_arr

if __name__ == '__main__':
    maxmin, data = normalize_and_divide(get_data_set(ROWS_TO_SKIP, ROWS_COUNT), LEARN_PERCENTAGE)
    y = []
    for x in data['learn']:
       y.append(x + random.uniform(-0.01, 0.01))

    plt.plot(data['learn'])
    plt.plot(y)
    plt.show()

    z = []
    for x in data['test']:
        z.append(x + random.uniform(-0.01, 0.01))
    plt.plot(data['test'])
    plt.plot(z)
    plt.show()
    sys.exit(0)
    kohonen_layer_network = Kohonen(data['learn'], ALPHA_KOHONEN, NEURONS_COUNT_KOHONEN, WINDOW_SIZE)
    mlp_network = NeuralNetwork(Topology(NN_LAYERS))
    kohonen_layer_learn_outputs = learn_hybrid_kohonen_network(kohonen_layer_network, mlp_network, data['learn'],
                                                               epochs=consts.EPOCHS)
    kohonen_layer_test_output = test_hybrid_kohonen_network(kohonen_layer_network, mlp_network, data['test'], maxmin)

    # alpha_arr, error_arr = test_alpha()
    # plt.xlabel('ALPHA')
    # plt.ylabel('RMSE')
    # plt.plot(alpha_arr, error_arr)
    # plt.show()

    # kohonen_neurons_count, error_arr = test_kohonen_neurons_count()
    # plt.xlabel('Kohonen neurons count')
    # plt.ylabel('RMSE')
    # plt.plot(kohonen_neurons_count, error_arr)
    # plt.show()

    # kohonen_neurons_count, error_arr = test_hidden_neurons_count()
    # plt.xlabel('Hidden neurons count')
    # plt.ylabel('RMSE')
    # plt.plot(kohonen_neurons_count, error_arr)
    # plt.show()
    # #
    # learn_size, error_arr = test_learn_size()
    # plt.xlabel('learn_size')
    # plt.ylabel('error')
    # plt.plot(learn_size, error_arr)
    # plt.show()

    # epochs, error_arr = test_epochs_count()
    # plt.xlabel('epochs_count')
    # plt.ylabel('error')
    # plt.plot(epochs, error_arr)
    # plt.show()