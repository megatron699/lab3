# -*- coding: utf-8 -*-

import random as rnd

import math

from consts import LEARNING_RATE, RANDOM_SEED, ALPHA_MOMENTO

rnd.seed(RANDOM_SEED)


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        if x > 0:
            return 1 / (1 + math.exp(-float('inf')))
        else:
            return 1 / (1 + math.exp(float('inf')))


def sigmoid_dx(x):
    sigm = sigmoid(x)
    return sigm * (1 - sigm)


class Neuron:
    neuron_type_enum = {"Input": 0, "Normal": 1, "Output": 2}

    def __init__(self, count, neuron_type="Normal"):
        # Тип нейронa
        self.neuron_type = neuron_type
        self.neuron_type_enum = self.neuron_type_enum[neuron_type]

        # Выходной результат
        self.output = 0

        # Веса
        if self.neuron_type == "Input":
            self.weights = [1 for i in range(count)]
            self.last_weights = [1 for i in range(count)]
        else:
            self.weights = [rnd.uniform(-1, 1) for i in range(count)]
            self.last_weights = [0 for i in range(count)]

        # Для обучения
        self.inputs = [0 for i in range(count)]
        self.delta = 0

    def feed_forward(self, inputs):
        self.inputs = inputs

        sum_list = [x * y for x, y in zip(self.weights, inputs)]
        if self.neuron_type != "Input":
            self.output = sigmoid(sum(sum_list))
        else:
            self.output = sum(sum_list)
        return self.output

    def learn(self, error):
        if self.neuron_type == "Input":
            return
        self.delta = error * sigmoid_dx(self.output)

        # Обучение
        for i in range(len(self.weights)):
            new_weight = self.weights[i] - self.inputs[i] * self.delta * LEARNING_RATE + \
                         ALPHA_MOMENTO*(self.weights[i]-self.last_weights[i])
            self.last_weights[i] = self.weights[i]
            self.weights[i] = new_weight


    def __str__(self):
        return "Inputs = {}\n" \
               "Neuron type = {}({})\n" \
               "Weights = {}\n" \
               "Output = {}\n" \
               "\n".format([float('%.2f' % elem) for elem in self.inputs],
                           self.neuron_type, self.neuron_type_enum,
                           [float('%.2f' % elem) for elem in self.weights],
                           self.output)


print("start")
if __name__ == "__main__":
    # Количество входов и тип нейрона
    neuron = Neuron(10, "Input")
    neuron.feed_forward([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(neuron)
    print("end")
