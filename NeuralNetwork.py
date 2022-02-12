# -*- coding: utf-8 -*-

import math

from Neuron import Neuron
from NeuronLayer import NeuronLayer
from consts import EPOCHS
from consts import LEARNING_RATE
from consts import WINDOW_SIZE


class NeuralNetwork:

    def __init__(self, topology):
        self.topology = topology
        self.layers = []
        self.create_layers()

    def create_layers(self):
        self._create_input_layer()
        self._create_hidden_layers()
        self._create_output_layers()

    def _create_input_layer(self):
        input_neurons = []
        for i in range(self.topology.get_input_count()):
            neuron = Neuron(1, "Input")
            input_neurons.append(neuron)
        layer = NeuronLayer(input_neurons)
        self.layers.append(layer)

    def _create_hidden_layers(self):
        for hidden_layer_count in self.topology.get_hidden_layer():
            hidden_neurons = []
            last_layer_count = len(self.layers[-1].neurons)
            for i in range(hidden_layer_count):
                neuron = Neuron(last_layer_count)
                hidden_neurons.append(neuron)
            hidden_layer = NeuronLayer(hidden_neurons)
            self.layers.append(hidden_layer)

    def _create_output_layers(self):
        output_neurons = []
        last_layer_count = len(self.layers[-1].neurons)
        for i in range(self.topology.get_output_count()):
            neuron = Neuron(last_layer_count, "Output")
            output_neurons.append(neuron)
        output_layer = NeuronLayer(output_neurons)
        self.layers.append(output_layer)

    def feed_forward(self, inputs):
        # Инпуты
        for i in range(len(inputs)):
            self.layers[0].neurons[i].feed_forward([inputs[i]])

        # Скрытые и выходной
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            prev_layer_signal = self.layers[i - 1].get_signals()
            for neuron in layer.neurons:
                neuron.feed_forward(prev_layer_signal)
        return self.layers[-1].neurons

    def learn(self, kohonen_outputs_windows, learning_rate=LEARNING_RATE, epochs=EPOCHS):
        sko = 0
        target_arr = []
        rmse_arr = []
        expected_arr = []
        for epoch in range(epochs):
            sko = 0
            window_size = len(kohonen_outputs_windows[0])
            for window_index in range(len(kohonen_outputs_windows) - 1):

                target = kohonen_outputs_windows[window_index]
                expected = [kohonen_outputs_windows[window_index + 1][window_size - 1]]
                sko += math.sqrt(self._back_propagation(expected, target) / (window_index + 1))
                expected_arr.append(expected)
                target_arr.append(target)

            print("epoch: {} - {}".format(epoch, math.sqrt(sko / (EPOCHS - 1))))
            rmse_arr.append(math.sqrt(sko / (EPOCHS - 1)))

        return math.sqrt(sko / (EPOCHS - 1)), rmse_arr, target_arr, expected_arr

    def test(self, kohonen_outputs_windows, data_list, window_size=WINDOW_SIZE):
        sko = 0
        prediction = []
        result = []
        real = []
        for window_index in range(len(kohonen_outputs_windows) - 1):
            window_last_part = []
            for i in range(1, window_size):
                current_window = kohonen_outputs_windows[window_index]

                if len(window_last_part) > 0:
                    current_window = current_window[:-i - 1]
                    current_window += window_last_part

                neuron = self.feed_forward(current_window)[0]
                #neuron.output = self.denormalize_data(data_list['max'], data_list['min'], neuron.output)
                if len(window_last_part) == 0:
                    result.append(neuron.output)

                print("Prediction = %f, Real = %f, diff = %f" % (neuron.output,
                                                                 kohonen_outputs_windows[window_index][i],
                                                                 abs(kohonen_outputs_windows[window_index][
                                                                         i] - neuron.output)))
                #prediction.append(neuron.output)
                #real.append(kohonen_outputs_windows[window_index][i])
                sko += math.pow(abs(kohonen_outputs_windows[window_index][i] - neuron.output), 2)
                window_last_part.append(neuron.output)
                sko /= (len(kohonen_outputs_windows) - 1)
                sko = math.sqrt(sko)
        return sko, prediction, real

    def denormalize_data(self, max, min, norm):
        return norm * (max - min) + min

    # Обратное распространение ошибки
    def _back_propagation(self, expected_values, inputs):
        differences = []
        for index in range(len(expected_values)):
            actual_layer = self.feed_forward(inputs)[index]
            actual = actual_layer.output
            expected = expected_values[index]
            difference = actual - expected
            differences.append(difference)

            # Нейроны последнего слоя учатся по difference
            self.layers[-1].neurons[index].learn(difference)

        for i in range(len(self.layers) - 2, 0, -1):
            layer = self.layers[i]
            prev_layer = self.layers[i + 1]
            for j in range(len(layer.neurons)):
                neuron = layer.neurons[j]
                for k in range(len(prev_layer.neurons)):
                    prev_neuron = prev_layer.neurons[k]
                    error = prev_neuron.weights[j] * prev_neuron.delta
                    # Нейроны скрытого слоя по error
                    neuron.learn(error)
        return sum([(difference ** 2) for difference in differences])

    def __str__(self):
        return "Topology = {}\n" \
               "Layers = {}\n" \
               "\n".format(self.topology,
                           [str(layer) for layer in self.layers])
