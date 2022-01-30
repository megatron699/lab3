# -*- coding: utf-8 -*-

class NeuronLayer:
    def __init__(self, neurons):
        self.neurons = neurons

    def get_signals(self):
        return [neuron.output for neuron in self.neurons]

    def __str__(self):
        return "Neurons = {}\n" \
               "\n".format([str(neuron) for neuron in self.neurons])

