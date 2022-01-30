# -*- coding: utf-8 -*-

class Topology:
    def get_input_count(self):
        return self.input_count

    def get_output_count(self):
        return self.output_count

    def get_hidden_layer(self):
        return self.layers

    def get_hidden_layer_count(self):
        return len(self.layers)

    def __init__(self, layers):
        self.input_count = layers[0]
        self.layers = layers[1:-1]
        self.output_count = layers[-1]

    def __str__(self):
        return "{}, {}, {}".format(self.input_count, self.layers, self.output_count)
