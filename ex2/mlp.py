import numpy as np
import csv
# import os

default_eta = 0.2

# epochs= 5000


def sigmoid(x, derivate=False):
    """Sigmoid activation function.
    """
    if (derivate):
        return x * (1 - x)
    else:
        return 1 / (1 + np.exp(-x))


class perceptron(object):
    """Class representing a Perceptron. """

    def __init__(self, size, activation=sigmoid, eta=default_eta):
        """
        Intialize the Perceptron class.
        weights: weight array for incoming signals
        activation : activation function
        """
        self.weights = np.random.rand(size)
        self.weights = 2 * self.weights - 1
        self.activation = activation

    def perceptron_output(self, input):
        """
        Generate the perceptron output applying the weighted input on activation function.
        """
        return self.activation(np.dot(self.weights, input))


class perceptron_layer(object):
    """A perceptron layer class containing perceptrons. """

    def __init__(self, size, input_signals, activation=sigmoid):
        """Intialize a perceptron layer
        size: number of perceptrons in the layer
        activation: activation function of each perceptron
        previous_layer: number of input signals entering in each perceptron
        """
        self.perceptrons = []
        for i in range(0, size):
            self.perceptrons.append(perceptron(input_signals, activation))


class mlp(object):
    """ Multi-Layer Perceptron Network class. """

    def __init__(self, input, hidden, hidden_layers, output, tolerance):
        """Intialize a MLP network. 
        input : Number of input signals
        hidden : Number of neurons in each hidden layer
        hidden_layers : Number of hidden layers 
        output : number of output signals
        tolerance : how good the accuracy of the network should be
        """
        self.layers = []
        previous_layer = input
        for i in range(0, hidden_layers):
            self.layers.append(perceptron_layer(hidden, previous_layer))
            previous_layer = hidden

        self.layers.append(perceptron_layer(output, previous_layer))

    def train(self, input_patterns):
        """Train the MLP network.
        """
        for current_pattern in input_patterns.data:
            # Feedforward
            input = np.asarray(current_pattern.input)
            for layer in self.layers:
                layer_output = []
                for perceptron in layer.perceptrons:
                    layer_output.append(perceptron.perceptron_output(input))
                input = np.asarray(layer_output)
            print(layer_output)

        return


class pattern(object):
    """Docstring for pattern. """

    def __init__(self):
        """TODO: to be defined1. """
        self.input = []
        self.output = []


class data_objects(object):
    """Docstring for data_structure. """

    def __init__(self, input_num, output_num, path):
        """TODO: to be defined1. """
        self.data = []
        with open(path, "r") as csvfile:
            spamreader = csv.DictReader(csvfile)
            for row in spamreader:
                aux_pattern = pattern()
                for i, name in enumerate(row):
                    if (i < input_num + output_num - 1):
                        aux_pattern.input.append(int(row[name]))
                    else:
                        aux_pattern.output.append(int(row[name]))

                self.data.append(aux_pattern)

    def test(self):
        """
        Print read values for testing purposes
        """
        for i in self.data:
            print(str(i.input))
            print(str(i.output))


a = data_objects(3, 1, "./bluetooth.csv")
b = mlp(3, 4, 2, 1, 0)
b.train(a)

