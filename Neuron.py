import numpy
from math import exp
import math
import random

class Neuron(object):
    """Neuron class"""
    def __init__(self, rate, sigmoid, inputs, dropout):
        """
        Neuron class constructor

        Keyword arguments:
        rate -- learning rate (float)
        sigmoid -- if learning rule is hebb, sigmoid function for weights (int)
        inputs -- number of edges into neuron (int)

        returns initialized Neuron object.
        """
        self.rate = rate
        self.weights = numpy.random.uniform(-1.0, 1.0, size=inputs)
        self.sigmoid = sigmoid
        self.inputs = inputs
        self.dropout = dropout
        self.rule_dict = {'hebbian': self.hebb_rule,
                    'oja': self.oja_rule}
        self.sigmoid_dict = { 1: Neuron.algebraic_abs,
                              2: Neuron.tanh,
                              3: Neuron.algebraic_sqrt,
                              0: Neuron.no_sigmoid,
                            }

    @staticmethod
    def logistic(value):
        """Logistic function returns sigmoid output"""
        sig_output = 1.0 / (1.0 + exp(-1.0 * float(value)))
        return sig_output

    @staticmethod
    def threshold(activation):
        """Threshold function returns threshold output"""
        if activation > 0.0:
            return 1.0
        elif activation < 0.0:
            return -1.0
        else:
            return 0.0

    @staticmethod
    def algebraic_abs(value):
        """Algebraic function returns sigmoid output"""
        return value/(1.0 + math.fabs(value))

    @staticmethod
    def tanh(value):
        """Tanh function returns sigmoid output"""
        return math.tanh(value)

    @staticmethod
    def algebraic_sqrt(value):
        """Algebraic function returns sigmoid output"""
        return value / math.sqrt(1.0 + math.pow(value, 2.0))

    @staticmethod
    def no_sigmoid(value):
        return value

    def hebb_rule(self, activation, example, weight):
        return self.rate + activation * example

    def oja_rule(self, activation, example, weight):
        return self.rate * activation * (example - activation * weight)


    def train(self, example, algorithm):
        """
        Calculates linear activation of neuron and updates incoming weights

        Keyword arguments:
        example -- input to neuron (list)
        algorithm -- name of update rule for weights (str)

        returns the threshold activation of the neuron
        """
        activation = self.compute(example)
        for i, weight in enumerate(self.weights):
            possibility = numpy.random.uniform(-1.0, 1.0, size=1)[0]
            if possibility < self.dropout:
                example[i] = float(example[i])
                delta = self.rule_dict[algorithm](activation, example[i], weight)
                if delta != 0.0:
                    weight = weight+delta
                    self.weights[i] = self.sigmoid_dict[self.sigmoid] (weight)
        return Neuron.threshold(activation)

    def clamp(self, example, clamp, algorithm):
        """
        Updates incoming weights of neuron for a clamped activation

        Keyword arguments:
        example  -- input to neuron (list)
        clamp -- clamped neuron output (float)
        algorithm -- name of update rule for weights (str)

        updates weights, no return value
        """
        if clamp == 0.0:
            clamp = -1.0
        for i, weight in enumerate(self.weights):
            possibility = numpy.random.uniform(-1.0, 1.0, size=1)[0]
            if possibility < self.dropout:
                delta = self.rule_dict[algorithm](clamp, float(example[i]), weight)
                if delta != 0.0:
                    weight = weight + delta
                    self.weights[i] = self.sigmoid_dict[self.sigmoid] (weight)

    def compute(self, inputs):
        """
        Computes linear activation of neuron

        Keyword arguments:
        inputs -- input to neuron (list)

        returns the threshold activation of the neuron
        """
        activation = 0.0
        for inp, weight in zip(inputs, self.weights):
            activation += float(inp) * float(weight)
        return Neuron.threshold(activation)

    def get_weights(self):
        """Returns current weights for incoming edges to neuron"""
        return_weights = []
        for weight in self.weights:
            return_weights.append(str(weight))
        return return_weights

    def set_weights(self, hard_weights):
        """Sets weights for incoming edges to neuron"""
        input_weights = []
        for weight in hard_weights:
            input_weights.append(float(weight))
        self.weights  = input_weights

    def increase_learning(self, factor):
        """Increases the current learning rate by 'factor'"""
        self.rate = self.rate * factor
        pass




