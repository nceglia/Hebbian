import numpy
from math import exp
import math

class Neuron(object):
    """Neuron class"""
    def __init__(self, rate, sigmoid, inputs):
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

    @staticmethod
    def logistic(value):
        """Logistic function returns sigmoid output"""
        sig_output = 1.0 / (1.0 + exp(-1.0 * float(value)))
        return sig_output

    @staticmethod
    def threshold(activation):
        """Threshold function returns threshold output"""
        if activation >= 0.0:
            return 1.0
        else:
            return -1.0

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
    def sigmoid_select(fxn, value):
        """
        Selects sigmoid function to be applied to weights after update rule

        Keyword arguments:
        fxn -- function to be selected (int)
        value -- weight value (float)

        returns sigmoid of value.
        """
        sigmoid_value = 0
        if fxn == 1:
            sigmoid_value = Neuron.algebraic_abs(value)
        elif fxn == 2:
            sigmoid_value = Neuron.tanh(value)
        elif fxn == 3:
            sigmoid_value = Neuron.algebraic_sqrt(value)
        elif fxn == 0:
            sigmoid_value = value
        return sigmoid_value

    def algorithm_select(self, algorithm, activation, example, weight):
        """
        Selects update rule to calculate change of weight

        Keyword arguments:
        algorithm -- name of update rule (str)
        activation -- post synaptic activation of neuron (float)
        example -- pre synaptic input to neuron (float)
        weight -- current weight value for input edge (float)

        returns change of weight (delta)
        """
        example = float(example)
        delta  = 0
        if algorithm == "hebbian":
            delta = self.rate + activation * example
        elif algorithm == "oja":
            delta = self.rate * activation * (example - activation * weight)
        else:
            print "Incorrect Weight rule!"
            exit(0)
        return delta

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
            example[i] = float(example[i])
            delta = self.algorithm_select(algorithm, activation, example[i], weight)
            if delta != 0.0:
                weight = weight+delta
                self.weights[i] = Neuron.sigmoid_select(self.sigmoid, weight)
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
            delta = self.algorithm_select(algorithm, clamp, float(example[i]), weight)
            if delta != 0.0:
                weight = weight + delta
                self.weights[i] = Neuron.sigmoid_select(self.sigmoid, weight)

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

    def rerandomize(self):
        """randomizes half of the incoming edge weights between -1 and 1"""
        pass

    def increase_learning(self, factor):
        """Increases the current learning rate by 'factor'"""
        pass

    def noise(self, stddev):
        """Adds gaussian noise to incoming edge weights with std dev"""
        pass