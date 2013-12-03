from math import exp
import math
from boolean import BOOLEAN
from Neuron import Neuron

class Network(object):
    """Network class"""
    def __init__(self, rate, sigmoid, hidden, examples, variables, layers, rule, dropout):
        """
        Feed-Forward Hebbian network for learning boolean functions
        with threshold gates.

        Keyword arguments:
        rate -- learning rate (float)
        sigmoid -- sigmoid function for weights if rule is basic hebbian (int)
        hidden -- number of hidden units, 0 removes hidden layer (int)
        examples -- number of random boolean examples to present.
        layers -- number of hidden layers 1 to N (int)
        rule -- learning rule, "hebbian" or "oja" (str)

        Initializes layers, weights, connections, variables for Network class
        """
        self.rate = rate
        self.sigmoid = sigmoid
        self.inputs = variables 
        self.vis_layer = []
        self.hidden_layers = []
        self.hidden = hidden
        self.variables = variables
        self.data = BOOLEAN(examples, self.variables)
        self.layers = layers-1
        self.rule = rule
        self.dropout = dropout
        self.length = int(math.pow(2, self.variables))
        for _ in xrange(self.hidden):
            self.vis_layer.append(Neuron(self.rate, self.sigmoid, self.inputs+1, dropout))
        for layer in xrange(self.layers):
            self.hidden_layers.append([])
            for _ in xrange(self.hidden):
                self.hidden_layers[layer].append(Neuron(self.rate, self.sigmoid, self.hidden+1, dropout))
        if self.hidden > 0:
            self.output_neuron = Neuron(self.rate, self.sigmoid, self.hidden+1, dropout)
        else:
            self.output_neuron = Neuron(self.rate, self.sigmoid, self.inputs+1, dropout)

    @staticmethod
    def threshold(activation):
        """
        Thresholds output neuron activation to 1 or 0.

        Keyword arguments:
        activation -- Output neuron activation (float)

        returns 1 or 0
        """
        if activation >= 0.0:
            return 1
        else:
            return 0

    def load(self, filename):
        """
        Loads a model file.

        Keyword arguments:
        filename -- name of model file ex: hebb1.txt (str)

        initializes network to weights in file
        """
        hebbian_weights = open(filename, "r").read().split('\n')
        for i in xrange(self.hidden):
            weights = hebbian_weights[i].split('\t')
            self.vis_layer[i].set_weights(weights)
        for i in xrange(self.layers):
            for j in xrange(self.hidden):
                weights = hebbian_weights[((i+1)*self.hidden)+j].split('\t')
                self.hidden_layers[i][j].set_weights(weights)
        weights = hebbian_weights[-2].split('\t')
        self.output_neuron.set_weights(weights)

    def save(self, filename):
        """
        Saves a model into a file.

        Keyword arguments:
        filename -- name of modle file ex: hebb1.txt (str)

        saves current model weights to file
        """
        hebbian_weights = open(filename, "w")
        for i in xrange(self.hidden):
            hebbian_weights.write("\t".join(self.vis_layer[i].get_weights()) + '\n')
        for i in xrange(self.layers):
            for j in xrange(self.hidden):
                hebbian_weights.write("\t".join(self.hidden_layers[i][j].get_weights()) + '\n')
        hebbian_weights.write("\t".join(self.output_neuron.get_weights()) + '\n')
        hebbian_weights.close()

    def compute(self, example):
        """
        Computes output of model given an example.

        Keyword arguments:
        example -- list of 1 and -1 (list)

        returns threshold value of output neuron.
        """
        activations = []
        if self.hidden > 0:
            for i in xrange(self.hidden):
                output = self.vis_layer[i].compute(example)
                activations.append(output)
            activations.append(1.0)
            for layer in xrange(self.layers):
                hidden_activations = []
                for i in xrange(self.hidden):
                    hidden_activations.append(self.hidden_layers[layer][i].compute(activations))
                hidden_activations.append(1.0)
                activations = hidden_activations
            output = self.output_neuron.compute(activations)
        else:
            output = self.output_neuron.compute(example)
        return Network.threshold(output)

    def index(self, example):
        """
        Finds expected output of example for target function

        Args:
        example (list): list of 1 and -1 

        returns index in current truthtable for given example.
        """
        for i in xrange(self.length):
            binary = bin(i).lstrip('0b')
            for i in xrange(self.variables-len(binary)):
                binary = '0'+binary
            for j in xrange(self.variables):
                index = True
                if example[j] == -1:
                    example[j] = 0
                if int(binary[j]) != example[j]:
                    index = False
                if index:
                    return i
        return False


    def train(self, table):
        """
        Trains model given rule for given examples on a single truth table

        Args:
        table (list): truth table of 1, -1 


        returns true if learned and false if not
        """
        training = self.data.load_training()
        for example in training:
            activations = []
            if self.hidden > 0:
                for i in xrange(self.hidden):
                    output = self.vis_layer[i].train(example, self.rule)
                    activations.append(output)
                activations.append(1.0)
                for layer in xrange(self.layers):
                    hidden_activations = []
                    for i in xrange(self.hidden):
                        hidden_activations.append(self.hidden_layers[layer][i].train(activations, self.rule))
                    hidden_activations.append(1.0)
                    activations = hidden_activations
                self.output_neuron.clamp(activations, table[self.index(example)], self.rule)
            else:
                self.output_neuron.clamp(example, table[self.index(example)], self.rule)
            learned_table = self.truthtable()
            learned = True
            for i in xrange(self.length):
                if learned_table[i] != table[i]:
                    learned = False
            if learned == True:
                break
        return learned

    def truthtable(self):
        """Builds the truth table for the current model and returns it"""
        table = []
        for i in xrange(self.length):
            inputs = []
            binary = bin(i).lstrip('0b')
            for i in xrange(len(binary)):
                inputs.append(int(binary[i]))
            inputs.append(1)
            table.append(self.compute(inputs))
        return table

    def test(self, load_file):
        """Loads a model from file and prints the table that model produces"""
        self.load(load_file)
        table = self.truthtable()
        print " .....  Test: Model Computes:", table


    def increase_learning(self, factor):
        """Increase the learning rate to by 'factor'"""
        pass

    def noise(self, stddev):
        """Add gaussian noise to all weights with stddev"""
        #add noise to weights
        pass
