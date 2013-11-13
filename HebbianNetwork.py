"""
Module script for constructing and training hebbian feed forward neural networks
for n variable boolean functions.
"""
import argparse
import numpy
from math import exp
import math
from boolean import BOOLEAN

def truth_tables(variables):
    """
    Builds a list of truth tables for n variable boolean functions

    Keyword arguments:
    variables -- boolean variables (float)

    returns list of truth tables
    """
    tables = []
    for i in range(int(math.pow(2, math.pow(2, variables)))):
        binary = bin(i).lstrip('0b')
        table = []
        for i in range(int(math.pow(2, variables)-len(binary))):
            binary = '0' + binary
        for digit in binary:
            table.append(int(digit))
        tables.append(table)
    return tables

def monotone(table, variables):
    """
    Determines wether a truth table for a boolean function is monotonic

    Keyword argument:
    table -- truth table (list)
    variables -- n variable boolean function

    returns true if function is monotonic and false if not.
    """
    monotone_table = True
    for i in range(len(table)):
        for j in range(i + 1, len(table)):
            first_number = []
            second_number = []
            binary = str(bin(i).lstrip('0b'))
            for _ in range(variables - len(binary)):
                binary = '0' + binary
            for index in range(len(binary)):
                first_number.append(int(binary[index]))
            binary = str(bin(j).lstrip('0b'))
            for _ in range(variables - len(binary)):
                binary = '0' + binary
            for index in range(len(binary)):
                second_number.append(int(binary[index]))
            if boolean_compare(first_number, second_number):
                if table[i] > table[j]:
                    monotone_table = False
    return monotone_table

def monotone_truth_tables(variables):
    """
    Finds monotonic truth tables for n variable boolean functions

    Keyword arguments:
    variables -- n variable boolean functions (float)

    returns list of monotonic truth tables
    """
    tables = truth_tables(variables)
    monotone_tables = []
    for table in tables:
        if monotone(table, variables) and table != []:
            monotone_tables.append(table)
    return monotone_tables

def bit_repr(table):
    """
    Takes a table as list and returns binary representation

    Keyword arguments:
    table -- truth table for boolean function (list)

    returns an integer representation of table (ex: 10101)
    """
    bits = ""
    for bit in table:
        bits  = bits + str(bit)
    return int(bits)

def boolean_compare(input1, input2):
    """
    Compares if one binary number is montonically greater than the second.

    Keyword arguments:
    input1 -- binary number as list (list)
    input2 -- binary number as list (list)

    returns true if monotonically increasing, false if not.
    """
    less_than = True
    for digit1, digit2 in zip(input1, input2):
        if digit1 > digit2:
            less_than = False
    return less_than

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
        self.mean_change = []

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
            self.mean_change.append(delta)
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


class Network(object):
    """Network class"""
    def __init__(self, rate, sigmoid, hidden, examples, variables, layers, rule):
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
        for _ in range(self.hidden):
            self.vis_layer.append(Neuron(self.rate, self.sigmoid, self.inputs+1))
        for layer in range(self.layers):
            self.hidden_layers.append([])
            for _ in range(self.hidden):
                self.hidden_layers[layer].append(Neuron(self.rate, self.sigmoid, self.hidden+1))
        if self.hidden > 0:
            self.output_neuron = Neuron(self.rate, self.sigmoid, self.hidden+1)
        else:
            self.output_neuron = Neuron(self.rate, self.sigmoid, self.inputs+1)

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
        for i in range(self.hidden):
            weights = hebbian_weights[i].split('\t')
            self.vis_layer[i].set_weights(weights)
        for i in range(self.layers):
            for j in range(self.hidden):
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
        for i in range(self.hidden):
            hebbian_weights.write("\t".join(self.vis_layer[i].get_weights()) + '\n')
        for i in range(self.layers):
            for j in range(self.hidden):
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
            for i in range(self.hidden):
                output = self.vis_layer[i].compute(example)
                activations.append(output)
            activations.append(1.0)
            for layer in range(self.layers):
                hidden_activations = []
                for i in range(self.hidden):
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

        Keyword arguments:
        example -- list of 1 and -1 (list)

        returns index in current truthtable for given example.
        """
        for i in range(int(math.pow(2, self.variables))):
            binary = bin(i).lstrip('0b')
            for i in range(self.variables-len(binary)):
                binary = '0'+binary
            for j in range(self.variables):
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

        Keyword arguments:
        table -- truth table of 1, -1 (list)


        returns true if learned and false if not
        """
        training = self.data.load_training()
        for example in training:
            activations = []
            if self.hidden > 0:
                for i in range(self.hidden):
                    output = self.vis_layer[i].train(example, self.rule)
                    activations.append(output)
                activations.append(1.0)
                for layer in range(self.layers):
                    hidden_activations = []
                    for i in range(self.hidden):
                        hidden_activations.append(self.hidden_layers[layer][i].train(activations, self.rule))
                    hidden_activations.append(1.0)
                    activations = hidden_activations
                self.output_neuron.clamp(activations, table[self.index(example)], self.rule)
            else:
                self.output_neuron.clamp(example, table[self.index(example)], self.rule)
            learned_table = self.truthtable()
            learned = True
            for i in range(int(math.pow(2, self.variables))):
                if learned_table[i] != table[i]:
                    learned = False
            if learned == True:
                break
        return learned

    def truthtable(self):
        """Builds the truth table for the current model and returns it"""
        table = []
        table_len = int(math.pow(2.0, self.variables))
        for i in range(table_len):
            inputs = []
            binary = bin(i).lstrip('0b')
            for i in range(len(binary)):
                inputs.append(int(binary[i]))
            inputs.append(1)
            table.append(self.compute(inputs))
        return table

    def test(self, load_file):
        """Loads a model from file and prints the table that model produces"""
        self.load(load_file)
        table = self.truthtable()
        print " .....  Test: Model Computes:", table

    def rerandomize(self):
        """Randomize a percentage of weights in network"""
        #randomize a random number of weights
        pass

    def increase_learning(self, factor):
        """Increase the learning rate to by 'factor'"""
        pass

    def noise(self, stddev):
        """Add gaussian noise to all weights with stddev"""
        #add noise to weights
        pass

def run(rate, sigmoid, hidden, examples, variables, layers, rule):
    """
    Creates network and trains a model for each boolean function

    Keyword arguments:
    rate -- learning rate (float)
    sigmoid -- sigmoid function for weights if rule is basic hebbian  (int)
    hidden -- number of hidden units, 0 removes hidden layer (int)
    examples -- number of random boolean examples to present.
    layers -- number of hidden layers 1 to N (int)
    rule -- learning rule, "hebbian" or "oja" (str)

    prints each function, whether it was able to learn it, and a summary.
    """
    functions = []
    example = 0
    monotone_fxns = 0
    tables = truth_tables(variables)
    #tables = monotone_truth_tables(variables)

    not_learned = ""
    for i in range(len(tables)):
        print "Learning", tables[i],
        example += 1
        learned = False
        tries = 0
        while not learned and tries < 1000:
            tries += 1
            model = Network(rate, sigmoid, hidden, examples, variables, layers, rule)
            learned = model.train(tables[i])
        if learned:
            print "Learned",
            functions.append(bit_repr(tables[i]))
            if monotone(tables[i], variables):
                monotone_fxns += 1
            model.save("models/hebb{0}.txt".format(example))
            model.test("models/hebb{0}.txt".format(example))
        else:
            not_learned = not_learned+str(i)+","
            print "Not Learned"
    return len(functions), monotone_fxns, not_learned

def main():
    """Parses command line args and calls run method"""

    parser = argparse.ArgumentParser(prog="Hebbian Network")
    parser.add_argument("--rate", type=float)
    parser.add_argument("--sigmoid", type=int)
    parser.add_argument("--examples", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--variables", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--rule", type=str)
    args = parser.parse_args()

    print run(args.rate, args.sigmoid, args.hidden, args.examples, args.variables, args.layers, args.rule)

if __name__ == "__main__":
    main()







