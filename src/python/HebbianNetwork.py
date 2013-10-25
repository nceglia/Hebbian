import sys
import os
import argparse
import numpy
from math import exp
import math
from boolean import BOOLEAN
import gflags


def truth_tables(variables):
    tables = []
    for i in range(int(math.pow(2,math.pow(2,variables)))):
        binary = bin(i).lstrip('0b')
        table = []
        for i in range(int(math.pow(2,variables)-len(binary))):
            binary = '0'+binary
        for digit in binary:
            table.append(int(digit))
        tables.append(table)
    return tables

def monotone(table, variables):
    monotone = True
    for i in range(len(table)): 
        for j in range(i+1,len(table)):
            first_number = []
            second_number = []
            binary = str(bin(i).lstrip('0b'))
            for k in range(variables-len(binary)):
                binary = '0'+binary
            for index in range(len(binary)):
                first_number.append(int(binary[index]))
            binary = str(bin(j).lstrip('0b'))
            for k in range(variables-len(binary)):
                binary = '0'+binary
            for index in range(len(binary)):
                second_number.append(int(binary[index]))
            if booleanCompare(first_number,second_number):
                if table[i] > table[j]:
                    monotone = False
    return monotone   

def monotone_truth_tables(variables):
    tables = truth_tables(variables)
    monotone_tables = []
    for table in tables:
        if monotone(table,variables) and table != []:
            monotone_tables.append(table)
    return monotone_tables

def bit_repr(table):
    bits = ""
    for bit in table:
        bits  = bits + str(bit)
    return int(bits)

def booleanCompare(input1,input2):
    lessThan = True
    for digit1,digit2 in zip(input1,input2):
        if digit1 > digit2:
            lessThan = False
    return lessThan

class Neuron(object):
    def __init__(self,rate,sigmoid,inputs):
        self.rate = rate
        self.weights = numpy.random.uniform(-1.0,1.0,size=inputs)
        self.sigmoid = sigmoid
        self.inputs = inputs

    @staticmethod
    def logistic(value): 
        try:
            sig_output = 1.0/(1.0+exp(-1.0*float(value)))
        except:
            if value > 0:
                sig_output = 1.0
            else:
                sig_output = 0.0 
        return sig_output

    @staticmethod
    def threshold(activation):
        if activation >= 0.0:
            return 1.0
        else:
            return -1.0

    @staticmethod
    def algebraic_abs(value):
        return value/(1.0+math.fabs(value))

    @staticmethod
    def tanh(value):
        return math.Neuron.tanh(value)

    @staticmethod
    def algebraic_sqrt(value):
        return value/math.sqrt(1.0 + math.pow(value,2.0))

    @staticmethod
    def sigmoid_select(fxn,value):
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

    def algorithm_select(self,algorithm,activation,example,weight):
        delta  = 0
        if algorithm == "hebbian":
            delta = self.rate + activation * float(example)
        elif algorithm == "oja":
            delta = self.rate * activation * (float(example) - activation * weight)
        else:
            print "Incorrect Weight Update!"
            exit(0)
        return delta

    def train(self,example,algorithm):
        activation = self.compute(example)  
        for i, weight in enumerate(self.weights):
            delta = self.algorithm_select(algorithm,activation,float(example[i]),weight)
            if delta != 0.0:
                weight = weight+delta
                self.weights[i] = Neuron.sigmoid_select(self.sigmoid,weight)
        return Neuron.threshold(activation)

    def selective(self,example,expected,algorithm):
        activation = self.compute(example)
        train = False  
        if expected  == -1.0 and activation < 0.0:
            train = True
        if expected  == 1.0 and activation >= 0.0:
            train = True
        if train:
            for i, weight in enumerate(self.weights):
                delta = self.algorithm_select(algorithm,activation,float(example[i]),weight)
                if delta != 0.0:
                    weight = weight+delta
                    self.weights[i] = Neuron.sigmoid_select(self.sigmoid,weight)
        return Neuron.threshold(activation)        

    def clamp(self,example,clamp,algorithm):
        if clamp == 0.0:
            clamp = -1.0
        for i, weight in enumerate(self.weights):
            delta = self. algorithm_select(algorithm,clamp,float(example[i]),weight)
            if delta != 0.0:
                weight = weight+delta
                self.weights[i] = Neuron.sigmoid_select(self.sigmoid,weight)

    def compute(self,inputs):
        activation = 0.0
        for inp,weight in zip(inputs,self.weights):
            activation += float(inp)*float(weight)
        return Neuron.threshold(activation)
    
    def get_weights(self):
        return_weights = []
        for weight in self.weights:
            return_weights.append(str(weight))
        return return_weights

    def set_weights(self,hard_weights):
        input_weights = []
        for weight in hard_weights:
            input_weights.append(float(weight))
        self.weights  = input_weights


class Network(object):
    def __init__(self,rate,sigmoid,hidden,examples,variables,layers,algorithm,update):
        self.rate = rate
        self.sigmoid = sigmoid
        self.inputs = variables
        self.vis_layer = []
        self.hidden_layers = []
        self.hidden = hidden
        self.variables = variables
        self.data = BOOLEAN(examples,self.variables)
        self.layers = layers-1
        self.algorithm = algorithm
        self.update = update
        for i in range(self.hidden):
            self.vis_layer.append(Neuron(self.rate,self.sigmoid,self.inputs+1))
        for layer in range(self.layers):
            self.hidden_layers.append([])
            for i in range(self.hidden):
                self.hidden_layers[layer].append(Neuron(self.rate,self.sigmoid,self.hidden+1))
        if self.hidden > 0:
            self.output_neuron = Neuron(self.rate,self.sigmoid,self.hidden+1)
        else:
            self.output_neuron = Neuron(self.rate,self.sigmoid,self.inputs+1)
        self.activations = []

    @staticmethod
    def threshold(activation):
        if activation >= 0.0:
            return 1
        else:
            return 0   

    def load(self,filename):
        hebbian_weights = open(filename,"r").read().split('\n')
        for i in range(self.hidden):
            weights = hebbian_weights[i].split('\t')
            self.vis_layer[i].set_weights(weights)
        for i in range(self.layers):
            for j in range(self.hidden):
                weights = hebbian_weights[(i*self.hidden)+j].split('\t')
                self.hidden_layers[i][j].set_weights(weights)
        weights = hebbian_weights[-2].split('\t')
        self.output_neuron.set_weights(weights)

    def save(self,filename):
        hebbian_weights = open(filename,"w")
        for i in range(self.hidden):
            hebbian_weights.write("\t".join(self.vis_layer[i].get_weights()) + '\n')
        for i in range(self.layers):
            for j in range(self.hidden):
                hebbian_weights.write("\t".join(self.hidden_layers[i][j].get_weights()) + '\n')
        hebbian_weights.write("\t".join(self.output_neuron.get_weights()) + '\n')
        hebbian_weights.close()

    def compute(self,example):
        self.activations = []
        if self.hidden > 0:
            for i in range(self.hidden):
                output = self.vis_layer[i].compute(example)
                self.activations.append(output)
            self.activations.append(1.0)
            for layer in range(self.layers):
                hidden_activations = []
                for i in range(self.hidden):
                    hidden_activations.append(self.hidden_layers[layer][i].compute(self.activations))
                hidden_activations.append(1.0)
                self.activations = hidden_activations
            output = self.output_neuron.compute(self.activations) 
        else:
            output = self.output_neuron.compute(example)
        return Network.threshold(output)  

    def index(self,example):
        for i in range(int(math.pow(2,self.variables))):
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


    def train(self,table):
        self.training = self.data.load_training()
        for num, example in enumerate(self.training):
            self.activations = []
            if self.hidden > 0:
                for i in range(self.hidden):
                    output = self.vis_layer[i].train(example,self.update)
                    self.activations.append(output)
                self.activations.append(1.0) #bias term
                for layer in range(self.layers):
                    hidden_activations = []
                    for i in range(self.hidden):
                        hidden_activations.append(self.hidden_layers[layer][i].train(self.activations,self.update))
                    hidden_activations.append(1.0)
                    self.activations = hidden_activations
                if self.algorithm == "clamp":
                    self.output_neuron.clamp(self.activations,table[self.index(example)],self.update)
                elif self.algorithm == "selective":
                    self.output_neuron.selective(self.activations,table[self.index(example)],self.update)
                else:
                    print "Incorrect arguments for algorithm!"
                    exit(0)
            else:
                if self.algorithm == "clamp":
                    self.output_neuron.clamp(example,table[self.index(example)],self.update)
                elif self.algorithm == "selective":
                    self.output_neuron.selective(example,table[self.index(example)],self.update)
                else:
                    print "Incorrect arguments for algorithm!"
                    exit(0)   
            learned_table = self.truthtable()
            learned = True
            for i in range(int(math.pow(2,self.variables))):
                if learned_table[i] != table[i]:
                    learned = False
            if learned == True:
                break
        return learned

    def truthtable(self):
        table = []
        table_len = int(math.pow(2.0,self.variables))
        for i in range(table_len):
            inputs = []
            binary = bin(i).lstrip('0b')
            for i in range(len(binary)):
                inputs.append(int(binary[i]))
            inputs.append(1)
            table.append(self.compute(inputs))
        return table

    def test(self,load_file):
        self.load(load_file)
        table = self.truthtable()
        print " .....  Test: Model Computes:", table


def run(rate,sigmoid,hidden,examples,variables,layers,algorithm,update):
    functions = []
    example = 0
    monotone_fxns = 0
    previous = 0
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
            model = Network(rate,sigmoid,hidden,examples,variables,layers,algorithm,update)
            learned = model.train(tables[i])
        if learned:
            print "Learned",
            functions.append(bit_repr(tables[i]))
            if monotone(tables[i],variables):
                monotone_fxns += 1
            model.save("hebb{0}.txt".format(example))
            model.test("hebb{0}.txt".format(example))
        else:
            not_learned = not_learned+str(i)+","
            print "Not Learned"
    return len(functions), monotone_fxns, not_learned

def main():
    parser = argparse.ArgumentParser(prog="Hebbian Network")
    parser.add_argument("--rate",type=float)
    parser.add_argument("--sigmoid",type=int)
    parser.add_argument("--examples",type=int)
    parser.add_argument("--hidden",type=int)
    parser.add_argument("--variables",type=int)
    parser.add_argument("--layers",type=int)
    parser.add_argument("--algorithm",type=str)
    parser.add_argument("--update",type=str)
    args = parser.parse_args()

    print run(args.rate,
              args.sigmoid,
              args.hidden,
              args.examples,
              args.variables,
              args.layers,
              args.algorithm,
              args.update)

if __name__=="__main__":
    main()







