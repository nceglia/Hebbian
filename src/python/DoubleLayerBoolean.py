
import sys
import os
import argparse
import numpy
from mnist import MNIST
from math import exp
import math
from boolean import BOOLEAN

def sigmoid(value): 
    try:
        sig_output = 1.0/(1.0+exp(-1.0*float(value)))
    except:
        if value > 0:
            sig_output = 1.0
        else:
            sig_output = 0.0
    return sig_output

def weights_fxn(value):
    return value/(1.0+math.fabs(value))

def tanh(value):
    return math.tanh(value)

def weights_sqrt(value):
    return value/math.sqrt(1.0 + math.pow(value,2.0))

def test_tonicity(table):
    monotone = False
    if table[1] >= table[0] and table[2] >= table[0]:
        if table[3] >= table[1] and table[3] >= table[2]:
            if table[4] >= table[0]:
                if table[5] >= table[4] and table[5] >= table[1]:
                    if table[6] >= table[4] and table[6] >= table[2]:
                        if table[7] >= table[6] and table[7] >= table[5] and table[7] >= table[3]:
                            monotone = True
    return monotone

def truth_tables():
    tables = []
    for a0 in [0,1]:
        for a1 in [0,1]:
            for a2 in [0,1]:
                for a3 in [0,1]:
                    for a4 in [0,1]:
                        for a5 in [0,1]:
                            for a6 in [0,1]: 
                                for a7 in [0,1]:
                                    tables.append([a0,a1,a2,a3,a4,a5,a6,a7])
    return tables


def bit_repr(table):
    bits = ""
    for bit in table:
        bits  = bits + str(bit)
    print bits
    return int(bits)


class Gate():
    def __init__(self,rate,sigmoid,inputs):
        self.rate = rate
        self.weights = numpy.random.uniform(-1.0,1.0,size=inputs)
        self.sigmoid = sigmoid
        self.inputs = inputs

    def train(self,example):
        activation = self.compute(example)
        for i, weight in enumerate(self.weights):
            delta = self.rate + activation * float(example[i])
            if delta != 0.0:
                weight = weight+delta
                if self.sigmoid == 1:
                    self.weights[i] = weights_fxn(weight)
                elif self.sigmoid == 2:
                    self.weights[i] = tanh(weight)
                elif self.sigmoid == 3:
                    self.weights[i] = weights_sqrt(weight)
        return activation

    def clamp(self,example,clamp):
        activation = self.compute(example)
        for i, weight in enumerate(self.weights):
            delta = self.rate + clamp * float(example[i])
            if delta != 0.0:
                weight = weight+delta
                if self.sigmoid == 1:
                    self.weights[i] = weights_fxn(weight)
                elif self.sigmoid == 2:
                    self.weights[i] = tanh(weight)
                elif self.sigmoid == 3:
                    self.weights[i] = weights_sqrt(weight)
        return activation

    def compute(self,inputs):
        activation = 0.0
        for inp,weight in zip(inputs,self.weights):
            activation += float(inp)*weight
        return activation
    
    def get_weights(self):
        return self.weights

class Network():
    def __init__(self,rate,sigmoid,inputs,hidden,examples,variables):
        self.rate = rate
        self.sigmoid = sigmoid
        self.inputs = inputs
        self.vis_layer = []
        self.hidden = hidden
        self.variables = variables
        data = BOOLEAN(examples,self.variables)
        print 'Loading {0} Var Boolean Data...'.format(self.variables)
        self.training = data.load_training()
        print 'Finished Loading'
        for i in range(self.hidden):
            self.vis_layer.append(Gate(self.rate,self.sigmoid,self.inputs+1))
        self.output_neuron = Gate(self.rate,self.sigmoid,self.hidden+1)
        self.activations = []

    def compute(self,example):
        activations = []
        for i in range(self.hidden):
            output = self.vis_layer[i].compute(example)
            self.activations.append(output)
        output = self.output_neuron.compute(self.activations) 
        if output >= 0.0:
            return 1
        else:
            return 0       

    def input_index(self,example):
        if example[0] == -1 and example[1] == -1 and example[1] == -1:
            return 0
        if example[0] == -1 and example[1] == -1 and example[1] == 1:
            return 1
        if example[0] == -1 and example[1] == 1 and example[1] == -1:
            return 2
        if example[0] == -1 and example[1] == 1 and example[1] == 1:
            return 3
        if example[0] == 1 and example[1] == -1 and example[1] == -1:
            return 4
        if example[0] == 1 and example[1] == -1 and example[1] == 1:
            return 5
        if example[0] == 1 and example[1] == 1 and example[1] == -1:
            return 6
        if example[0] == 1 and example[1] == 1 and example[1] == 1:
            return 7

    def train(self,table):
        for num, example in enumerate(self.training):
            self.activations = []
            for i in range(self.hidden):
                output = self.vis_layer[i].train(example)
                self.activations.append(output)
            self.activations.append(1) #bias term
            self.output_neuron.clamp(self.activations,table[self.input_index(example)])
        
        learned_table, monotone = self.truthtable()
        learned = True
        for i in range(8):
            if learned_table[i] != table[i]:
                learned = False
        return learned, monotone

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
        return table,test_tonicity(table)

def main():
    parser = argparse.ArgumentParser(prog="Hebbian Network")
    parser.add_argument("--rate",type=float)
    parser.add_argument("--sigmoid",type=int)
    parser.add_argument("--models",type=int)
    parser.add_argument("--examples",type=int)
    parser.add_argument("--hidden",type=int)
    parser.add_argument("--variables",type=int)
    args = parser.parse_args()
    functions = []
    examples = 0
    monotone_fxns = 0
    previous = 0
    tables = truth_tables()
    for i in range(256):
        examples+=1
        model = Network(args.rate,args.sigmoid,3,args.hidden,args.examples,args.variables)
        learned = False
        while not learned:
            learned, monotone = model.train(tables[i])
        if learned:
            fxn, monotone = model.truthtable()
            functions.append(bit_repr(fxn))
        if monotone:
            monotone_fxns += 1
        print "Example ", examples, len(functions), "unique variable fxns learned."
    print monotone_fxns, "Unique Monotone Functions Learned"

if __name__=="__main__":
    main()







