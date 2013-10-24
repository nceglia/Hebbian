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

    def threshold(self,activation):
        if activation >= 0.0:
            return 1.0
        else:
            return -1.0

    def train(self,example):
        activation = self.compute(example)  
        for i, weight in enumerate(self.weights):
            #delta = self.rate + activation * float(example[i])
            delta = self.rate * activation * (float(example[i]) - activation * weight)
            if delta != 0.0:
                weight = weight+delta
                if self.sigmoid == 1:
                    self.weights[i] = weights_fxn(weight)
                elif self.sigmoid == 2:
                    self.weights[i] = tanh(weight)
                elif self.sigmoid == 3:
                    self.weights[i] = weights_sqrt(weight)
                elif self.sigmoid == 0:
                    self.weights[i] = weight
        return activation

    def selective_train(self,example,expected):
        activation = self.compute(example)
        train = False  
        if expected  == -1.0 and activation < 0.0:
            train = True
        if expected  == 1.0 and activation >= 0.0:
            train = True
        if train:
            for i, weight in enumerate(self.weights):
                #delta = self.rate + activation * float(example[i])
                delta = self.rate * activation * (float(example[i]) - activation * weight)
                if delta != 0.0:
                    weight = weight+delta
                    if self.sigmoid == 1:
                        self.weights[i] = weights_fxn(weight)
                    elif self.sigmoid == 2:
                        self.weights[i] = tanh(weight)
                    elif self.sigmoid == 3:
                        self.weights[i] = weights_sqrt(weight)
                    elif self.sigmoid == 0:
                        self.weights[i] = weight
        return activation        

    def clamp(self,example,clamp):
        if clamp >= 0.0:
            clamp = 1.0
        else:
            clamp = -1.0
        for i, weight in enumerate(self.weights):
            #delta = self.rate + clamp * float(example[i])
            delta = self.rate * clamp * (float(example[i]) - clamp * weight)
            if delta != 0.0:
                weight = weight+delta
                if self.sigmoid == 1:
                    self.weights[i] = weights_fxn(weight)
                elif self.sigmoid == 2:
                    self.weights[i] = tanh(weight)
                elif self.sigmoid == 3:
                    self.weights[i] = weights_sqrt(weight)
                elif self.sigmoid == 0:
                    self.weights[i] = weight

    def compute(self,inputs):
        activation = 0.0
        for inp,weight in zip(inputs,self.weights):
            activation += float(inp)*weight
        return activation
    
    def get_weights(self):
        return self.weights

class Network(object):
    def __init__(self,rate,sigmoid,hidden,examples,variables,layers,algorithm):
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

    def compute(self,example):
        self.activations = []
        if self.hidden > 0:
            for i in range(self.hidden):
                output = self.vis_layer[i].compute(example)
                self.activations.append(output)
            self.activations.append(1.0)
            output = self.output_neuron.compute(self.activations) 
        else:
            output = self.output_neuron.compute(example)
        if output >= 0.0:
            return 1
        else:
            return 0     

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
                    output = self.vis_layer[i].train(example)
                    self.activations.append(output)
                self.activations.append(1.0) #bias term
                for layer in range(self.layers):
                    hidden_activations = []
                    for i in range(self.hidden):
                        hidden_activations.append(self.hidden_layers[layer][i].train(self.activations))
                    hidden_activations.append(1.0)
                    self.activations = hidden_activations
                if self.algorithm == 0:
                    self.output_neuron.clamp(self.activations,table[self.index(example)])
                if self.algorithm == 1:
                    self.output_neuron.selective_train(self.activations,table[self.index(example)])
                if self.algorithm == 2:
                    self.output_neuron.train(self.activations)
            else:
                if self.algorithm == 0:
                    self.output_neuron.clamp(example,table[self.index(example)])
                if self.algorithm == 1:
                    self.output_neuron.train(example)
                if self.algorithm == 2:
                    self.output_neuron.selective_train(example,table[self.index(example)])
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

    def monotone(self,table):
        monotone = True
        for i in range(len(table)):
            for j in range(i+1,len(table)):
                first_number = []
                second_number = []
                binary = str(bin(i).lstrip('0b'))
                for k in range(self.variables-len(binary)):
                    binary = '0'+binary
                for index in range(len(binary)):
                    first_number.append(int(binary[index]))
                binary = str(bin(j).lstrip('0b'))
                for k in range(self.variables-len(binary)):
                    binary = '0'+binary
                for index in range(len(binary)):
                    second_number.append(int(binary[index]))
                if booleanCompare(first_number,second_number):
                    if table[i] > table[j]:
                        monotone = False
        return monotone


def run(rate,sigmoid,hidden,examples,variables,layers,algorithm):
    functions = []
    example = 0
    monotone_fxns = 0
    previous = 0
    tables = truth_tables(variables)
    #tables = monotone_truth_tables(variables)
    not_learned = ""
    for i in range(len(tables)):
        print "Learning", tables[i],
        example+=1
        learned = False
        tries = 0
        while not learned and tries < 5000:
            tries += 1
            model = Network(rate,sigmoid,hidden,examples,variables,layers,algorithm)
            learned = model.train(tables[i])
        if learned:
            print "Learned"
            monotone = model.monotone(tables[i])
            functions.append(bit_repr(tables[i]))
            if monotone:
                monotone_fxns += 1
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
    parser.add_argument("--algorithm",type=int)
    args = parser.parse_args()
    
    print run(args.rate,args.sigmoid,args.hidden,args.examples,args.variables,args.layers,args.algorithm)

if __name__=="__main__":
    main()






