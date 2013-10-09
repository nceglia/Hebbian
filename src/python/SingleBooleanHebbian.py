import sys
import os
import argparse
import numpy
from mnist import MNIST
from math import exp
import math
from xor import XOR
from xor import BOOLEAN

def sigmoid(value):
    try:
        sig_output = 1.0/(1.0+exp(-1.0*float(value)))
    except:
        if value > 0:
            sig_output = 1.0
        else:
            sig_output = 0.0
    return sig_output

def sigmoid2(value):
    return value/(1.0+math.fabs(value))

def tanh(value):
    return math.tanh(value)

def sigmoid3(value):
    return value/math.sqrt(1.0 + math.pow(value,2.0))

def compute(inputs,weights):
    activation = 0.0
    for inp,weight in zip(inputs,weights):
        activation += float(inp)*weight
    return activation

def continue_learning(conditions):
    proceed = False
    for condition in conditions:
        if condition == 0:
            proceed = True
    return proceed


class HebbianBooleanSingular():
    def __init__(self,rate):
        self.rate = rate
        self.weights = numpy.random.uniform(-1.0,1.0,size=2)

        print "BOOLEAN DATA"
        data = BOOLEAN()

        print '\tLoading data...'
        self.training = data.load_training()
        print '\tFinished loading data'

    def test_learning(self):
        print "Testing"
        activations=[]
        activations.append(compute([-1,1],self.weights))
        activations.append(compute([1,1],self.weights))
        #True
        print "0 activates:",
        if activations[0] >= 0.0:
            print 'True'
        else:
            print 'False'
        print "1 activates:", 
        if activations[1] >= 0.0:
            print 'True'
        else:
            print 'False'


    def train(self):
        #perform one epoch
        for num, example in enumerate(self.training):
            #print '\tExample', num
            activation = compute(example,self.weights)
            for index in range(2):
                for i, weight in enumerate(self.weights):
                    delta = self.rate * activation * float(example[i])
                    if delta != 0.0:
                        weight = weight+delta
                        self.weights[i] = check_weights(weight)
            #print "\t\tInputs:",example[0],example[1]
            #print "\t\tWeights:", self.weights[0], self.weights[1]
            #print "\t\tOutput:", activation
    
    def save(self):
        model_file = open("model.txt","w")
        for wt in self.weights:
            model_file.write(wt+'\t')
        model_file.close()

def continue_learning(conditions):
    proceed = False
    for condition in conditions:
        if condition == 0:
            proceed = True
    return proceed