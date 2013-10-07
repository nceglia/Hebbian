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

def check_weights(value):
    return value/(1.0+math.fabs(value))

def tanh(value):
    return math.tanh(value)

def check_weights_sqrt(value):
    return value/math.sqrt(1.0 + math.pow(value,2.0))

def compute(inputs,weights):
    activation = 0.0
    for inp,weight in zip(inputs,weights):
        activation += float(inp)*weight
    return activation

class HebbianAutoencoder():
    def __init__(self,inputs,hidden,rate,tied,data,freq):
        #layers = 2

        self.data = data
        self.hidden = hidden
        self.output = inputs
        self.freq = freq

        self.tied = tied
        self.rate = rate
        self.weights = []
        self.weights.append([])
        for i in range(hidden):
            self.weights[0].append(numpy.random.uniform(-1.0,1.0,size=inputs))
        self.weights.append([])
        for i in range(inputs):
            self.weights[1].append(numpy.random.uniform(-1.0,1.0,size=hidden))
        self.bias = []
        #inputs
        self.bias.append(numpy.random.uniform(-1.0,1.0,size=inputs))
        #outputs
        self.bias.append(numpy.random.uniform(-1.0,1.0,size=hidden))

        self.average_input = []
        self.average_hidden = []
        self.average_output = []
        self.activation_hidden = []
        self.activation_output = []

        for i in range(hidden):
            self.average_hidden.append(0.0)
            self.activation_hidden.append(0.0)
        for i in range(inputs):
            self.average_input.append(0.0)
            self.average_output.append(0.0)
            self.activation_output.append(0.0)

        if self.data.lower() == 'mnist':
            print "MNIST DATA..."
            data = MNIST('/home/dock/workspace/nceglia/data/')
        elif self.data.lower() == "xor":
            print "XOR DATA...."
            data = XOR()

        print 'Loading data...'
        self.training = data.load_training()
        self.testing = data.load_testing() 
        print 'Finished loading data'


    def monitor(self,outputs, inputs):
        error = 0.0
        for out, inp in zip(outputs,inputs):
            error += math.pow(out - inp,2)
        return math.sqrt(error)

    def train(self):
        #perform one epoch
        for num, example in enumerate(self.training[0]):
            for i in range(len(example)):
                self.average_input[i] += float(example[i])
            print '\t\tExample', num,
            for index in range(self.hidden):
                self.activation_hidden[index] = compute(example,self.weights[0][index],self.bias[0])
                self.average_hidden[index] += self.activation_hidden[index]
                for i, weight in enumerate(self.weights[0][index]):
                    #delta = self.rate * (self.activation_hidden[index] - self.average_hidden[index]/(num+1)) * (example[i] - self.average_input[i]/(num+1))
                    delta = self.rate * self.activation_hidden[index] * float(example[i])/255.0
                    if delta != 0.0:
                        weight = weight+delta
                        self.weights[0][index][i] = sigmoid(weight)

            weights_mirror = numpy.transpose(numpy.array(self.weights[0]))
            for index in range(self.output):
                if not self.tied:
                    self.activation_output[index] = sigmoid(compute(self.activation_hidden,self.weights[1][index],self.bias[0]))
                    self.average_output[index] += self.activation_output[index]
                    for i, weight in enumerate(self.weights[1][index]):
                        #delta = self.rate * (self.activation_output[index]-self.average_output[index]/(num+1)) * (self.activation_hidden[i] - self.average_hidden[i]/(num+1))
                        delta = self.rate * self.activation_output[index] * self.activation_hidden[i]
                        if delta != 0.0:
                            weight = weight+delta
                            self.weights[1][index][i] = sigmoid(weight) 
                else:
                    activation_output.append(neuron.compute(activation_hidden,weights_mirror[index],self.bias[1]))
            current = self.monitor(self.activation_output,example)
            print current
            if num%self.freq == 0 and num > 0:
                print '\t\t\tSaving Model...'
                self.save()
        return current
 
    def save(self):
        model_file = open("model.txt","w")
        for layer in self.weights:
             for i in range(len(layer)):
                 model_file.write(str(layer[i])+'\n')
        model_file.close()


class HebbianBoolean():
    def __init__(self,rate):
        self.rate = rate
        self.weights = numpy.random.uniform(-1.0,1.0,size=3)

        print "BOOLEAN DATA"
        data = BOOLEAN()

        self.boolean_equations = []

        print '\tLoading data...'
        self.training = data.load_training()
        print '\tFinished loading data'

    def test_learning(self):
        print "Testing"
        activations=[]
        activations.append(compute([-1,0,1],self.weights))
        activations.append(compute([-1,1,1],self.weights))
        activations.append(compute([1,-1,1],self.weights))
        activations.append(compute([1,1,1],self.weights))
        #f0
        learned = True
        for activation in activations:
            if activation >= 0.0:
                learned = False
        if learned:    
            print "Learned F0, FALSE"
            return 0
        #f1
        learned = True
        if activations[0] >= 0.0:
            learned = False
        if activations[1] >= 0.0:
            learned = False
        if activations[2] >= 0.0:
            learned = False
        if activations[3] < 0.0:
            learned = False
        if learned:
            print "Learned F1, AND"
            return 1
        #f2
        learned = True
        if activations[0] >= 0.0:
            learned = False
        if activations[1] >= 0.0:
            learned = False
        if activations[2] < 0.0:
            learned = False
        if activations[3] >= 0.0:
            learned = False
        if learned:
            print "Learned F2, a AND NOT b" 
            return 2
        #f3
        learned = True
        if activations[0] >= 0.0:
            learned = False
        if activations[1] >= 0.0:
            learned = False
        if activations[2] < 0.0:
            learned = False
        if activations[3] < 0.0:
            learned = False
        if learned:
            print "Learned F3, a"       
            return 3
        #f4
        learned = True
        if activations[0] >= 0.0:
            learned = False
        if activations[1] < 0.0:
            learned = False
        if activations[2] >= 0.0:
            learned = False
        if activations[3] >= 0.0:
            learned = False
        if learned:
            print "Learned F4, NOT a AND b" 
            return 4
        #f5
        learned = True
        if activations[0] >= 0.0:
            learned = False
        if activations[1] < 0.0:
            learned = False
        if activations[2] >= 0.0:
            learned = False
        if activations[3] < 0.0:
            learned = False
        if learned:
            print "Learned F5, b" 
            return 5
        #f6
        learned = True
        if activations[0] >= 0.0:
            learned = False
        if activations[1] < 0.0:
            learned = False
        if activations[2] < 0.0:
            learned = False
        if activations[3] >= 0.0:
            learned = False
        if learned:
            print "Learned F6, XOR" 
            return 6
        #f7
        learned = True
        if activations[0] >= 0.0:
            learned = False
        if activations[1] < 0.0:
            learned = False
        if activations[2] < 0.0:
            learned = False
        if activations[3] < 0.0:
            learned = False
        if learned:
            print "Learned F7, OR"
            return 7
        #f8
        learned = True
        if activations[0] < 0.0:
            learned = False
        if activations[1] >= 0.0:
            learned = False
        if activations[2] >= 0.0:
            learned = False
        if activations[3] >= 0.0:
            learned = False
        if learned:
            print "Learned F8, NOR"  
            return 8

        #f9
        learned = True
        if activations[0] < 0.0:
            learned = False
        if activations[1] >= 0.0:
            learned = False
        if activations[2] >= 0.0:
            learned = False
        if activations[3] < 0.0:
            learned = False
        if learned:
            print "Learned F9, XNOR"  
            return 9
        #f10
        learned = True
        if activations[0] < 0.0:
            learned = False
        if activations[1] >= 0.0:
            learned = False
        if activations[2] < 0.0:
            learned = False
        if activations[3] >= 0.0:
            learned = False
        if learned:
            print "Learned F10, NOT b"  
            return 10
        #f11
        learned = True
        if activations[0] < 0.0:
            learned = False
        if activations[1] >= 0.0:
            learned = False
        if activations[2] < 0.0:
            learned = False
        if activations[3] < 0.0:
            learned = False
        if learned:
            print "Learned F11, a OR NOT b"  
            return 11
        #f12
        learned = True
        if activations[0] < 0.0:
            learned = False
        if activations[1] < 0.0:
            learned = False
        if activations[2] >= 0.0:
            learned = False
        if activations[3] >= 0.0:
            learned = False
        if learned:
            print "Learned F12, NOT a"  
            return 12
        #f13
        learned = True
        if activations[0] < 0.0:
            learned = False
        if activations[1] < 0.0:
            learned = False
        if activations[2] >= 0.0:
            learned = False
        if activations[3] < 0.0:
            learned = False
        if learned:
            print "Learned F13, NOT a or b"  
            return 13

        #f14
        learned = True
        if activations[0] < 0.0:
            learned = False
        if activations[1] < 0.0:
            learned = False
        if activations[2] < 0.0:
            learned = False
        if activations[3] >= 0.0:
            learned = False
        if learned:
            print "Learned F14, NAND"  
            return 14
        #f15
        learned = True
        for activation in activations:
            if activation < 0.0:
                learned = False
        if learned:    
            print "Learned F15, TRUE"
            return 15


    def train(self):
        #perform one epoch
        for num, example in enumerate(self.training):
            #print '\tExample', num
            activation = compute(example,self.weights)
            for index in range(3):
                for i, weight in enumerate(self.weights):
                    delta = self.rate + activation * float(example[i])
                    if delta != 0.0:
                        weight = weight+delta
                        self.weights[i] = check_weights(weight)
            if num%100:
                self.save(num)
            #print "\t\tInputs:",example[0],example[1]
            #print "\t\tWeights:", self.weights[0], self.weights[1]
            #print "\t\tOutput:", activation
    
    def save(self,example):
        model_file = open("/home/dock/workspace/nceglia/data/hebbian/model{0}.txt".format(str(example)),"a")
        for wt in self.weights:
            model_file.write(str(wt)+'\t')
        model_file.write('\n')
        model_file.close()


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

def main():

    parser = argparse.ArgumentParser(prog="Boolean Autoencoder")

    args = parser.parse_args()
    boolean_equations = []
    for i in range(16):
        boolean_equations.append(0)

    examples = 0
    while(continue_learning(boolean_equations) or examples > 10000):
        rate = numpy.random.uniform(0.0,1.0,size=1)
        model = HebbianBoolean(rate[0])
        print "*********************"
        examples+=1
        print "Example:", examples
        print "Rate",rate[0]
        model.train()
        fxn = model.test_learning()
        boolean_equations[fxn] += 1
        print "Function Distribution:", boolean_equations
        print "\n"

    for i in range(16):
        if not boolean_equations[i]:
            print "Fxn: ", i, "not learned." 
if __name__=="__main__":
    main()
