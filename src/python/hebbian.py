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
    return value/(1+math.fabs(value))

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

        #f15
        learned = True
        for activation in activations:
            if activation < 0.0:
                learned = False
        if learned:    
            print "Learned F15, TRUE"



    def train(self):
        #perform one epoch
        for num, example in enumerate(self.training):
            #print '\tExample', num
            activation = compute(example,self.weights)
            for index in range(3):
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


def main():

    parser = argparse.ArgumentParser(prog="Boolean Autoencoder")

    args = parser.parse_args()

    for i in range(100):
        rate = numpy.random.uniform(0.0,1.0,size=1)
        model = HebbianBoolean(rate[0])
        print "*********************"

        print "Rate",rate[0]
        model.train()
        model.test_learning()
        print "*********************\n\n\n"

if __name__=="__main__":
    main()
