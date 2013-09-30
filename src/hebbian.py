import sys
import os
import argparse
import numpy
from mnist import MNIST
from math import exp
import math

def sigmoid(value):
    try:
        sig_output = 1.0/(1.0+exp(-1.0*float(value)))
    except:
        if value > 0:
            sig_output = 1.0
        else:
            sig_output = 0.0
    return sig_output


def compute(inputs,weights,bias):
    activation = 0.0
    for inp,weight,b in zip(inputs,weights,bias):
        activation += float(inp)*weight+b
    return activation

class HebbianAutoencoder():
    def __init__(self,inputs,hidden,rate,tied):
        #layers = 2

        self.hidden = hidden
        self.output = inputs

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

        data = MNIST('/home/dock/workspace/nceglia/data/')
        print 'Loading data...'
        self.training = data.load_training()
        self.testing = data.load_testing() 
        print 'Finished loading data'


    def monitor(self,outputs, inputs):
        error = 0.0
        for out, inp in zip(outputs,inputs):
            error += math.pow(out - inp,2)
        return error

    def train(self):
        #perform one epoch
        for num, example in enumerate(self.training[0]):
            for i in range(len(example)):
                self.average_input[i] += float(example[i])
            print '\t\tExample', num,
            for index in range(self.hidden):
                self.activation_hidden[index] = sigmoid(compute(example,self.weights[0][index],self.bias[0]))
                self.average_hidden[index] += self.activation_hidden[index]
                for i, weight in enumerate(self.weights[0][index]):
                    #delta = self.rate * (self.activation_hidden[index] - self.average_hidden[index]/(num+1)) * (example[i] - self.average_input[i]/(num+1))
                    delta = self.rate * self.activation_hidden[index] * float(example[i])
                    if delta != 0.0:
                        weight = weight+delta
                        self.weights[0][index][i] = sigmoid(weight)

            weights_mirror = numpy.transpose(numpy.array(self.weights[0]))
            for index in range(self.output):
                if not self.tied:
                    self.activation_output[index] = compute(self.activation_hidden,self.weights[1][index],self.bias[0])
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
            if num%10 == 0 and num > 0:
                print '\t\t\tSaving Model...'
                self.save()
        return current
 
    def save(self):
        model_file = open("model.txt","w")
        for layer in self.weights:
             for i in range(len(layer)):
                 model_file.write(str(layer[i])+'\n')
        model_file.close()

def main():

    parser = argparse.ArgumentParser(prog="Hebbian Autoencoder")

    parser.add_argument("--hidden",type= int)
    parser.add_argument("--input",type=int)
    parser.add_argument("--rate",type=float)
    parser.add_argument('--tied', action='store_true', default=False)

    args = parser.parse_args()
    model = HebbianAutoencoder(args.input,args.hidden,args.rate,args.tied)
    error = 1.0
    epoch = 1
    while(error > 0.05):
        print "\tStarting Epoch:", epoch
        error = model.train()

    model.save()

    

if __name__=="__main__":
    main()
