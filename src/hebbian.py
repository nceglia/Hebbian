import sys
import os
import argparse
import numpy
from mnist import MNIST
from math import exp

def sigmoid(value):
    return 1.0/(1.0+exp(-1.0*float(value)))

class LinearNeuron():
    def __init__(self):
        self.activation = 0.0
    def compute(self,inputs,weights,bias):
        self.activation = 0.0
        for input,weight,b in zip(inputs,weights,bias):
            self.activation = self.activation + float(input)*weight+b
        return self.activation

class HebbianAutoencoder():
    def __init__(self,inputs,hidden,rate,tied):
        #layers = 2
        self.tied = tied
        self.rate = rate
        self.weights = []
        self.weights.append([])
        for i in range(hidden):
            self.weights[0].append(numpy.random.uniform(-0.15,0.15,size=inputs))
        self.weights.append([])
        for i in range(inputs):
            self.weights[1].append(numpy.random.uniform(-0.15,0.15,size=hidden))
        self.bias = []
        #inputs
        self.bias.append(numpy.random.uniform(-0.15,0.15,size=inputs))
        #outputs
        self.bias.append(numpy.random.uniform(-0.15,0.15,size=hidden))

        self.hidden = []
        self.output = []
        for i in range(hidden):
            self.hidden.append(LinearNeuron())
        for i in range(inputs):
            self.output.append(LinearNeuron())

        data = MNIST('/home/dock/workspace/nceglia/data/')
        print 'Loading data'
        self.training = data.load_training()
        self.testing = data.load_testing() 
        self.previous_state = None

    def monitor(self,outputs, inputs):
        error = 0.0
        for out, inp in zip(outputs,inputs):
            error += out - inp
        return error/len(inputs)

    def train(self):
        #perform one epoch
        for num, example in enumerate(self.training[0]):
            print 'Example', num,
            activation_hidden = []
            activation_output = []
            for index, neuron in enumerate(self.hidden):
                activation_hidden.append(neuron.compute(example,self.weights[0][index],self.bias[0]))
                for i, weight in enumerate(self.weights[0][index]):
                    delta = self.rate * activation_hidden[index] * example[i]
                    weight = weight+sigmoid(delta)
                    self.weights[0][index][i] = weight

            weights_mirror = numpy.transpose(numpy.array(self.weights[0]))
            for index, neuron in enumerate(self.output):
                if not self.tied:
                    activation_output.append(neuron.compute(activation_hidden,self.weights[1][index],self.bias[0]))
                    for i, weight in enumerate(self.weights[1][index]):
                        delta = self.rate * activation_hidden[index] * (example[i] - weight * activation_hidden[index])
                        weight = weight+sigmoid(delta)
                        self.weights[1][index][i] = weight 
                else:
                    activation_output.append(neuron.compute(activation_hidden,weights_mirror[index],self.bias[1]))
            current = self.monitor(activation_output,example)
            print current
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
        print "Epoch:", epoch
        error = model.train()

    model.save()

    

if __name__=="__main__":
    main()
