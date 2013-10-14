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

def compute(inputs,weights):
    activation = 0.0
    for inp,weight in zip(inputs,weights):
        activation += float(inp)*weight
    return activation

def continue_learning(conditions):
    proceed = False
    for i, condition in enumerate(conditions):
        if i == 6 or i == 9:
            continue
        else:
            if condition == 0:
                proceed = True
    return proceed

class HebbianBoolean():
    def __init__(self,rate,sigmoid,examples,presentation):
        self.rate = rate
        self.weights = numpy.random.uniform(-1.0,1.0,size=3)
        self.sigmoid = sigmoid
        self.examples = examples
        self.presentation = presentation

        print "BOOLEAN DATA"
        data = BOOLEAN(self.examples,self.presentation)

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
            print example
            activation = compute(example,self.weights)
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
    
    def get_weights(self):
        return str(self.weights[0])+"\t"+str(self.weights[1])+"\t"+str(self.weights[2])

def main():
    parser = argparse.ArgumentParser(prog="Boolean Hebbian")
    parser.add_argument("--rate",type=float)
    parser.add_argument("--sigmoid",type=int)
    #3 different types ... see top
    parser.add_argument("--examples",type=int)
    #number of examples to train 1 model
    parser.add_argument("--presentation",type=int)
    #1 = random, 2 = ordered
    args = parser.parse_args()
    boolean_equations = []
    for i in range(16):
        boolean_equations.append(0)
    model_file = open("/home/dock/workspace/nceglia/data/hebbian/hebb_{0}_{1}_{2}_{3}.txt".format(args.rate,args.sigmoid,args.examples,args.presentation),"w")
    examples = 0
    while(continue_learning(boolean_equations) or examples < 1000):
        if examples > 20000:
            break
        rate = numpy.random.uniform(0.0,1.0,size=1)
        model = HebbianBoolean(args.rate,args.sigmoid,args.examples,args.presentation)
        examples+=1
        model.train()
        fxn = model.test_learning()
        boolean_equations[fxn] += 1
        model_file.write(str(examples)+'\t'+str(fxn)+'\t'+model.get_weights()+'\n')
        print examples

    boolean_equations = [str(i) for i in boolean_equations]
    model_file.write("\t".join(boolean_equations)+'\n')

if __name__=="__main__":
    main()
