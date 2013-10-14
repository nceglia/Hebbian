import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, splitext
import sys

mypath = sys.argv[1]
sig_list = ["x/(1+|x|)","tanh(x)","x/sqrt(1+x^2)"]
pres_list = ["Random","Ordered"]

def get_files():
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    return onlyfiles

def create_bar(name,conditions,data):
    functions = ['FALSE', 'AND', 'A AND NOT B','A','NOT A and B','B','XOR','OR','NOR','XNOR','NOT B','A OR NOT B','NOT A','NOT A OR B','NAND','TRUE']
    fig = plt.figure()
    plt.title(conditions)
    ax = plt.subplot(111)
    width=0.8
    ax.bar(range(len(functions)), data, width=width)
    ax.set_xticks(np.arange(len(functions)) + width/2)
    ax.set_xticklabels(functions, rotation=45)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(7) 
    plt.savefig(mypath+'plots/{0}.png'.format(name), bbox_inches=0)
    plt.close()

def get_data(name):
    experiment = open(mypath+name,"r").read().split('\n')
    dist = experiment[-2].split()
    distribution = []
    if len(dist) == 16:
        for val in dist:
            distribution.append(int(val))
        return distribution
    else:
        print "Data incorrect!"
        return False

def main():
    plt.close("all")
    files = get_files()
    for name in files:
        conds = splitext(name)[0].split("_")
        if len(conds) > 3:
            #args.rate,args.sigmoid,args.examples,args.presentation
            rate = conds[1]
            sigmoid = conds[2]
            examples = conds[3]
            presentation = conds[4]
            title = "Rate: "+rate+" Sigmoid: "+sig_list[int(sigmoid)-1]+" Examples: "+examples+" "+pres_list[int(presentation)-1]
            data = get_data(name)
            if data:
                create_bar(splitext(name)[0],title,data)
    plt.close("all")

if __name__ == '__main__':
    main()