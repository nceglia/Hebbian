import sys
import os
from subprocess import call
import math 

def main():
    for rate in range(1,11):
    	for sigmoid in range(3):
    		for hidden in range(10):
    			for variables in range(5):
    				print "Rate:",str(float(rate)/10.0),\
    				"Sigmoid:",sigmoid,"Hidden:",hidden,"Variables:",variables,"Examples:",str(variables*25) 
    				call(['python','../src/python/HebbianNetwork.py','--rate',str(float(rate)/10.0),\
    					'--sigmoid',str(sigmoid),'--hidden',str(hidden),'--variables',str(variables),\
    					'--examples',str(variables*25)])
if __name__=="__main__":
    main()
