import sys
import os
import math 
from HebbianNetwork import run

def main():
	csv_hebbian = open("hebb_results.csv","w")
	for variables in range(2,5):
		for rate in range(1,5):
			for sigmoid in range(1,4):
				for hidden in range(6):
					csv_hebbian.write(str(float(rate)/10.0) + '\t' + str(sigmoid) + '\t' + str(hidden) + '\t' + str(variables) + '\t' + str(int(math.pow(2,variables))*25) + '\t')
					learned, monotone, not_learned, not_stable = run(float(rate)/10.0,sigmoid,hidden,variables,int(math.pow(2,variables))*25)
					csv_hebbian.write(str(learned) + '\t' + str(monotone) + '\t' + not_learned + '\t' + not_stable + '\n')
	csv_hebbian.close()

if __name__=="__main__":
	main()
