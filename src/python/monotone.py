

import math

def test_tonicity(table):
    monotone = False
    if table[1] >= table[0] and table[2] >= table[0]:
        if table[3] >= table[1] and table[3] >= table[2]:
            if table[4] >= table[0]:
                if table[5] >= table[4] and table[5] >= table[1]:
                    if table[6] >= table[4] and table[6] >= table[2]:
                        if table[7] >= table[6] and table[7] >= table[5] and table[7] >= table[3]:
                            monotone = True
    return monotone

def test_tone2(table):
    monotone = True
    previous = 0
    for var in table:
        if var < previous:
            monotone = False
        previous = var
    return monotone

def booleanCompare(input1,input2):
    lessThan = True
    for digit1,digit2 in zip(input1,input2):
        if digit1 > digit2:
            lessThan = False
    return lessThan

def monotone(table):
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

'''def monotone(table):
    monotone = True
    previous = []
    digits = 2
    variables = math.pow(2,digits)
    for i in range(digits):
        previous.append(0) 
    for i in range(1,int(variables)):
        inputs = []
        binary = str(bin(i).lstrip('0b'))
        for i in range(digits-len(binary)):
            binary = '0'+binary
        lessThan = True
        for j in range(digits):
            digit = int(binary[j])
            lessThan = True
            if previous[j] > digit:
                lessThan = False
            inputs.append(digit)
        previous = inputs
        if lessThan:
            if table[i-1] > table[i]:
                monotone = False
    return monotone'''

count = 0
for a in [0,1]:
    for b in [0,1]:
        for c in [0,1]:
            for d in [0,1]:
                if monotone([a,b,c,d]):
                    count += 1
print count

count1 = 0
count2 = 0
count3 = 0
total = 0
for a0 in [0,1]:
    for a1 in [0,1]:
        for a2 in [0,1]:
            for a3 in [0,1]:
                for a4 in [0,1]:
                    for a5 in [0,1]:
                        for a6 in [0,1]: 
                            for a7 in [0,1]:
                                total +=1
                                truth = [a0,a1,a2,a3,a4,a5,a6,a7]
                                if monotone(truth):
                                    count3+=1
print count3
print total
