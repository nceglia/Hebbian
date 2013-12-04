"""
Constructing and training hebbian feed forward neural networks
for n variable boolean functions.
"""
import argparse
from math import exp
import math
from Network import Network

def truth_tables(variables):
    """
    Builds a list of truth tables for n variable boolean functions

    Keyword arguments:
    variables -- boolean variables (float)

    returns list of truth tables
    """
    tables = []
    for i in range(int(math.pow(2, math.pow(2, variables)))):
        binary = bin(i).lstrip('0b')
        table = []
        for i in range(int(math.pow(2, variables)-len(binary))):
            binary = '0' + binary
        for digit in binary:
            table.append(int(digit))
        tables.append(table)
    return tables

def monotone(table, variables):
    """
    Determines wether a truth table for a boolean function is monotonic

    Keyword argument:
    table -- truth table (list)
    variables -- n variable boolean function

    returns true if function is monotonic and false if not.
    """
    monotone_table = True
    for i in range(len(table)):
        for j in range(i + 1, len(table)):
            first_number = []
            second_number = []
            binary = str(bin(i).lstrip('0b'))
            for _ in range(variables - len(binary)):
                binary = '0' + binary
            for index in range(len(binary)):
                first_number.append(int(binary[index]))
            binary = str(bin(j).lstrip('0b'))
            for _ in range(variables - len(binary)):
                binary = '0' + binary
            for index in range(len(binary)):
                second_number.append(int(binary[index]))
            if boolean_compare(first_number, second_number):
                if table[i] > table[j]:
                    monotone_table = False
    return monotone_table

def monotone_truth_tables(variables):
    """
    Finds monotonic truth tables for n variable boolean functions

    Keyword arguments:
    variables -- n variable boolean functions (float)

    returns list of monotonic truth tables
    """
    tables = truth_tables(variables)
    monotone_tables = []
    for table in tables:
        if monotone(table, variables) and table != []:
            monotone_tables.append(table)
    return monotone_tables

def bit_repr(table):
    """
    Takes a table as list and returns binary representation

    Keyword arguments:
    table -- truth table for boolean function (list)

    returns an integer representation of table (ex: 10101)
    """
    bits = ""
    for bit in table:
        bits  = bits + str(bit)
    return int(bits)

def boolean_compare(input1, input2):
    """
    Compares if one binary number is montonically greater than the second.

    Keyword arguments:
    input1 -- binary number as list (list)
    input2 -- binary number as list (list)

    returns true if monotonically increasing, false if not.
    """
    less_than = True
    for digit1, digit2 in zip(input1, input2):
        if digit1 > digit2:
            less_than = False
    return less_than

def monotone_generator(inputs):
    variables = [[0],[1]]
    for i in range(int(inputs)):
        next_set = []
        for variable1 in variables:
            for variable2 in variables:
                if boolean_compare(variable1,variable2):
                    next_set.append(variable1+variable2)
        variables = next_set
    return variables

def run(rate, sigmoid, hidden, examples, variables, layers, rule, dropout):
    """
    Creates network and trains a model for each boolean function

    Keyword arguments:
    rate -- learning rate (float)
    sigmoid -- sigmoid function for weights if rule is basic hebbian  (int)
    hidden -- number of hidden units, 0 removes hidden layer (int)
    examples -- number of random boolean examples to present.
    layers -- number of hidden layers 1 to N (int)
    rule -- learning rule, "hebbian" or "oja" (str)
    dropout -- percentage of edge weights to update
    prints each function, whether it was able to learn it, and a summary.
    """
    functions = []
    example = 0
    monotone_fxns = 0
    #tables = truth_tables(variables)
    tables = monotone_generator(variables)
    not_learned = ""
    for i in range(len(tables)):
        print "Learning", tables[i],
        example += 1
        learned = False
        tries = 0
        while not learned and tries < 200000:
            tries += 1
            model = Network(rate, sigmoid, hidden, examples, variables, layers, rule, dropout)
            learned = model.train(tables[i])
        if learned:
            print "Learned with {0} models".format(tries)
            functions.append(bit_repr(tables[i]))
            model.save("models/hebb{0}.txt".format(example))
            #model.test("models/hebb{0}.txt".format(example))
        else:
            not_learned = not_learned+str(i)+","
            print "Not Learned"
    return "Learned:", len(functions),"Not Learned", not_learned

def distribute(rate, sigmoid, hidden, examples, variables, layers, rule, dropout, table):
    example = 0
    not_learned = ""
    tables = monotone_generator(variables)
    print "Learning", tables[table-1],
    learned = False
    tries = 0
    while not learned and tries < 200000:
        tries += 1
        model = Network(rate, sigmoid, hidden, examples, variables, layers, rule, dropout)
        learned = model.train(tables[table-1])
    if learned:
        print "Learned with {0} models".format(tries)
        model.save("hebb{0}.txt".format(table-1))
    else:
        print "Not Learned"
    return

def main():
    """Parses command line args and calls run method"""

    parser = argparse.ArgumentParser(prog="Hebbian Network")
    parser.add_argument("--rate", type=float)
    parser.add_argument("--sigmoid", type=int)
    parser.add_argument("--examples", type=int)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--variables", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--rule", type=str)
    parser.add_argument("--dropout",type=float)
    parser.add_argument("--table",type=int)
    args = parser.parse_args()

    if args.table == 0:
        print run(args.rate, args.sigmoid, args.hidden, args.examples, args.variables, args.layers, args.rule, args.dropout)
    else:
        distribute(args.rate, args.sigmoid, args.hidden, args.examples, args.variables, args.layers, args.rule, args.dropout, args.table)

if __name__ == "__main__":
    main()



