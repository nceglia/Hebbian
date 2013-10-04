from random import randint
#class to mimic MNIST and generate XOR data

class XOR():
    def __init__(self):
        pass

    def load_training(self):
        examples = []
        inputs = []
        classifications = []
        for i in range(5000):
            a = randint(0,1)
            b = randint(0,1)
            inputs.append([a,b])
            classifications.append(self.xor(a,b))
        examples.append(inputs)
        examples.append(classifications)
        return examples

    def load_testing(self):
        examples = []
        inputs = []
        classifications = []
        for i in range(10000):
            a = randint(0,1)
            b = randint(0,1)
            inputs.append([a,b])
            classifications.append(self.xor(a,b))
        examples.append(inputs)
        examples.append(classifications)
        return examples

    def xor(self,a,b):
        if a == b:
            return 1
        else:
            return 0

class BOOLEAN():
    def __init__(self):
        pass

    def load_training(self):
        inputs = []
        for i in range(100000):
            a = randint(-1,1)
            while a == 0:
                a = randint(-1,1)
            b = randint(-1,1)
            while b == 0:
                b = randint(-1,1)
            inputs.append([a,b,1])
        return inputs

class BOOLEAN_SINGULAR():
    def __init__(self):
        pass

    def load_training(self):
        inputs = []
        for i in range(1000):
            a = randint(-1,1)
            while a == 0:
                a = randint(-1,1)
            inputs.append([a,1])
        return inputs