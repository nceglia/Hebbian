from random import randint
#class to mimic MNIST and generate BOOLEAN data

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
    def __init__(self,examples,variables):
        self.number = examples
        self.variables = variables            

    def load_training(self):
        inputs = []
        for i in range(self.number):
            example = []
            for i in range(self.variables):
                a = randint(-1,1)
                while a == 0:
                    a = randint(-1,1)
                example.append(a)
            example.append(1)
            inputs.append(example)
        return inputs
        