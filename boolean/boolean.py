from random import randint
#class to mimic MNIST and generate BOOLEAN data


class BOOLEAN():
    def __init__(self,examples,variables):
        self.variables = variables            
        self.examples = examples

    def load_training(self):
        inputs = []
        for i in range(self.examples):
            example = []
            for i in range(self.variables):
                a = randint(-1,1)
                while a == 0:
                    a = randint(-1,1)
                example.append(a)
            example.append(1)
            
            inputs.append(example)
        return inputs


    def load_testing(self):
        inputs = []
        for i in range(self.examples):
            example = []
            for i in range(self.variables):
                a = randint(-1,1)
                while a == 0:
                    a = randint(-1,1)
                example.append(a)
            example.append(1)
            
            inputs.append(example)
        return inputs
        
