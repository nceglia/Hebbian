
from mock import patch, MagicMock
from Neuron import Neuron
from Network import Network
import unittest

class NeuronTests(unittest.TestCase):

    def setUp(self):
        pass

    #test train method
    def test_train(self):
        test_neuron = Neuron(0.01, 0, 2, 1.0)
        test_neuron.set_weights([0.5,0.5,0.5])
        test_neuron.compute = MagicMock(return_value=1.0)
        result = test_neuron.train([1,1,1],"oja")
        updated = test_neuron.get_weights()
        self.assertEqual(updated,['0.505','0.505','0.505'])

    

    #test clamp method
    def test_clamp(self):
        test_neuron = Neuron(0.01, 0, 2, 1.0)
        test_neuron.set_weights([0.5,0.5,0.5])
        result = test_neuron.clamp([1,1,1],1.0,"oja")
        updated = test_neuron.get_weights()
        self.assertEqual(updated,['0.505','0.505','0.505'])

    #test compute method
    def test_compute(self):
        test_neuron = Neuron(0.01, 0, 2, 1.0)
        test_neuron.set_weights([0.75,-0.5,-0.5])
        linear_activation = test_neuron.compute([1,1,1])
        self.assertEqual(linear_activation,-1.0) 

    def test_noise(self):
        pass

    def tearDown(self):
        pass

class NetworkTets(unittest.TestCase):

    def setUp(self):
        pass

    @patch('random.uniform')
    def test_compute(self,mock_np_dist):
        mock_np_dist.return_value = [0.5,0.5,0.5]
        model = Network(0.01, 0, 2, 1, 2, 1, "oja", 1.0)
        output = model.compute([-1,-1,-1])
        self.assertEqual(output,0)

    @patch('random.uniform')
    def test_truthtable(self,mock_np_dist):
        mock_np_dist.return_value = [1.0,1.0,1.0]
        model = Network(0.01, 0, 2, 1, 2, 1, "oja", 1.0)
        table = model.truthtable()
        self.assertEqual(table,[1,1,1,1])

    def test_noise(self):
        pass

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()
