import unittest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from lib.feeder import Feeder
from lib.helper.functions import costFunction
from mindy import Mindy
import numpy as np

class TestMindy(unittest.TestCase):
    
    def assign_io_data(self):
        
        
        self.input = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
        ])
    
        self.output = np.array([
        [0],
        [1],
        [1],
        [0],
        [1]
        ])
        


    def test_single_randomWeights_weightsMatrix(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        self.assertEqual(_neurons * len(self.input.T), len(_feed.inputWeights) * len(_feed.inputWeights.T))
        
    def test_single_randomWeights_hiddenWeights(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        self.assertEqual(_neurons, len(_feed.hiddenOneWeights))
        
    def test_single_pushForward_predictedOutput(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        _feed.feedForward(_mindSingle.inputM , _mindSingle.outputM)
        self.assertEqual(_feed.predictedOutput.shape, self.output.shape)
    
    def test_single_costFunction(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        _feed.feedForward(_mindSingle.inputM, _mindSingle.outputM)
        _costFunc = costFunction(_feed)
        self.assertEqual(_feed.predictedOutput.shape, self.output.shape)
  
    def test_feedBackwards(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        _feed.feedForward(_mindSingle.inputM, _mindSingle.outputM)
        _feed.feedBackwards(0.1)
        self.assertEqual(_feed.inputWeights.shape, _feed.deltaInputChange.shape)

        

    def test_single_outputSum_deltaOutputSum(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        _feed.feedForward(_mindSingle.inputM, _mindSingle.outputM)
        _feed.feedBackwards(0.1)
        self.assertEqual(_feed.deltaOutputSum.shape, self.output.shape)
        
    
    def test_single_hiddenLayerBackward_deltaHiddenChange(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        _feed.feedForward(_mindSingle.inputM, _mindSingle.outputM)
        _feed.feedBackwards(0.1)
        self.assertEqual(_feed.deltaHiddenChange.shape, _feed.hiddenOneWeights.shape)
        
    def test_single_inputLayerBackward_deltaInputSum(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        _feed.feedForward(_mindSingle.inputM, _mindSingle.outputM)
        _feed.feedBackwards(0.1)
        self.assertEqual(_feed.deltaInputSum.shape, _feed.inputToHiddenOne.shape)
    
    def test_single_inputLayerBackward_deltaInputChange(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        _feed.feedForward(_mindSingle.inputM, _mindSingle.outputM)
        _feed.feedBackwards(0.1)
        self.assertEqual(_feed.deltaInputChange.shape, _feed.inputWeights.shape)
        
    def test_single_train_weightsMatrix(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        _feed.feedForward(_mindSingle.inputM, _mindSingle.outputM)
        _feed.feedBackwards(0.1)
        _mindSingle.train(10)
        self.assertEqual(_feed.inputWeights.shape, _feed.deltaInputChange.shape)
    
    def test_single_train_hiddenWeights(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        _feed.feedForward(_mindSingle.inputM, _mindSingle.outputM)
        _feed.feedBackwards(0.1)
        _mindSingle.train(10)
        self.assertEqual(_feed.hiddenOneWeights.shape, _feed.deltaHiddenChange.shape)
        
        





        

if __name__ == "__main__":
    unittest.main()
    

