import unittest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from mindy import Mindy
import numpy as np

class TestMindy(unittest.TestCase):
    
    def assign_io_data(self):
        
        
        self.inputM = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
        ])
    
        self.outputM = np.array([
        [0],
        [1],
        [1],
        [0],
        [1]
        ])
        

    

    def test_single_randomInputWeights_weightsMatrix(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.inputM, self.outputM, _neurons, 0.1)
        _mindSingle.randomInputWeights()
        self.assertEqual(_neurons * len(self.inputM.T), len(_mindSingle.weightsMatrix) * len(_mindSingle.weightsMatrix.T))
        
    def test_single_randomInputWeights_hiddenWeights(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.inputM, self.outputM, _neurons, 0.1)
        _mindSingle.randomInputWeights()
        self.assertEqual(_neurons, len(_mindSingle.hiddenWeights))
        
    def test_single_hiddenLayerSumForward_predictedOutput(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.inputM, self.outputM, _neurons, 0.1)
        _mindSingle.randomInputWeights()
        _mindSingle.hiddenLayerSumForward()
        self.assertEqual(_mindSingle.predictedOutput.shape, self.outputM.shape)
        
    def test_single_outputSum_deltaOutputSum(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.inputM, self.outputM, _neurons, 0.1)
        _mindSingle.randomInputWeights()
        _mindSingle.hiddenLayerSumForward()
        _mindSingle.outputSum()
        self.assertEqual(_mindSingle.deltaOutputSum.shape, self.outputM.shape)
        
        
    def test_single_hiddenLayerBackward_deltaHiddenChange(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.inputM, self.outputM, _neurons, 0.1)
        _mindSingle.randomInputWeights()
        _mindSingle.hiddenLayerSumForward()
        _mindSingle.outputSum()
        _mindSingle.hiddenLayerBackward()
        self.assertEqual(_mindSingle.deltaHiddenChange.shape, _mindSingle.hiddenWeights.shape)
        
    def test_single_inputLayerBackward_deltaInputSum(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.inputM, self.outputM, _neurons, 0.1)
        _mindSingle.randomInputWeights()
        _mindSingle.hiddenLayerSumForward()
        _mindSingle.outputSum()
        _mindSingle.hiddenLayerBackward()
        _mindSingle.inputLayerBackward()
        self.assertEqual(_mindSingle.deltaInputSum.shape, _mindSingle.inputHidden.shape)
    
    def test_single_inputLayerBackward_deltaInputChange(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.inputM, self.outputM, _neurons, 0.1)
        _mindSingle.randomInputWeights()
        _mindSingle.hiddenLayerSumForward()
        _mindSingle.outputSum()
        _mindSingle.hiddenLayerBackward()
        _mindSingle.inputLayerBackward()
        self.assertEqual(_mindSingle.deltaInputChange.shape, _mindSingle.weightsMatrix.shape)
        
    def test_single_train_weightsMatrix(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.inputM, self.outputM, _neurons, 0.1)
        _mindSingle.randomInputWeights()
        _initialWeights = _mindSingle.weightsMatrix
        _mindSingle.hiddenLayerSumForward()
        _mindSingle.outputSum()
        _mindSingle.hiddenLayerBackward()
        _mindSingle.inputLayerBackward()
        _mindSingle.adjustWeights()
        _mindSingle.train(10)
        self.assertEqual(_mindSingle.weightsMatrix.shape, _initialWeights.shape)
    
    def test_single_train_hiddenWeights(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.inputM, self.outputM, _neurons, 0.1)
        _mindSingle.randomInputWeights()
        _initialHiddenWeights = _mindSingle.hiddenWeights
        _mindSingle.hiddenLayerSumForward()
        _mindSingle.outputSum()
        _mindSingle.hiddenLayerBackward()
        _mindSingle.inputLayerBackward()
        _mindSingle.adjustWeights()
        _mindSingle.train(10)
        #print _mindSingle.weightsMatrix.shape
        self.assertEqual(_mindSingle.hiddenWeights.shape, _initialHiddenWeights.shape)
        
    def test_single_error_greaterZero(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.inputM, self.outputM, _neurons, 0.1)
        _mindSingle.randomInputWeights()
        _initialHiddenWeights = _mindSingle.hiddenWeights
        _mindSingle.hiddenLayerSumForward()
        _mindSingle.outputSum()
        _mindSingle.hiddenLayerBackward()
        _mindSingle.inputLayerBackward()
        _mindSingle.adjustWeights()
        _mindSingle.train(100)
        self.assertGreater(_mindSingle.modelError(), 0)
        
    def test_single_error_oneLower(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.inputM, self.outputM, _neurons, 0.1)
        _mindSingle.randomInputWeights()
        _initialHiddenWeights = _mindSingle.hiddenWeights
        _mindSingle.hiddenLayerSumForward()
        _mindSingle.outputSum()
        _mindSingle.hiddenLayerBackward()
        _mindSingle.inputLayerBackward()
        _mindSingle.adjustWeights()
        _mindSingle.train(100)
        print _mindSingle.predict([1,1,1])
        self.assertLess(_mindSingle.modelError(), 1)

if __name__ == "__main__":
    unittest.main()
    

