# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:30:24 2017

@author: Nstaa
"""
import unittest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from lib.feeder import Feeder
from lib.helper.functions import costFunction
from mindy import Mindy
import numpy as np






class TestNumericalGradient(unittest.TestCase):
    
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

    def test_computeNumericalGradient(self):

        self.assign_io_data()
        _neurons = 10
        self.mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        self.computeNumericalGradient(1e-4, self.mindSingle.inputM, self.mindSingle.outputM)
        _numgrad = self.numgrad
        _grad = self.computeGradients()
        
        
        """VISUAL TEST. CHECK IF THE ARE ALMOST EQUAL"""
        print _numgrad
        print _grad
 
    def computeNumericalGradient(self, _epsilon, X, y):
        """Setting parameters to perform numerical gradient checking"""
        self.Feeder = Feeder(self.mindSingle)
        self.Feeder.randomWeights()
        _paramsInitial = np.hstack((np.array(self.Feeder.inputWeights).ravel(), np.array(self.Feeder.hiddenOneWeights).ravel()))
        self.numgrad = np.zeros(_paramsInitial.shape)
    
        _perturb = np.zeros(_paramsInitial.shape)
     
        for p in range(len(_paramsInitial)):
            _perturb[p] = _epsilon     
            self.resetParams(_paramsInitial + _perturb)
            self.Feeder.feedForward(X, y)
            _loss2 = costFunction(self.Feeder)
                     
            self.resetParams(_paramsInitial - _perturb)
            self.Feeder.feedForward(X, y)
            _loss1 = costFunction(self.Feeder)
                     
            #Computing numericak gradient
            self.numgrad[p] = (_loss2 - _loss1) / (2 * _epsilon) 
            
            #Return the value we changed back to zero
            _perturb[p] = 0

    
        self.resetParams(_paramsInitial)

    def computeGradients(self):
       self.Feeder.feedBackwards(1, _numgrad = True)
       return np.hstack((self.Feeder.deltaInputChange.ravel(), self.Feeder.deltaHiddenChange.ravel()))
        
    
    def resetParams(self, params):
        self.Feeder.inputWeights = np.matrix(params[:-self.mindSingle.neurons]).reshape(len(self.mindSingle.inputM.T), self.mindSingle.neurons)
        self.Feeder.hiddenOneWeights = np.matrix(params[-self.mindSingle.neurons:][:, np.newaxis])
    
if __name__ == "__main__":
    unittest.main()