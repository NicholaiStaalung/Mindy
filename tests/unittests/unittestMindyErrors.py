# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:44:59 2017

@author: Nstaa
"""

import unittest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from lib.feeder import Feeder
from lib.helper.functions import costFunction
from lib.errors.mindyErrors import MindyErrors
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
               
    def test_single_error_greaterZero(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        _feed.feedForward(_mindSingle.inputM, _mindSingle.outputM)
        _feed.feedBackwards(0.1)
        _mindSingle.train(100)
        _errors = MindyErrors(_mindSingle)
        self.assertGreater(_errors.modelError(), 0)
        
    def test_single_error_oneLower(self):
        self.assign_io_data()
        _neurons = 10
        _mindSingle = Mindy(self.input, self.output, _neurons, 0.1)
        _feed = Feeder(_mindSingle)
        _feed.randomWeights()
        _feed.feedForward(_mindSingle.inputM, _mindSingle.outputM)
        _feed.feedBackwards(0.1)
        _mindSingle.train(100)
        _errors = MindyErrors(_mindSingle)
        self.assertLess(_errors.modelError(), 1)
        





        

if __name__ == "__main__":
    unittest.main()
    

