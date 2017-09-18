# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 20:31:05 2017

@author: Nstaa
"""

import unittest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from mindy import Mindy
import numpy as np
from mysqlQ.mysqlScript import mysqlQuery



class TestOvrMindy(unittest.TestCase):
    
    
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
        [2],
        [1]
        ])
        
        
    def test_ovr(self):

        self.assign_io_data()
        _mind = Mindy(self.inputM, self.outputM, 500, 0.1, multinomial='ovr')
        
        _unique, _counts = np.unique(self.outputM, return_counts=True)
        _len = len(_mind.OVR.outputOvr.T)
        self.assertEqual(_len, len(_unique))
        
    def test_ovrOutputVectors(self):
        self.assign_io_data()
        _mind = Mindy(self.inputM, self.outputM, 500, 0.1, multinomial='ovr')
        

    
    def test_ovrVsSingle(self):
        
        
        self.assign_io_data()
        #OVR
        _neurons = 1
        _trainingIt = 1
        _mind = Mindy(self.inputM, self.outputM, _neurons, 0.1, multinomial='ovr')

        
        _mind.train(_trainingIt)
        _predictionObs = [1,1,1] #remember the bias beta_0
        
        _i = 0
        while _i < len(_mind.OVR.outputOvr.T):
            
            _outSingle = np.array(_mind.OVR.outputOvr[:, _i])
            
            #Single
            _mindSingle = Mindy(self.inputM, _outSingle, _neurons, 0.1)
            _mindSingle.train(_trainingIt)

            if _i == 0:
                _singlePredict = _mindSingle.predict(_predictionObs)
            elif _i > 0:
                _singlePredict = np.hstack((_singlePredict, _mindSingle.predict(_predictionObs)))
            if len(_mind.OVR.outputOvr.T) < 3:
                raise ValueError ('Doesnt make sense with less than three variabels')
                exit
            _i += 1

        #self.assertEqual(_mind.predict(_predictionObs).tolist(), _singlePredict.tolist())
        
    def test_ovr_predict(self):
        self.assign_io_data()
        #OVR
        _neurons = 300
        _trainingIt = 1000
        _mind = Mindy(self.inputM, self.outputM, _neurons, 0.1, multinomial='ovr')
        _mind.train(_trainingIt)
        _predictionObs = [1,1,1] #remember the bias beta_0
        self.assertRaises(_mind.predict(_predictionObs))
        
if __name__ == "__main__":
    unittest.main()
    