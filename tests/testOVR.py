# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 20:31:05 2017

@author: Nstaa
"""

import unittest
import sys
sys.path.append('C:\Python27')
from mysqlQ.mysqlScript import mysqlQuery
from Mindy.mindy import Mindy
import numpy as np


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
        _len = len(_mind.outputOvr.T)
        self.assertEqual(_len, len(_unique))
    
    def test_ovrVsSingle(self):
        
        
        self.assign_io_data()
        #OVR
        _neurons = 100
        _trainingIt = 10000
        _mind = Mindy(self.inputM, self.outputM, _neurons, 0.1, multinomial='ovr')

        
        _mind.train(_trainingIt)
        
        _i = 0
        while _i < len(_mind.outputOvr.T):
            
            _outSingle = np.array(_mind.outputOvr[:, _i])
            
            #Single
            _mindSingle = Mindy(self.inputM, _outSingle, _neurons, 0.1)
            _mindSingle.train(_trainingIt)

            if _i == 0:
                _singlePredict = _mindSingle.predict([1,1,1])
            elif _i > 0:
                _singlePredict = np.hstack((_singlePredict, _mindSingle.predict([1,1,1])))
            if len(_mind.outputOvr.T) == 2:
                print 'Doesn make sense with less than three variabels'
                exit
            _i += 1

        self.assertEqual(_mind.predict([1,1,1]).tolist(), _singlePredict.tolist())
                
        
if __name__ == "__main__":
    unittest.main()
    