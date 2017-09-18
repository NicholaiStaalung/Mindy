# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:49:51 2017

@author: Nstaa
"""
import numpy as np
import traceback


class OVR():
    def __init__(self, mindy):
        """Constructor"""
        self.mindy = mindy
        
        
    def oneVsRestCategorial(self, _output):
        """If the input is categorial, and "one vs rest" is the desired analysis"""
        self.unique, _counts = np.unique(_output, return_counts=True)
        _m = 0 #Indicator for prepping the matrix or stacking the matrix
        for _p in self.unique:
            _j = 0
            _prepForBin = np.copy(_output)
            for _i in _prepForBin:
                if _i != _p:
                    _prepForBin[_j] = 0
                else:
                    _prepForBin[_j] = 1
                _j += 1
            if _m == 0:
                _prepBinForMatrix = np.array(_prepForBin)
                _m += 1
            else:
                _prepBinForMatrix = np.hstack((_prepBinForMatrix, _prepForBin))
        self.outputOvr = np.matrix(_prepBinForMatrix)
        self.yCats = len(self.unique)