# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:40:48 2017

@author: Nstaa
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from mindy import Mindy
import numpy as np


class MindyErrors():
    """Calculating model errors"""
    
    def __init__(self, mindy):
        """Constructor"""
        self.mindy = mindy
    
    def modelError(self):
        """Computates the root-mean-squared error for the network. The lower the better"""
        
        if self.mindy.ovr:
            pass
        elif not self.mindy.ovr:
            _numerator = np.sum(self.mindy.residual**2)
            _denominator = np.sum(self.mindy.outputM - np.sum(self.mindy.outputM) / len(self.mindy.inputM))**2
            _rmse = _numerator / _denominator
            _error = _rmse
        return _error