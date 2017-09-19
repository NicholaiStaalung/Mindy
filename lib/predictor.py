# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:39:06 2017

@author: Nstaa
"""
import numpy as np
import traceback
from helper.functions import sigmoid, dsigmoid


class Predictor():
    
    def __init__(self, mindy):
        """Constructor"""
        
        
        self.mindy = mindy
        
    def calcOutput(self, _input, _inputWeights, _hiddenOneWeights):
        """Calculates the output from the inputs and weights"""
        _inputToHidden = sigmoid(np.matrix(_input) * _inputWeights)
        _hiddenToOutput = sigmoid(_inputToHidden * _hiddenOneWeights)
        return _hiddenToOutput