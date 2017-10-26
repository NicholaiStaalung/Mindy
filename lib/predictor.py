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
    
    def chooseOvrPrediction(self, _hiddenToOutput):
        """Chooses the best prediction for the categorial ouput data"""
        _predVal = {'Prediction' : 0, 'Likelihood': 0}
        _index = self.mindy.OVR.unique
        if len(_hiddenToOutput.T) == len(_index):
            for _i, _j in zip(_hiddenToOutput.T, _index):
                if 0.5 <= _i and 1 >= _i:       
                    if _j > 0: #We are only returning the highest predicted value
                        _predVal['Prediction'], _predVal['Likelihood'] = _j, round(_i.item(), 3) 
                    elif _predVal['Prediction'] != 0 and _predVal['Likelihood'] < _i:
                        _predVal['Prediction'], _predVal['Likelihood'] = _j, round(_i.item(), 3)
                    elif _predVal['Prediction'] != 0 and _predVal['Likelihood'] == _i:
                        raise ValueError('Prediction error, two values are predicted equally')
                elif _j == 0 and _i < 0.5:
                    _predVal['Likelihood'] = round(_i.item(), 3)     
        else:
            raise ValueError('Predction problem, trying to predict a value, that doesnt exist')
        return _predVal
    
    def chooseBinPrediction(self, _hiddenToOutput):
        """Choosing the prediction for the binary output data"""
        try: 
            _predVal = {'Prediction' : 0, 'Likelihood': 0}
            if _hiddenToOutput < 0.5:
                _predVal['Likelihood'] = round(_hiddenToOutput.item(), 3)
            elif _hiddenToOutput >= 0.5:
                _predVal['Prediction'], _predVal['Likelihood'] = 1, round(_hiddenToOutput.item(), 3)
        except Exception as err:
            traceback.print_exc()
            raise ValueError('Error: %s - Input for prediction does not match number of input variables in the network' %(err))
        return _predVal