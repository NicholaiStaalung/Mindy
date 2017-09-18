# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:19:34 2017

@author: Nstaa
"""

import numpy as np
import traceback
from helper.functions import sigmoid, dsigmoid

class Feeder():
    """Helper for the Mindy class. Takes care of forward and backwards movements"""
    
    def __init__(self, mindy):
        """Constructor"""
        self.mindy = mindy
        
    
    def randomWeights(self):
        """If first iteration, this creates random input weights"""
        _mu, _sigma = 0, 0.1 # mean and standard deviation
        _inputWeights = np.random.normal(_mu, _sigma, len(self.mindy.inputM.T) * self.mindy.neurons)
        self.inputWeights = np.array(_inputWeights).reshape(len(self.mindy.inputM.T), self.mindy.neurons)
        _hiddenOneWeights = np.random.normal(_mu, _sigma, self.mindy.neurons)[np.newaxis, :]
        self.hiddenOneWeights = np.matrix(_hiddenOneWeights).T #One column for n neurons
        
        
    def feedForward(self, X, y):
        """Performing forward propagation"""
        self.inputToHiddenOne = np.dot(X, self.inputWeights) #notation z^(2) nXm inputs mXp weights = nXp layer
        self.hiddenLayer = sigmoid(self.inputToHiddenOne) #notation a^(2) nXp layer (Activity of each synaps)
        self.hiddenOneToOutput =  np.dot(self.hiddenLayer, self.hiddenOneWeights) #notation z^(3) nXp layer pX1 weights
        self.predictedOutput = sigmoid(self.hiddenOneToOutput) #notation yhat nX1 outputs
        
        self.residual = np.array(y - self.predictedOutput) #Predition error nX1 residuals
        
    def feedBackwards(self, learningRate, _numgrad = False):
        """Controller for backward propagation"""
        self.hiddenLayerBackward(learningRate)
        self.inputLayerBackward(learningRate)
        if not _numgrad:
            self.adjustWeights()

    def hiddenLayerBackward(self, learningRate):
        """calculates new hidden weights as part of the backward propagation"""
        self.deltaOutputSum =  np.multiply(self.residual, dsigmoid(sigmoid(self.hiddenOneToOutput))) #element wise
        self.deltaHiddenChange = self.hiddenLayer.T * self.deltaOutputSum * learningRate
            
    def inputLayerBackward(self, learningRate):
        """Calculates the new initial input weights as part of the backward propagation"""
        self.deltaInputSum = np.multiply(np.array(np.dot(self.deltaOutputSum,  self.hiddenOneWeights.T)), dsigmoid(sigmoid(self.inputToHiddenOne))) #matrix calc and elementwise
        self.deltaInputChange = np.dot(self.mindy.inputM.T,  self.deltaInputSum) * learningRate
        
    def adjustWeights(self):
        self.hiddenOneWeights += self.deltaHiddenChange
        self.inputWeights += self.deltaInputChange
