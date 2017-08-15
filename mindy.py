import numpy as np
import traceback

def sigmoid(y):
    """helper function to calculate sigmoid"""
    return 1 / (1 + np.matrix(np.exp(-y)))

def dsigmoid(x):
    """sigmoid prime"""
    return x * (1 - x).T

class NeuralNetwork():
    """Class for a simple neural network, Still under construction. NB: Only one hidden layer"""
    def __init__(self, inputData, outputData, neurons, learningRate):
        """initializor, take input and output data and constructs a numpy array, and accepts the amount of intitial neurons"""
        try:
            self.inputM = np.matrix(inputM)
            self.outputM = np.matrix(outputM)
            self.learningRate = learningRate
        except:
            print "Cant process input and output data, check validity"
        try:
            self.neurons = int(neurons)
        except:
            print "define neurons as integers"
        

        
    def randomInputWeights(self):
        """If first iteration, this creates random input weights"""
        self.mu, self.sigma = 0, 0.1 # mean and standard deviation
        _weights = np.random.normal(self.mu, self.sigma, len(self.inputM.T) * self.neurons)
        self.weightsMatrix = np.matrix(_weights).reshape(len(self.inputM.T), self.neurons)
        _hiddenWeights = np.random.normal(self.mu, self.sigma, self.neurons)[np.newaxis, :]
        self.hiddenWeights = np.matrix(_hiddenWeights).T #One column for n neurons
    
    def hiddenLayerSumForward(self):
        """Calculates the hidden inputs based on the weights and initial inputs as part of a forward propagation 1st step"""
        self.inputHidden = self.inputM * self.weightsMatrix
        self.hiddenLayerResult = sigmoid(self.inputHidden)
        self.sumOfHiddenLayer =  self.hiddenLayerResult * self.hiddenWeights
        self.predictedOutput = sigmoid(self.sumOfHiddenLayer)
        
    def outputSum(self):
        """Calculates the outut som from the hidden layer"""
        self.residual = self.outputM - self.predictedOutput
        self.deltaOutputSum =  dsigmoid(sigmoid(self.sumOfHiddenLayer)) * self.residual
    
    def hiddenLayerBackward(self):
        """calculates new hidden weights as part of the backward propagation"""
        self.deltaHiddenChange = self.hiddenLayerResult.T * self.deltaOutputSum * self.learningRate
        
            
    def inputLayerBackward(self):
        """Calculates the new and hopefully improved initial input weights as part of the backward propagation"""
        self.deltaInputSum = (self.deltaOutputSum * self.hiddenWeights.T).T * dsigmoid(sigmoid(self.inputHidden))
        self.deltaInputChange = self.deltaInputSum * self.inputM * self.learningRate
        
        
    def adjustWeights(self):
        self.hiddenWeights += self.deltaHiddenChange
        self.weightsMatrix += self.deltaInputChange.T
        
    def train(self, _iterations):
        """Thinker for the mind"""

        self.randomInputWeights()
        
        _i = 0
        while _i < _iterations:
            try:
                self.hiddenLayerSumForward()
                self.outputSum()
                self.hiddenLayerBackward()
                self.inputLayerBackward()
                self.adjustWeights()
            except Exception as err:
                traceback.print_exc()
                break

            if _i == 0 or _i == _iterations - 1:
                print "yhat: %s" %(np.mean(self.predictedOutput, axis=0))
                print "residual: %s" %(np.mean(self.residual, axis=0))
            _i += 1

    def predict(self, _input):
        """Using the weights and predicting the output"""
        _inputToHidden = sigmoid(np.matrix(_input) * self.weightsMatrix)
        _hiddenToOutput = sigmoid(_inputToHidden * self.hiddenWeights)
        return _hiddenToOutput