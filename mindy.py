import numpy as np
import traceback

def sigmoid(y):
    """helper function to calculate sigmoid"""
    _sigmoid = 1 / (1 + np.matrix(np.exp(-y)))
    return _sigmoid

def dsigmoid(x):
    """sigmoid prime"""
    return x * (1 - x.T)

class Mindy():
    """Class for a simple neural network, Still under construction. NB: Only one hidden layer"""
    def __init__(self, inputData, outputData, neurons, learningRate, **kwargs):
        """initializor, take input and output data and constructs a numpy array, and accepts the amount of intitial neurons"""
        try:
            self.inputM = np.matrix(inputData)
            self.outputM = np.matrix(outputData)
            self.learningRate = learningRate
        except Exception as err:
            traceback.print_exc()
        try:
            self.neurons = int(neurons)
        except Exception as err:
            traceback.print_exc()
        
        self.yCats = 0 #Index value -> means 1 vector for y
        self.ovr = False
        try:
            for key, value in kwargs.iteritems():
                if key == 'multinomial':
                    if value == 'ovr':
                        self.ovr = True
                        self.oneVsRestCategorial(np.array(self.outputM))
        except Exception as err:
            traceback.print_exc()
            
        

        
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
        self.outputSum()
        
    def outputSum(self):
        """Calculates the outut som from the hidden layer"""
        self.residual = self.outputM - self.predictedOutput
        self.deltaOutputSum =  dsigmoid(sigmoid(self.sumOfHiddenLayer)) * self.residual
        self.hiddenLayerBackward()
    
    def hiddenLayerBackward(self):
        """calculates new hidden weights as part of the backward propagation"""
        self.deltaHiddenChange = self.hiddenLayerResult.T * self.deltaOutputSum * self.learningRate
        self.inputLayerBackward()
            
    def inputLayerBackward(self):
        """Calculates the new and hopefully improved initial input weights as part of the backward propagation"""
        self.deltaInputSum = (self.deltaOutputSum * self.hiddenWeights.T) * dsigmoid(sigmoid(self.inputHidden.T)).T
        self.deltaInputChange = self.inputM.T * self.deltaInputSum * self.learningRate
        self.adjustWeights()
        
    def adjustWeights(self):
        self.hiddenWeights += self.deltaHiddenChange
        self.weightsMatrix += self.deltaInputChange

    def train(self, _iterations):
        """Thinker for the mind""" 
        try:
            if self.ovr:
                _i = 0
                while _i < self.yCats:
                    self.randomInputWeights()
                    self.outputM = self.outputOvr[:, _i]
                    self.trainingIterations(_iterations)
                    if _i == 0:
                        self.weightsOvr = np.array(self.weightsMatrix)
                        self.hiddenWeightsOvr = np.array(self.hiddenWeights)
                    elif _i > 0:
                        try:
                            self.weightsOvr = np.stack((self.weightsOvr, np.array(self.weightsMatrix)), axis=0)
                            self.hiddenWeightsOvr = np.stack((self.hiddenWeightsOvr, np.array(self.hiddenWeights)), axis=0)
                        except:
                            self.weightsOvr = np.concatenate((self.weightsOvr, np.array(self.weightsMatrix)[np.newaxis, :]), axis=0)
                            self.hiddenWeightsOvr = np.concatenate((self.hiddenWeightsOvr, np.array(self.hiddenWeights)[np.newaxis, :]), axis=0)
                    _i += 1
            if not self.ovr:
                self.randomInputWeights()
                self.trainingIterations(_iterations)

        except Exception as err:
            traceback.print_exc()

        self.yhat = np.mean(self.predictedOutput, axis=0)
        self.residual = np.mean(self.residual, axis=0)

    def trainingIterations(self, _iterations):
        """For training iterations"""
        _i = 0
        while _i < _iterations:
            try:
                self.hiddenLayerSumForward()

            except Exception as err:
                traceback.print_exc()
                break
            _i += 1
            
            
    def predict(self, _input):
        """Using the weights and predicting the output"""
        if self.ovr:
            _i = 0
            while _i < self.yCats:
                try:
                    _inputToHidden = sigmoid(np.matrix(_input) * self.weightsOvr[_i, :, :])
                    if _i == 0:
                        _hiddenToOutput = sigmoid(_inputToHidden * self.hiddenWeightsOvr[_i, :, :])
                    elif _i > 0:
                        _hiddenToOutput = np.hstack((_hiddenToOutput, sigmoid(_inputToHidden * self.hiddenWeightsOvr[_i, :, :])))
                    _i += 1
                    
                except Exception as err:
                    traceback.print_exc()
                    raise ValueError('Input for prediction does not match number of input variables in the network')
            _predVal = {'Prediction' : 0, 'Likelihood': 0}
            _index = self.unique
            if len(_hiddenToOutput.T) == len(_index):
                for _i, _j in zip(_hiddenToOutput.T, _index):
                    if 0.5 <= _i and 1 >= _i:
                        if _predVal['Prediction'] == 0:
                            _predVal['Prediction'], _predVal['Likelihood'] = _j, round(_i.item(), 3) 
                        elif _predVal['Prediction'] != 0 and _predVal['Likelihood'] < _i:
                            _predVal['Prediction'], _predVal['Likelihood'] = _j, round(_i.item(), 3)
                        elif _predVal['Prediction'] != 0 and _predVal['Likelihood'] == _i:
                            raise ValueError('Prediction error, two values are predicted equally')
                    elif _j == 0 and _i < 0.5:
                        _predVal['Likelihood'] = round(_i.item(), 3)
                    
                    
            else:
                raise ValueError('Predction problem, trying to predict a value, that doesnt exist')
                        
                        
                    #Add the looping to choose the highest value if it is larger than 0.5, otherwise choose 0 sa prediction value
        elif not self.ovr:
            try: 
                _inputToHidden = sigmoid(np.matrix(_input) * self.weightsMatrix)
                _hiddenToOutput = sigmoid(_inputToHidden * self.hiddenWeights)
                _predVal = {'Prediction' : 0, 'Likelihood': 0}
                if _hiddenToOutput < 0.5:
                    _predVal['Likelihood'] = round(_hiddenToOutput.item(), 3)
                elif _hiddenToOutput >= 0.5:
                    _predVal['Prediction'], _predVal['Likelihood'] = 1, round(_hiddenToOutput.item(), 3)
            except Exception as err:
                traceback.print_exc()
                raise ValueError('Input for prediction does not match number of input variables in the network')
        
        return _predVal, _hiddenToOutput
    
    def modelError(self):
        """Computates the root-mean-squared error for the network. The lower the better"""
        
        if self.ovr:
            pass
        elif not self.ovr:
            _numerator = np.sum(self.residual)**2
            _denominator = np.sum(self.predictedOutput - self.outputM)**2
            _rmse = _numerator / _denominator
            _error = _rmse
        return _error
        
        
    def oneVsRestCategorial(self, _output):
        """If the input is categorial, and "one vs rest" is the desired analysis"""
        self.unique, _counts = np.unique(_output, return_counts=True)
        _uniqueCounts = dict(zip(self.unique, _counts))
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
       
            
                
                
            