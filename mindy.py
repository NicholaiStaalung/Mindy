import numpy as np
import traceback
from lib.feeder import Feeder
from lib.ovr import OVR
from lib.helper.functions import sigmoid, costFunction
from lib.predictor import Predictor

class Mindy():
    """Class for a simple neural network, Still under construction. NB: Only one hidden layer"""
    def __init__(self, inputData, outputData, neurons, learningRate, **kwargs):
        """initializor, take input and output data and constructs a numpy array, and accepts the amount of intitial neurons"""
        try:
            self.inputM = inputData
            self.outputM = outputData
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
                        self.OVR = OVR(self)
                        self.OVR.oneVsRestCategorial(np.array(self.outputM))
        except Exception as err:
            traceback.print_exc()
    
    def train(self, _iterations):
        """Thinker for the mind""" 
        try:
            if self.ovr:
                self.yCats = self.OVR.yCats
                _i = 0
                while _i < self.yCats:
                    """Adding 3 dimensional matrices as weigths with shape in 3rd dim corresponding to number of categories"""
                    self.Feeder = Feeder(self)
                    self.Feeder.randomWeights()
                    self.outputM = self.OVR.outputOvr[:, _i]
                    self.trainingIterations(_iterations)
                    
                    if _i == 0:
                        self.weightsOvr = np.array(self.Feeder.inputWeights)
                        self.hiddenOneWeightsOvr = np.array(self.Feeder.hiddenOneWeights)
                    
                    elif _i > 0:
                        try: #because stack does not let you add a 2 dim to a 3 dim, but concatenate will. Stack only adds a new dim
                            self.weightsOvr = np.stack((self.weightsOvr, np.array(self.Feeder.inputWeights)), axis=0)
                            self.hiddenOneWeightsOvr = np.stack((self.hiddenOneWeightsOvr, np.array(self.Feeder.hiddenOneWeights)), axis=0)
                        except:
                            self.weightsOvr = np.concatenate((self.weightsOvr, np.array(self.Feeder.inputWeights)[np.newaxis, :]), axis=0)
                            self.hiddenOneWeightsOvr = np.concatenate((self.hiddenOneWeightsOvr, np.array(self.Feeder.hiddenOneWeights)[np.newaxis, :]), axis=0)
                    
                    _i += 1
            if not self.ovr:
                self.Feeder = Feeder(self)
                self.Feeder.randomWeights()
                self.trainingIterations(_iterations)

        except Exception as err:
            traceback.print_exc()

        self.yhat = np.mean(self.Feeder.predictedOutput, axis=0)
        self.residual = np.mean(self.Feeder.residual, axis=0)

    def trainingIterations(self, _iterations):
        """For training iterations"""
        _i = 0
        while _i < _iterations:
            try:
                self.Feeder.feedForward(self.inputM, self.outputM)
                self.Feeder.feedBackwards(self.learningRate)

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
                    _predict = Predictor(self)
                    if _i == 0:
                        _hiddenToOutput = _predict.calcOutput(_input, self.weightsOvr[_i, :, :], self.hiddenOneWeightsOvr[_i, :, :])
                        
                    elif _i > 0:
                        _hiddenToOutput = np.hstack((_hiddenToOutput, _predict.calcOutput(_input, self.weightsOvr[_i, :, :], self.hiddenOneWeightsOvr[_i, :, :])))
                    _i += 1  
                except Exception as err:
                    traceback.print_exc()
                    raise ValueError('Error: %s -Input for prediction does not match number of input variables in the network' %(err)) 
            _predVal = _predict.chooseOvrPrediction(_hiddenToOutput)                  
        #Add the looping to choose the highest value if it is larger than 0.5, otherwise choose 0 sa prediction value
        elif not self.ovr:
            _predict = Predictor(self)
            _hiddenToOutput = _predict.calcOutput(_input, self.Feeder.inputWeights, self.Feeder.hiddenOneWeights)
            _predVal = _predict.chooseBinPrediction(_hiddenToOutput)
        return _predVal
    
        
    
        
        

       
                
        
        
        
        
        
        
        
        