import numpy as np

def sigmoid(y):
    """helper function to calculate sigmoid"""
    _sigmoid = 1 / (1 + np.exp(-y))
    return _sigmoid

def dsigmoid(x):
    """sigmoid prime. Param has to be a sigmoid function"""
    return np.multiply(x, (1 - x))

def costFunction(obj):
    return np.sum(obj.residual**2 / 2)