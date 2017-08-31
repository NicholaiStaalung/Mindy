# Mindy 

A binary multilayer recurrent neural network - currently with only one hidden layer.

This is a hobby project. It will be updated regularly, since its still under construction.

## Get started

Clone or fork the repo

### 1st step
Organize a (n x m) numpy array of explanatory variables as input data (vertical rows and horizontal columns) and a (n x 1) numpy array as response variable with output data.

If outout varaiables are not binary, add the OVR module, for One Vs Rest. 

Example with binary outpu variable
```
inputM = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [1, 0, 1],
    [0, 0, 0],
    [1, 1, 1]
])

outputM = np.array([
    [0],
    [1],
    [1],
    [0],
    [1]
])
```
### 2nd step

Choose a number of neurons and a learning rate

Example
```
learningRate = 0.1
neurons = 2000
```

### 3rd step
Prepare the neural network

Binary output
```
nn = NeuralNetwork(inputM, outputM, neurons, learningRate) 
```

Categorial output (3 or more categories)
```
nn = NeuralNetwork(inputM, outputM, neurons, learningRate, multinomial='ovr')
```
### 4th step 
Train the neural network with a given set of iterations

Example
```
nn.train(10000)
```


### 5th step
Predict an outcome from a given situation 

Example (Should predict 1 in this case)
```
nn.predict([1,1,1])
```

### 6th step
Get the model error (The lower the better)
```
nn.modelError()
```

