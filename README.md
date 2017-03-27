# Predicting-Unemployment-with-NNs
simplest model with 1 hidden unit

NOTE: Data downloaded from FRED St. Louis

This is the simplest neural network: 1 hidden unit, and it already learns the pattern well enough to output a time series almost identical aside form a scalar factor. The input is time series of either zeroes, ones, or np.random.rand(len(x)). It doesn't matter, because the network learns to adjust its weights accordingly given a loss function which is the squared distance from the desired output, which is unemployment. Note that this is the most simple possible model because we use no other regressions or economic predictors. Only one time series, and it is only used in the cost function calculation. The rest of the output is thanks to the hidden unit learning the general seasonal waviness of the unemployment rate graph. 

![alt tag](https://github.com/ConsciousMachines/Predicting-Unemployment-with-NNs/blob/master/graf.gif)

