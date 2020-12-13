#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 9 15:07:24 2020

@author: alfredocu
"""

import numpy as np

###############################################################################

class  LogisticNeuron:
    def __init__(self, n_inputs, learning_rate = 0.1):
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()
        self.eta = learning_rate # Factor de aprendizaje.


    # Probabilidad.
    def predict_proba(self, X):
        Z = np.dot(self.w, X) + self.b # Propagación.
        Y_est = 1 / (1 + np.exp(-Z))
        return Y_est
    
    
    # Predicción.
    def predict(self, X):
        Z = np.dot(self.w, X) + self.b # Propagación.
        Y_est = 1 / (1 + np.exp(-Z))
        return 1 * (Y_est > 0.5)
    
    
    # Entrenamiento.
    def train(self, X, Y, epochs=50):
        p = X.shape[1] # Obtener la cantidad de patrones.
        
        for _ in range(epochs):            
            y_est = self.predict_proba(X) # Probabilidad.
            self.w += (self.eta / p) * np.dot((Y - y_est), X.T).ravel()
            self.b += (self.eta / p) * np.sum(Y - y_est)
                  

###############################################################################
        
# X datos.  
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

# Y Valores deseados.
# Y = np.array([0, 0, 0, 1]) # COMPUERTA AND.
Y = np.array([0, 1, 1, 1]) # COMPUERTA OR.
# Y = np.array([0, 1, 1, 0]) # COMPUERTA XOR.

neuron = LogisticNeuron(2, 1)

###############################################################################

# Entrenamineto.

# Sin entrenar.
# print("Predict: ", neuron.predict_proba(X))

# Entrenado.
# neuron.train(X, Y, 1000)
# print("Train: ", neuron.predict_proba(X))

###############################################################################

# Categorias.

# Sin entrenar.
print("No train: ", neuron.predict(X))

# Entrenado.
neuron.train(X, Y, 1000)
print("Train: ", neuron.predict(X))

###############################################################################