import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def shallow_neural(X_train_scaled, y_train, X_test_scaled, y_test, n_epochs=150, debug=1):
    number_output_neurons = 1
    number_input_features=X_train_scaled.shape[1]
    hidden_nodes_layer1= (number_input_features + number_output_neurons)//2
    hidden_nodes_layer2= (hidden_nodes_layer1 + number_output_neurons)//2
    
    # shallow model

    model=Sequential()
    model.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation = "relu"))
    model.add(Dense(units=hidden_nodes_layer2, activation = "relu"))
    model.add(Dense(units=1, activation = "tanh"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    X_train_scaled_array=np.asarray(X_train_scaled).astype(np.int)

    y_train_array=np.asarray(y_train).astype(np.int)
    
    fit_model = model.fit(X_train_scaled_array, y_train_array, validation_split=.3, epochs=n_epochs, verbose=debug)
    
    trained_predictions = pd.DataFrame((model.predict(X_test_scaled)), index=y_test.index)
    
    return trained_predictions


def deep_neural(X_train_scaled, y_train, X_test_scaled, y_test, n_epochs=150, debug=1):
    number_output_neurons = 1
    number_input_features=X_train_scaled.shape[1]
    hidden_nodes_layer1= (number_input_features + number_output_neurons)//2
    hidden_nodes_layer2= (hidden_nodes_layer1 + number_output_neurons)//2
    
    # shallow model

    model=Sequential()
    model.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation = "relu"))
    model.add(Dense(units=hidden_nodes_layer2, activation = "relu"))
    model.add(Dense(units=hidden_nodes_layer2//4, activation = "relu"))
    model.add(Dense(units=hidden_nodes_layer2//8, activation = "relu"))
    model.add(Dense(units=hidden_nodes_layer2//16, activation = "relu"))
    model.add(Dense(units=1, activation = "tanh"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    
    
    fit_model = model.fit(X_train_scaled, y_train, validation_split=.3, epochs=n_epochs, verbose=debug)
    
    trained_predictions = pd.DataFrame((model.predict(X_test_scaled)), index=y_test.index)
    
    return trained_predictions