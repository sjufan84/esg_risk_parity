import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

from matplotlib import pyplot

def shallow_neural(X_train_scaled, y_train, X_test_scaled, y_test, n_epochs=75):
    number_output_neurons = 1
    number_input_features=X_train_scaled.shape[1]
    hidden_nodes_layer1= (number_input_features + number_output_neurons)
    hidden_nodes_layer2= (hidden_nodes_layer1 + number_output_neurons)/1.5
    
    
    # shallow model

    model=Sequential()
    model.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation = "relu"))
    model.add(Dense(units=hidden_nodes_layer2, activation = "relu"))
    model.add(Dense(units=hidden_nodes_layer2/2, activation = "relu"))
    model.add(Dense(units=hidden_nodes_layer2/4, activation = "relu"))
    model.add(Dense(units=hidden_nodes_layer2/8, activation = "relu"))
    model.add(Dense(units=hidden_nodes_layer2/16, activation = "relu"))
    model.add(Dense(units=1, activation = "tanh"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1, patience = 5)
    
    fit_model = model.fit(X_train_scaled, y_train, validation_split=.3, epochs=n_epochs)
    
    _, train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    
    pyplot.plot(fit_model.history['loss'], label='train')
    pyplot.plot(fit_model.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    pyplot.plot(fit_model.history['accuracy'], label='train')
    pyplot.plot(fit_model.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()
    
    trained_predictions = pd.DataFrame((model.predict(X_test_scaled)))
    
    return trained_predictions


def deep_neural(X_train_scaled, y_train, X_test_scaled, y_test, n_epochs=50):
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
    model.add(Dense(units=1, activation = "tanh"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    es = EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1, patience = 25)
    
    fit_model = model.fit(X_train_scaled, y_train, validation_split=.3, epochs=n_epochs)
    
    _, train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    
    pyplot.plot(fit_model.history['loss'], label='train')
    pyplot.plot(fit_model.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    pyplot.plot(fit_model.history['accuracy'], label='train')
    pyplot.plot(fit_model.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()
    
    trained_predictions = pd.DataFrame((model.predict(X_test_scaled)))
    
    return trained_predictions