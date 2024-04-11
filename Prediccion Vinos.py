# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:01:57 2024

@author: Ricardo
"""

#Importar las librerías necesarias:

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Importar el dataset:

dataset = pd.read_csv('C:/Users/Ricardo/Downloads/redes neuronales/winequality-red.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Binarizar la variable objetivo en 0 y 1
y = np.where(y >= 7, 1, 0)

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Escalado de variables
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Parte 2 - Construir la RNA
#Importar Keras y librerías adicionales:
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Inicializar la RNA
classifier = Sequential()

# Añadir las capas de entrada y primera capa oculta
classifier.add(Dense(units = 15, kernel_initializer = "uniform",  activation = "relu", input_dim = 11))
classifier.add(Dropout(rate = 0.2))

# Añadir la tercera capa oculta
classifier.add(Dense(units = 15, kernel_initializer = "uniform",  activation = "relu"))
classifier.add(Dropout(rate = 0.2))  

# Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer = "uniform",  activation = "sigmoid"))

# Compilar la RNA
classifier.compile(optimizer = "Adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train,  batch_size = 48, epochs = 500)

# Parte 3 - Evaluar el modelo y calcular predicciones finales
# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)
print(y_pred)
y_pred = (y_pred>0.5)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("El porcentaje de predicción es de: ", (cm[0][0]+cm[1][1])/cm.sum()*100)

from sklearn.preprocessing import StandardScaler

# Crear un objeto StandardScaler
sc = StandardScaler()
# Ajustar el escalador a tus datos de entrenamiento (suponiendo que X_train son tus características de entrenamiento)
sc.fit(X_train)


"""Acidez fija: 10.80
Acidez volátil: 1.16
Ácido cítrico: 0.60
Azúcares residuales: 8.86
Cloruros: 0.27
Dióxido de azufre libre: 46.86
Dióxido de azufre total: 129.84
Densidad: 1.0022
pH: 3.96
Sulfatos: 0.97
Alcohol: 13.55"""

print("La prediccion del ejemplo 1:", end="\n")

# Transformar los datos de entrada utilizando el escalador ajustado
new_data = sc.transform(np.array([[10.80, 1.16, 0.60, 8.86, 0.27, 46.86, 129.84, 1.0022, 3.96, 0.97, 13.55]]))
# Hacer predicciones utilizando los datos transformados
new_prediction = classifier.predict(new_data)
new_prediction = (new_prediction > 0.5)

if new_prediction:
    print("Alta calidad", end="\n")
else:
    print("Baja calidad", end="\n")

"""
Acidez fija: 9.31
Acidez volátil: 1.17
Ácido cítrico: 0.00
Azúcares residuales: 5.31
Cloruros: 0.10
Dióxido de azufre libre: 7.56
Dióxido de azufre total: 58.71
Densidad: 0.9948
pH: 3.24
Sulfatos: 1.23
Alcohol: 11.1  

"""

print("La prediccion del ejemplo 2:", end="\n")
# Transformar los datos de entrada utilizando el escalador ajustado
new_data2 = sc.transform(np.array([[9.31, 1.17, 0.00, 5.31, 0.10, 7.56, 58.71, 0.9948, 3.24, 1.23, 11.1]]))
# Hacer predicciones utilizando los datos transformados
new_prediction2 = classifier.predict(new_data2)
new_prediction2 = (new_prediction2 > 0.5)

if new_prediction2:
    print("Alta calidad", end="\n")
else:
    print("Baja calidad", end="\n")
