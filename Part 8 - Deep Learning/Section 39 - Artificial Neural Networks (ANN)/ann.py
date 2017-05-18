# Artificial Neural Network

# Installing Theano
# Librería de redes neuronales desarrollada por la universidad de Montreal
# Compatibilidad con CPU y GPU
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# pip install theano

# Installing Tensorflow
# Desarrollada por Google Brain, actualmente bajo Apache 2.0
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# Basada en las dos anteriores
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) # La columna países de X se ha hecho numérica
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) # La columna género de X se ha hecho numérica
onehotencoder = OneHotEncoder(categorical_features = [1]) # Se selecciona la columna países para descomponerla en dummy variables
X = onehotencoder.fit_transform(X).toarray() # Se efectúa la transformación. Las dummy variables pasan a estar en las primeras columnas
X = X[:, 1:] # Para evitar la dummy variable trap, esto es, que haya una columna de valores que no aporta información, se elimina una seleccionando todo el dataset de la columna 1 en adelante, desechando la nº 0

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Se usará una función de activación rectifier para las capas internas y una función sigmoidea para la última ( se quiere saber la probabilidad de que se abandone el banco)

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))
# Ctrl + I para inspeccionar los argumentos de Dense
# La elección de output_dim (nº de nodos) puede llegar a ser cuasi artística
# Sin experiencia, tomar la media de los nodos de entrada y los de salida, en este caso (11+1)/2

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# softmax sería la función de activación si hubiera más de dos categorías de salida

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# classification_crossentropy sería la función de coste si hubiera más de dos categorías de salida

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
# nb_epoch cambiará a epochs

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) # Se convierten las probabilidades a true o false para saber si hay posibilidades serias de que el cliente se marche o no

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)