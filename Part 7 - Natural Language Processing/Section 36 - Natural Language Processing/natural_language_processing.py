# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk # Natural Processing Tool Kit: Librería de Python para procesamiento de lenguaje
nltk.download('stopwords') # Conjunto de palabras irrelevantes
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # Toma las raíces de las palabras para eliminar tiempos verbales y simplificar
corpus = [] # Colección de texto, en este caso contendrá las reviews procesadas en el bucle
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) # Mantener todas las letras de la review y sustituir lo que no sean letras por espacios
    review = review.lower()
    review = review.split() # Se separa la cadena en una lista de elementos palabra
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # El conjunto stepwords es convertido en un set ya que recorrerlo es más rápido
    review = ' '.join(review) # Se vuelve a crear una cadena a partir de los tokens alterados
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # CountVectorizer también cuenta con muchos parámetros con los que limpiar el set de datos
#◙ Con max_features se filtran las 1500 palabras más frecuentes
X = cv.fit_transform(corpus).toarray() # Se genera una matriz X en la que cada columna representa una palabra y cada fila una review (1 si aparece y 0 si no)
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)