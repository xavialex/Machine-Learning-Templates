# Apriori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None) # Se especifica el header = None para no perder la primera línea
# Conversión del dataset a una lista de listas
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
# min_support: El support de un producto que es adquirido 3 veces al día: 3*7/7501
# min_confidence: Buena combinación para min_support
# min_lift: Porque sí, hacer pruebas para obtener buenos valores para los parámetros

# Visualising the results
results = list(rules)
myResults = [list(x) for x in results]
# La lista se encuentra ordenada de mayor a menor relevancia (lift)
# En frozenset aparecen los productos que están relacionados
# Hacer doble click en la lista que aparece para cada valor hasta llegar a la última
# El primer valor numérico es la confidence, en el primer caso 0.29, por lo que la gente que compre light cream tiene un 29 % de probabilidades de comprar pollo
# El segundo y último es el lift. Un valor de 4.84 es alto con respecto a nuestro límite inferior (3), por lo que tiene sentido que sea la asociación más fuerte del dataset
