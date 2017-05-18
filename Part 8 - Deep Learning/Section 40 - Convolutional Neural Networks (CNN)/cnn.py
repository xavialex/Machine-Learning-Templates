# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()
# Dos maneras de inicializar la CNN (al igual que las ANN): como secuencia de capas o como grafo

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
# 32 filtros de tamaño 3x3
# input_shape: Tamaño de las imágenes. 64x64 a color (3 matrices en lugar de una (blanco y negro))
# El orden original (Ctrl+I) es para Theano, pero al usar TensorFlow se cambia a como está ahora(64, 64, 3)
# activation: Rectifier

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2))) # Reducción de los feature maps

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu')) # output_dim fruto de la experimentación (ni muy grande ni muy pequeño, escoger potencias de 2)
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
# La función de activación de la última capa cambia a sigmoidea para obtener la probabilidad de que la foto sea de un perro o un gato

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# optimizer: adam, se refiere a Stochastic Gradient Descent
# loss: función de costes de cross-entropy

# Part 2 - Fitting the CNN to the images
# Documentación de Keras: https://keras.io/
from keras.preprocessing.image import ImageDataGenerator
# En concreto https://keras.io/preprocessing/image/
# Example of using .flow_from_directory(directory):
    
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64), # Tamaño de las imágenes esperadas por la CNN como figura arriba
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000, # Nº de imágenes en el conjunto de entrenamiento
                         epochs = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000) # Nº de imágenes en el conjunto de test