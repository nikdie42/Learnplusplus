# Import aller benötigten Bibliotheken und Datensatz
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

# Einlesen und aufteilen der Daten	
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data() 
noOfClasses = 10

# Vorbereiten und verkleinern der Daten
YTrain = to_categorical(yTrain, noOfClasses)
YTest  = to_categorical(yTest, noOfClasses)
XTrain = xTrain/255.0
XTest  = xTest/255.0

#Regularisierung
l2Reg = 0.001

#Aufbau des Netztes
CNN = Sequential()
CNN.add(layers.Conv2D(32,(3,3),padding='same',activation='relu',kernel_regularizer=l2(l2Reg),input_shape=(32,32,3)))
CNN.add(layers.MaxPool2D(pool_size=(2, 2),padding='same'))
CNN.add(layers.Conv2D(32,(3,3),padding='same',activation='relu',kernel_regularizer=l2(l2Reg)))
CNN.add(layers.MaxPool2D(pool_size=(2, 2),padding='same'))
CNN.add(layers.Conv2D(64,(3,3),padding='same',activation='relu',kernel_regularizer=l2(l2Reg)))
CNN.add(layers.MaxPool2D(pool_size=(2, 2),padding='same'))
CNN.add(layers.Conv2D(64,(3,3),padding='same',activation='relu',kernel_regularizer=l2(l2Reg)))
CNN.add(layers.MaxPool2D(pool_size=(2, 2),padding='same'))
CNN.add(layers.Flatten())
CNN.add(layers.Dense(512,activation='relu',kernel_regularizer=l2(l2Reg)))
CNN.add(layers.Dense(256,activation='relu',kernel_regularizer=l2(l2Reg)))
CNN.add(layers.Dense(10,activation='softmax'))
#CNN.summary()
#Laden der Gewichte
CNN.load_weights("cifar10weights.h5")
CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#CNN.fit(XTrain,YTrain,epochs=20,batch_size=64)
scores = CNN.evaluate(XTest,YTest,batch_size=64)
#Ausgeben der Genauigkeit
print("Accuracy: %.2f%%" % (scores[1]*100))