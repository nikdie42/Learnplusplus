#Learn++
import pandas as pd
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
import math
import random

from tempfile import TemporaryFile

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# laden der aufgeteilten Kategorien
Testset = np.load('TestsetCifar10.npy')
Trainingsset = np.load('TrainingssetCifar10.npy')
yTest = np.load('yTestset.npy')
yTraining = np.load('yTrainingsset.npy')
data=Testset[2][3]

# Erstellen eines zufälligen Test- und Trainings Subset

def addy(dataset):
    outfile = TemporaryFile()
    i=0
    newdata=[]
    weight=1/6000
    for bildtype in dataset:
        newdata.append([])
        j=0
        for picture in bildtype:
            newdata[i].append([dataset[i][j],i,weight])
            j=j+1
        i=i+1
    return newdata

def split_dataset(K,dataset):
    datasize=math.floor(6000/K)
    newdataset=[]
    for i in range(K):
        newdataset.append([])
        for j in range(5):
            newdataset[i].append(dataset[j][i*datasize:(i+1)*datasize])
    return newdataset

def prepare_data(K,newdataset):
    prepared_data=[]
    for i in range(K):
        prepared_data.append(newdataset[i][0]+newdataset[i][1]+newdataset[i][2]+newdataset[i][3]+newdataset[i][4])
        shuffle(prepared_data[i])
    prepared_data=np.asarray(prepared_data)
    return prepared_data

def getnerate_random_dataset(dataset,trainsize,testsize, K, datasize):
    trainx=[]
    trainy=[]
    trainpicturenumbers=[]
    testx=[]
    testy=[]
    testpicturenumbers=[]
    
    for i in range(trainsize):
        compare_weight=20
        picturnumber = 0
        picturtype = 0
        while 1:
            picturtype=random.randrange(5)
            picturnumber=random.randrange(datasize)
            compare_weight=random.uniform(0, 1)
            if compare_weight <= dataset[picturtype][picturnumber][2] * 1000:
                trainx.append(dataset[picturtype][picturnumber][0])
                trainy.append(dataset[picturtype][picturnumber][1])
                trainpicturenumbers.append([picturtype,picturnumber])
                break
    for i in range(testsize):
        compare_weight=20
        picturnumber = 0
        picturtype = 0
        while 1:
            picturtype=random.randrange(5)
            picturnumber=random.randrange(datasize)
            compare_weight=random.uniform(0, 1)
            if compare_weight <= dataset[picturtype][picturnumber][2] * 1000:
                testx.append(dataset[picturtype][picturnumber][0])
                testy.append(dataset[picturtype][picturnumber][1])
                testpicturenumbers.append([picturtype,picturnumber])
                break
    trainx = np.array(trainx)
    trainy = np.array(trainy)
    trainy = to_categorical(trainy,5)
    testx = np.array(testx)
    testy = np.array(testy)
    testyvergleich = testy
    testy = to_categorical(testy,5)
    #print(np.shape(trainx))
    #print(np.shape(trainy))
    #print(np.shape(testx))
    #print(np.shape(testy))
    return trainx, trainy, trainpicturenumbers, testx, testy, testpicturenumbers, testyvergleich

def cnn_learn(trainx, trainy, testx, testy, testyvergleich):
    l2Reg = 0.1
    CNN = Sequential()
    CNN.add(layers.Conv2D(32,(3,3),padding='same',activation='relu',kernel_regularizer=l2(l2Reg),input_shape=(32,32,3)))
    CNN.add(layers.MaxPool2D(pool_size=(2, 2),padding='same'))
    CNN.add(layers.Conv2D(64,(3,3),padding='same',activation='relu',kernel_regularizer=l2(l2Reg)))
    CNN.add(layers.MaxPool2D(pool_size=(2, 2),padding='same'))
    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(40,activation='relu',kernel_regularizer=l2(l2Reg)))
    CNN.add(layers.Dense(5,activation='softmax'))
    CNN.summary()
    CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    CNN.fit(trainx,trainy,epochs=7,batch_size=64)
    scores = CNN.evaluate(testx,testy,batch_size=64)
    print("Accuracy: %.2f%%" % (scores[1]*100))


    predy = CNN.predict(testx)
    choise = np.argmax(predy, axis=1)
    predy2 =np.argmax(predy,axis=1)

    y = predy2 - testyvergleich
    fehlermatrix = []
    for i in range(len(testyvergleich)):
        if y[i] != 0:
            fehlermatrix.append(0)
        else:
            fehlermatrix.append(1)
    #print("Fehlermatrix:")
    #print(fehlermatrix)
    return fehlermatrix

def calc_error(fehlermatrix, testnpicturenumbers, dataset):
    i=0
    error = 0
    for element in fehlermatrix:
        if element == 0:
            error = error+(dataset[testnpicturenumbers[i][0]][testnpicturenumbers[i][1]][2])
    print("Fehler:")
    print(error)
    return error

def calc_sum(fehlermatrix, testnpicturenumbers, dataset,beta):
    summe = 0
    for element in fehlermatrix:
        if element == 1:
            summe = summe + math.log(1/beta)
    print("Summe:")
    print(summe)
    return summe

def calc_errorE(fehlermatrix, testnpicturenumbers, dataset):
    i=0
    error = 0
    for element in fehlermatrix:
        if element == 0:
            error = error+(dataset[testnpicturenumbers[i][0]][testnpicturenumbers[i][1]][2])
    print("FehlerE:")
    print(error)
    return error

def calc_wheigths(fehlermatrix, testnpicturenumbers, dataset, beta):
    i=0
    for element in fehlermatrix:
        if element == 0:
           (dataset[testnpicturenumbers[i][0]][testnpicturenumbers[i][1]][2]) = beta*(dataset[testnpicturenumbers[i][0]][testnpicturenumbers[i][1]][2])
 
"""Learn++"""

def learn_plusplus(Trainingsset, K, T): 
    beta_t = []
    H_t = []
    newdata=addy(Trainingsset)
    split_data=split_dataset(K,newdata)
    m = len(split_data[0])
    datasize=len(split_data[0][0])
    testsize = 1500
    trainsize = 3000
    for k in range(K):
        print("K:")
        print(k)
      
        # Zählschleife mit Anzahl der Iterationen
        t = 0
        while(t < T):  
                print("T:")
                print(t)
                anz = len(split_data[k]*K)
                # Gewichte auf das Datenset anwenden
                trainx, trainy, trainpicturenumbers, testx, testy, testpicturenumbers, testyvergleich = getnerate_random_dataset(split_data[k],trainsize,testsize, K, datasize)
               
                # call weak learn / neural network mit zufälligen Trainings- und Testsubset
                
                
                fehlermatrix = cnn_learn(trainx, trainy, testx, testy, testyvergleich)
                epsilon = calc_error(fehlermatrix, testpicturenumbers, split_data[k])
                
                
                if epsilon <= 0.125:
                    
                    beta = epsilon/(1-epsilon)
                    print("beta")
                    print(beta)
                    beta_Ht = math.log(1/beta)
                    beta_t.append(beta_Ht)
                    print("betat0")
                    
                    
                        
                    beta_t1 = np.array(beta_t)
                    summe = np.sum(beta_t1)
                    Ht = summe
                    H_t.append(summe)
                    print("Ht:")
                    print(Ht)
                    
                    Et = calc_errorE(fehlermatrix, testpicturenumbers, split_data[k])
                    
                    if Et <= (testsize/m)*0.5:
                        Bt = Et/(1-Et)
                    
                    
                        calc_wheigths(fehlermatrix, testpicturenumbers, split_data[k], Bt)
                        t = t + 1
        Hfinal = K * np.sum(H_t)
        print("Hfinal:")
        print(Hfinal)

""" MAIN SCRIPT ============================================================="""
if __name__ == '__main__':
    K=2
    T=2
    learn_plusplus(Trainingsset, K, T)

