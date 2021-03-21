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
#Testset = np.load('TestsetCifar10.npy')
#Trainingsset = np.load('TrainingssetCifar10.npy')
yTest = np.load('yTestset.npy')
yTraining = np.load('yTrainingsset.npy')
cl0 = np.load('cl0.npy')
cl1 = np.load('cl1.npy')
cl2 = np.load('cl2.npy')
cl3 = np.load('cl3.npy')
cl4 = np.load('cl4.npy')
cl5 = np.load('cl5.npy')
cl6 = np.load('cl6.npy')
cl7 = np.load('cl7.npy')
cl8 = np.load('cl8.npy')
cl9 = np.load('cl9.npy')

Testset = []
Testset.append(cl5)
Testset.append(cl6)
Testset.append(cl7)
Testset.append(cl8)
Testset.append(cl9)
Trainingsset = []
Trainingsset.append(cl0)
Trainingsset.append(cl1)
Trainingsset.append(cl2)
Trainingsset.append(cl3)
Trainingsset.append(cl4)

Testset = np.array(Testset)
Trainingsset = np.array(Trainingsset)
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
    trainselection=[]

    testx=[]
    testy=[]
    testpicturenumbers=[]
    testselection=[]

    trainnumber=0
    testnnumber=0
    weightsum=0

    for picturetype in dataset:
        for picture in picturetype:
            weightsum += picture[2]
    for i in range(trainsize):
        trainselection.append(random.uniform(0, weightsum))
    trainselection.append(weightsum+1)
    for i in range(testsize):
        testselection.append(random.uniform(0, weightsum))
    testselection.append(weightsum+1)
    trainselection=np.asarray(trainselection)
    trainselection=np.sort(trainselection)
    testselection=np.asarray(testselection)
    testselection=np.sort(testselection)
    weightsum=0
    for picturetype in range(len(dataset)):
        for picturnumber in range(len(dataset[picturetype])):
            weightsum += dataset[picturetype][picturnumber][2]
            if weightsum>trainselection[trainnumber]:
                trainpicturenumbers.append([picturetype,picturnumber])
                trainx.append(dataset[picturetype][picturnumber][0])
                trainy.append(dataset[picturetype][picturnumber][1])
                trainnumber += 1

            elif weightsum>testselection[testnnumber]:
                testpicturenumbers.append([picturetype,picturnumber])
                testx.append(dataset[picturetype][picturnumber][0])
                testy.append(dataset[picturetype][picturnumber][1])
                testnnumber +=1

    trainpicturenumbers = np.array(trainpicturenumbers)
    testpicturenumbers = np.array(testpicturenumbers)
    trainx = np.array(trainx)
    trainy = np.array(trainy)
    trainy = to_categorical(trainy,5)
    testx = np.array(testx)
    testy = np.array(testy)
    testyvergleich = testy
    testy = to_categorical(testy,5)
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
    
    print("predy2")
    print(predy2)

    y = predy2 - testyvergleich
    
    fehlermatrix = []
    
    for i in range(len(testyvergleich)):
        if y[i] != 0:
            fehlermatrix.append(0)
        else:
            fehlermatrix.append(1)
            
    klassen = [0,0,0,0,0]
    for j in range(len(testyvergleich)):
        if predy2[j] == 0 and fehlermatrix[j] == 1:
            klassen[0] = klassen[0] + 1
        if predy2[j] == 1 and fehlermatrix[j] == 1:
            klassen[1] = klassen[1] + 1
        if predy2[j] == 2 and fehlermatrix[j] == 1:
            klassen[2] = klassen[2] + 1    
        if predy2[j] == 3 and fehlermatrix[j] == 1:
            klassen[3] = klassen[3] + 1
        if predy2[j] == 4 and fehlermatrix[j] == 1:
            klassen[4] = klassen[4] + 1
                    
    klassen = np.array(klassen)
    besteKlasse = klassen[np.argmax(klassen)]
    klasse = np.argmax(klassen)
   
            
    return fehlermatrix, besteKlasse, klasse

def calc_error(fehlermatrix, testnpicturenumbers, dataset):
    i=0
    error = 0
    for element in fehlermatrix:
        if element == 0:
            error = error+(dataset[testnpicturenumbers[i][0]][testnpicturenumbers[i][1]][2])
    print("Fehler:")
    print(error)
    return error

def calc_besteBilder(anz_bilder_klasse):
    #print(anz_bilder_klasse)
    
    summen = [0,0,0,0,0]
    for i in range(len(anz_bilder_klasse)):
        if anz_bilder_klasse[i][1] == 0:
            summen[0] = summen[0] + anz_bilder_klasse[i][0]
        if anz_bilder_klasse[i][1] == 1:
            summen[1] = summen[1] + anz_bilder_klasse[i][0]
        if anz_bilder_klasse[i][1] == 2:
            summen[2] = summen[2] + anz_bilder_klasse[i][0]
        if anz_bilder_klasse[i][1] == 3:
            summen[3] = summen[3] + anz_bilder_klasse[i][0]
        if anz_bilder_klasse[i][1] == 4:
            summen[4] = summen[4] + anz_bilder_klasse[i][0]
    summen = np.array(summen)
    maxBilder = summen[np.argmax(summen)]
    return maxBilder
            

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
    anz_bilder_klasse = []
    B_t =  []
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
        beta_t = []
        H_t = []
        while(t < T):  
                print("T:")
                print(t)
                anz = len(split_data[k]*K)
                # Gewichte auf das Datenset anwenden
                trainx, trainy, trainpicturenumbers, testx, testy, testpicturenumbers, testyvergleich = getnerate_random_dataset(split_data[k],trainsize,testsize, K, datasize)
               
                # call weak learn / neural network mit zufälligen Trainings- und Testsubset
                
                
                fehlermatrix, bilderBesteKlasse, besteKlasse = cnn_learn(trainx, trainy, testx, testy, testyvergleich)
                epsilon = calc_error(fehlermatrix, testpicturenumbers, split_data[k])
                bilder_klasse = [bilderBesteKlasse, besteKlasse]
                anz_bilder_klasse.append(bilder_klasse)
                
                
                if epsilon <= 0.125:
                    
                    beta = epsilon/(1-epsilon)
                    print("beta")
                    print(beta)
                    beta_Ht = math.log(1/beta)
                    beta_t.append(beta_Ht)
                    print("betat0")
                    
                    
                        
                    beta_t1 = np.array(beta_t)
                    summe = np.sum(beta_t1)
                    Ht = summe * bilderBesteKlasse
                    H_t.append(summe)
                    print("Ht:")
                    print(Ht)
                    
                    Et = calc_errorE(fehlermatrix, testpicturenumbers, split_data[k])
                    
                    if Et <= 0.125:
                        Bt = Et/(1-Et)
                        B_t.append(Bt)
                    
                    
                        calc_wheigths(fehlermatrix, testpicturenumbers, split_data[k], Bt)
                        bilder_klasse = [bilderBesteKlasse, besteKlasse]
                        anz_bilder_klasse.append(bilder_klasse)
                        t = t + 1
       
    B_t = np.array(B_t)
    summe_B = np.sum(B_t)
    besteBilder = calc_besteBilder(anz_bilder_klasse)
    print(besteBilder)
    Hfinal = summe_B * besteBilder
    print("Hfinal:")
    print(Hfinal)

""" MAIN SCRIPT ============================================================="""
if __name__ == '__main__':
    K=5
    T=3
    learn_plusplus(Trainingsset, K, T)

