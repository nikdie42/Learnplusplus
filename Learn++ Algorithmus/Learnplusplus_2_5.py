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
from tensorflow.keras.datasets import cifar10
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()



def load_dataset():
    yTest = np.load('yTestset.npy')             #Laden der Daten in das Programm
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


    dataset1 = []               #Bildklassen 1 bis 5
    dataset1.append(cl0)        #Bildklassen der liste hinzufügen
    dataset1.append(cl1)
    dataset1.append(cl2)
    dataset1.append(cl3)
    dataset1.append(cl4)

    dataset2 = []                #Bildklassen 6 bis 10
    dataset2.append(cl5)
    dataset2.append(cl6)
    dataset2.append(cl7)
    dataset2.append(cl8)
    dataset2.append(cl9)
    print(len(cl9))

    return dataset1, dataset2


def addy(dataset):
    #Diese Funktion fügt jedem Datensatz ein Gewicht sowie die Nummer des Bildtyps zu.

    dataset_with_weights=[]
    weight=1/6000

    i=0
    for bildtype in dataset:
        dataset_with_weights.append([])
        j=0
        for picture in bildtype:
            dataset_with_weights[i].append([dataset[i][j],i,weight])
            j=j+1
        i=i+1
    return dataset_with_weights

def split_dataset(K,dataset):
    #splits up data into K groups of equal size

    datasize=math.floor(6000/K)     #!! 6000=Größe des Datensatzes
    newdataset=[]
    for i in range(K):
        newdataset.append([])
        for j in range(5):
            newdataset[i].append(dataset[j][i*datasize:(i+1)*datasize])
    return newdataset

def calc_errorE(fehlermatrix, testnpicturenumbers, dataset):
    i=0
    error = 0
    for element in fehlermatrix:
        if element == 0:
            error = error+(dataset[testnpicturenumbers[i][0]][testnpicturenumbers[i][1]][2])
        i += 1
    print("FehlerE:")
    print(error)
    return error


def getnerate_random_dataset(dataset,trainsize,testsize, K, datasize):
    #Diese Funktion Wählt aus dem gegebenen Datensatz zufälige Daten aus. 
    #Diese werden in Trainings- und Testdaten unterteilt, welche dann für das CNN verwendet werden können

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
    for i in range(trainsize):
        trainselection.append(random.uniform(0, 1))     #Generiere Zufallszahlen, welche später entscheiden welche Daten gewählt werden
    trainselection.append(2)                            #Füge eine zusätzliche Zufallszahl hinzu, die einen potzenziellen overflow verhindert
    for i in range(testsize):                           #Die obere for Schleife war für die Trainingsdaten, diese ist für die Testdaten.
        testselection.append(random.uniform(0, 1))      #Das Prinzip ist das gleiche
    testselection.append(2)
    trainselection=np.asarray(trainselection)
    trainselection=np.sort(trainselection)              #Sortiere die Generierten Zahlen der größe nach
    testselection=np.asarray(testselection)
    testselection=np.sort(testselection)
    weightsum=0
    for picturetype in range(len(dataset)):             #Durchlaufe alle Daten des Datensatzes einmal
        for picturnumber in range(len(dataset[picturetype])):
            weightsum += dataset[picturetype][picturnumber][2]          #addiere alle Gewichte von durchlaufenden Daten auf.
            if weightsum>trainselection[trainnumber]:                   #Wenn: Summe aller durchlaufenen daten > kleinste noch nicht verwendete zuvor generierte Zahl
                trainpicturenumbers.append([picturetype,picturnumber])  #Dann: Speichere aktuellen Datensatz und springe zur nächst größenen zuvor generierten Zahl
                trainx.append(dataset[picturetype][picturnumber][0])
                trainy.append(dataset[picturetype][picturnumber][1])
                trainnumber += 1

            elif weightsum>testselection[testnnumber]:                  #Das gleiche Verfahren nur für die Testdaten.
                testpicturenumbers.append([picturetype,picturnumber])
                testx.append(dataset[picturetype][picturnumber][0])
                testy.append(dataset[picturetype][picturnumber][1])
                testnnumber +=1

    #Daten werden noch leicht formatiert, damit später einfacher damit gearbeitet werden kann.
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

def cnn_build_modul():
    l2Reg = 0.05
    CNN = Sequential()
    CNN.add(layers.Conv2D(32,(3,3),padding='same',activation='relu',kernel_regularizer=l2(l2Reg),input_shape=(32,32,3)))
    CNN.add(layers.MaxPool2D(pool_size=(2, 2),padding='same'))
    CNN.add(layers.Conv2D(64,(3,3),padding='same',activation='relu',kernel_regularizer=l2(l2Reg)))
    CNN.add(layers.MaxPool2D(pool_size=(2, 2),padding='same'))
    CNN.add(layers.Flatten())
    CNN.add(layers.Dense(40,activation='relu',kernel_regularizer=l2(l2Reg)))
    CNN.add(layers.Dense(5,activation='softmax'))
    CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    return CNN

def cnn_learn(CNN, trainx, trainy, testx, testy, testyvergleich):
    global treecount
    #In dieser Funkion wird der CNN-Lerner angewendet.
    #Initialisierung des CNN mit geeigneten Parametern.
    
    CNN.fit(trainx,trainy,epochs=10,batch_size=64)       #Trainieren des CNN
    scores = CNN.evaluate(testx,testy,batch_size=64)    #Testen des CNN
    print("Accuracy: %.2f%%" % (scores[1]*100))

    predy = CNN.predict(testx)
    choise = np.argmax(predy, axis=1)
    predy2 =np.argmax(predy,axis=1)
    
    print("predy2")
    print(predy2)

    y = predy2 - testyvergleich
    
    fehlermatrix = []           #Die Fehlermatrix beschreibt welche Datensätze richtig und welche falsch erkannt wurden.
    
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
    anzahl_der_richtigen_bilder_der_Besten_Klasse = klassen[np.argmax(klassen)]
    beste_klasse = np.argmax(klassen)
            
    return fehlermatrix, anzahl_der_richtigen_bilder_der_Besten_Klasse, beste_klasse, klassen, CNN




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



def calc_error(fehlermatrix, testnpicturenumbers, dataset):
    #Der Fehler in Abhängigkeit von den Gewichten wird berechnet.
    i=0
    error = 0
    for element in fehlermatrix:
        if element == 0:
            error = error+(dataset[testnpicturenumbers[i][0]][testnpicturenumbers[i][1]][2])
        i += 1
    print("Fehler:")
    print(error)
    return error

def calc_wheigths(fehlermatrix, testnpicturenumbers, dataset, beta):
    i=0
    for element in fehlermatrix:
        if element == 1:
            dataset[testnpicturenumbers[i][0]][testnpicturenumbers[i][1]][2] = beta*(dataset[testnpicturenumbers[i][0]][testnpicturenumbers[i][1]][2])
        i += 1
    return dataset

def norm_weights(dataset):
    weightsum=0
    for picturetype in dataset:
        for picture in picturetype:
            weightsum += picture[2]
    multiplyer=1/weightsum
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            dataset[i][j][2] = dataset[i][j][2]*multiplyer
    return dataset
 
def testlearnplusplus(allCNN1, allCNN2, xTest, yTest):
    predictions1 = []
    predictions2 = []
    for element in allCNN1:
        predictions1.append(element.predict(xTest))
    for element in allCNN2:
        predictions2.append(element.predict(xTest))
    allresults = []
    for i in range(len(xTest)):
        
        result = [0,0,0,0,0,0,0,0,0,0]
        for prediction in predictions1:
            for j in range(5):
                result[j] = result[j]+prediction[i][j]
            
        for prediction in predictions2:
            for j in range(5):
                result[j+5] = result[j+5]+prediction[i][j]

        allresults.append(result)
    
    yresults = []
    allresults = np.array(allresults)
    i=0
    for element in allresults:
        yresults.append(np.argmax(element))
        i=i+1
    false = 0
    true = 0
    for i in range(len(yresults)):
        if yresults[i]==yTest[i]:
            true = true +1
        else:
            false = false +1

    i=0
    fehlermatrixx=[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],
    for element in yTest:
        fehlermatrixx[element[0]][yresults[i]] += 1
        i=i+1
    print(fehlermatrixx)
    df = pd.DataFrame(fehlermatrixx)
    df.to_excel(excel_writer = "Matrix.xlsx")
    print(false)
    print(true)



"""Learn++"""

def learn_plusplus(Trainingsset, K, T):
    global treecount
    treecount = 0
    anz_bilder_klasse = []
    B_t =  []
    CNN = cnn_build_modul()
    newdata=addy(Trainingsset)
    split_data=split_dataset(K,newdata)
    m = len(split_data[0])
    datasize=len(split_data[0][0])
    testsize = 1000
    trainsize = 1000
    datasetsize = len(split_data[0][0])*5
    alleklassen = []
    alleklassenfailed = []
    allCNN=[]
    print(0.5/(datasetsize/testsize))
    print(np.shape(np.array(split_data)))
    for k in range(K):
        #testsumm(split_data[k])
        split_data[k] = norm_weights(split_data[k])
        print("K: "+str(k))
      
        # Zählschleife mit Anzahl der Iterationen
        t = 0
        beta_t = []
        while(t < T):  
            print("T: "+str(t))
            anz = len(split_data[k]*K)
            # Gewichte auf das Datenset anwenden
            trainx, trainy, trainpicturenumbers, testx, testy, testpicturenumbers, testyvergleich = getnerate_random_dataset(split_data[k],trainsize,testsize, K, datasize)
            # call weak learn / neural network mit zufälligen Trainings- und Testsubset
            
            CNN = cnn_build_modul()
            fehlermatrix, anzahl_der_richtigen_bilder_der_Besten_Klasse, besteKlasse, klassen, CNN = cnn_learn(CNN, trainx, trainy, testx, testy, testyvergleich)
            epsilon = calc_error(fehlermatrix, testpicturenumbers, split_data[k])           #Epsilon = Summe der Gewichte der falsch zugeordneten Bilder

            
            if epsilon <= 0.6/(datasetsize/testsize):   #0.5 durch den Anteil der Daten, mit denen getestet wird
                alleklassen.append(klassen)
                beta = epsilon/(1-epsilon)          #Normalisierte Fehler         
                beta_t.append(math.log(1/beta))     #log von inv von beta
                summe = np.sum(np.array(beta_t))    #Summe aller vorherigen wird berechnet
                Ht = summe * anzahl_der_richtigen_bilder_der_Besten_Klasse  #HT = Zusammengesetzte Hypothese der aktuellen Iteration  
                Et = calc_errorE(fehlermatrix, testpicturenumbers, split_data[k])   # E_t = Summe der Gewichte der falsch zugeordneten Bilder
                
                if Et <= 0.6/(datasetsize/testsize):    
                    B_t.append(Et/(1-Et))           #Normalisierte Fehler von E_t
                    split_data[k] = calc_wheigths(fehlermatrix, testpicturenumbers, split_data[k], Et/(1-Et))    #Aktuallisierung der Gewichte
                    split_data[k] = norm_weights(split_data[k])                                                         #Gewichte werden normalisiert
                    anz_bilder_klasse.append([anzahl_der_richtigen_bilder_der_Besten_Klasse, besteKlasse])  #Speichern der Ergebnisse
                    t = t + 1                      #Durchlauf war erfolgreich, t wir erhöt
                    treecount = treecount+1
                    allCNN.append(CNN)
            else:
                alleklassenfailed.append(klassen)
       
    B_t = np.array(B_t)
    summe_B = np.sum(B_t)                               #Summe von allen B_t
    besteBilder = calc_besteBilder(anz_bilder_klasse)   #Welches Bild wurde insgesammt am besten zugeordet
    Hfinal = summe_B * besteBilder                      #Berechnung der finalen Hypothese 
    print("Hfinal: " +str(Hfinal))

    
    return alleklassen, alleklassenfailed, allCNN

def calcvalue(values):
    x=[0,0,0,0,0]
    for element in values:
        i=0
        for thing in element:
            x[i]=x[i]+element
            i=i+1
    return x



""" MAIN SCRIPT ============================================================="""
if __name__ == '__main__':
    dataset1, dataset2=load_dataset()
    K=5
    T=5
    alleklassen2, alleklassenfailed2, allCNN2 = learn_plusplus(dataset2, K, T)
    alleklassen1, alleklassenfailed1, allCNN1 = learn_plusplus(dataset1, K, T)

    print("alleklassen1")
    print(len(alleklassen1))
    print(calcvalue(alleklassen1))

    print("alleklassenfailed1")
    print(len(alleklassenfailed1))
    print(calcvalue(alleklassenfailed1))

    print("alleklassen2")
    print(len(alleklassen2))
    print(calcvalue(alleklassen2))

    print("alleklassenfailed2")
    print(len(alleklassenfailed2))
    print(calcvalue(alleklassenfailed2))


    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    testlearnplusplus(allCNN1, allCNN2, xTest, yTest)

