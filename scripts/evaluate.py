'''
AUTHORS: Abhishek Chowdhry and Pratik Chhajer

PURPOSE: To find testing accuracy on testing dataset for our three generated models

HOW TO USE: There is a folder named model in parent directory of scripts
            All the three generated models are present in folder named model in parent directory of scripts
            Run this code ie.. python3 evaluate.py

SAMPLE OUTPUT: 
        Reading Model 1
        Evaluating Mdel 1
        800/800 [==============================] - 6s 8ms/step
        Test loss for Model 1: 1.76433014154
        Test accuracy for Model 1: 0.35
        ############################################
        Reading Model 2
        Evaluating Model 2
        800/800 [==============================] - 8s 10ms/step
        Test loss for Model 2: 1.43449003458
        Test accuracy for Model 2: 0.36375
        ############################################
        Reading Model 3
        Evaluating Mdel 3
        800/800 [==============================] - 4s 5ms/step
        Test loss for Model 3: 1.43597866058
        Test accuracy for Model 3: 0.41375
        ############################################ 
'''

from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
import random as rand
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import keras

def test_input():
    FILE_NAMES = []
    VALUES = []

    filename = "../testing.txt"
    with open(filename) as f:
        for line in f:
            x = line.split("\t")
            FILE_NAMES.append(x[1][:-1])
            VALUES.append(float(x[0]))

    Y_test = []

    for i in range(len(VALUES)):
        x = VALUES[i]
        if float(x) <= 0.2:
            Y_test.append(0)
            VALUES[i] = 0
        elif float(x) <=0.4:   
            Y_test.append(1)
            VALUES[i] = 1
        elif float(x) <= 0.6:
            Y_test.append(2)
            VALUES[i] = 2
        elif float(x) <= 0.8:
            Y_test.append(3)
            VALUES[i] = 3   
        else:
            Y_test.append(4)
            VALUES[i] = 4

    Y_test = np.array(Y_test)
    Y_test = Y_test
    num_classes = np.unique(Y_test).shape[0]
    Y_test = np_utils.to_categorical(Y_test, num_classes)
    VALUES = [x+1 for x in VALUES]

    folder = "../dataset/testing/"
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    ims = [read(os.path.join(folder, filename)) for filename in FILE_NAMES]
    del FILE_NAMES[:]
    X_test = np.array(ims, dtype='float32')
    X_test /= np.max(X_test)
    return X_test,Y_test,VALUES

if __name__ == "__main__":

    X_test, Y_test, ORIGINAL = test_input()

    print("Reading Model 1")
    model1 = load_model('../model/model1.h5')
    print("Evaluating Mdel 1")
    score=model1.evaluate(X_test, Y_test, verbose=1)
    print('Test loss for Model 1:', score[0])
    print('Test accuracy for Model 1:', score[1])
    file1 = open('../accuracy/accuracy1.txt','w')
    file1.write(str(score[1]))
    accuracy1 = score[1]
    file1.close()
    prediction=model1.predict( X_test, batch_size=32, verbose=0)
    y1_classes = prediction.argmax(axis=-1)
    y1_classes[:] += 1
    #print("Prediction for Model 1: ", y1_classes)
    print("############################################")

    print("Reading Model 2")
    model2 = load_model('../model/model2.h5')
    print("Evaluating Model 2")
    score=model2.evaluate(X_test, Y_test, verbose=1)
    print('Test loss for Model 2:', score[0])
    print('Test accuracy for Model 2:', score[1])
    file1 = open('../accuracy/accuracy2.txt','w')
    file1.write(str(score[1]))
    accuracy2 = score[1]
    file1.close()
    prediction=model2.predict( X_test, batch_size=32, verbose=0)
    y2_classes = prediction.argmax(axis=-1)
    y2_classes[:] += 1
    #print("Prediction for Model 2: ", y2_classes)
    print("############################################")

    print("Reading Model 3")
    model3 = load_model('../model/model3.h5')
    print("Evaluating Mdel 3")
    score=model3.evaluate(X_test, Y_test, verbose=1)
    print('Test loss for Model 3:', score[0])
    print('Test accuracy for Model 3:', score[1])
    file1 = open('../accuracy/accuracy3.txt','w')
    file1.write(str(score[1]))
    accuracy3 = score[1]
    file1.close()
    prediction=model3.predict( X_test, batch_size=32, verbose=0)
    y3_classes = prediction.argmax(axis=-1)
    y3_classes[:] += 1
    #print("Prediction for Model 3: ", y3_classes)
    print("############################################")