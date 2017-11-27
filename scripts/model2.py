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

def shuffle(A,B):
    n = len(A)
    m = int(0.9*n)
    for i in range(m):
        x = rand.randint(i,m-1)
        A[i],A[x] = A[x],A[i]
        B[i],B[x] = B[x],B[i]
    return
    for i in range(m,n):
        x = rand.randint(i,n-1)
        A[i],A[x] = A[x],A[i]
        B[i],B[x] = B[x],B[i]
    return

##########################################################################
batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 30 # we iterate 25 times over the entire training set
kernel_size = 5 # we will use 3x3 kernels throughout
pool_size = 4 # we will use 2x2 pooling throughout
conv_depth_1 = 10 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 20 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.50 # dropout after pooling with probability 0.25
drop_prob_2 = 0.60 # dropout in the FC layer with probability 0.5
hidden_size = 50 # the FC layer will have 512 neurons
##########################################################################


'''
Reading Training Dataset
'''

FILE_NAMES = []
VALUES = []

filename = "../training.txt"
with open(filename) as f:
    for line in f:
        x = line.split("\t")
        FILE_NAMES.append(x[1][:-1])
        VALUES.append(float(x[0]))

shuffle(FILE_NAMES,VALUES)

Y_train = []

for x in VALUES:
    if float(x) <= 0.2:
        Y_train.append(0)
    elif float(x) <=0.4:   
        Y_train.append(1)
    elif float(x) <= 0.6:
        Y_train.append(2)
    elif float(x) <= 0.8:
        Y_train.append(3)   
    else:
        Y_train.append(4)

Y_train = np.array(Y_train)
y_train = Y_train

del VALUES[:]

folder = "../dataset/training/"
read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
ims = [read(os.path.join(folder, filename)) for filename in FILE_NAMES]
del FILE_NAMES[:]
X_train = np.array(ims, dtype='float32')


##########################################################################


'''
Reading Testing Dataset
'''

FILE_NAMES = []
VALUES = []

filename = "../testing.txt"
with open(filename) as f:
    for line in f:
        x = line.split("\t")
        FILE_NAMES.append(x[1][:-1])
        VALUES.append(float(x[0]))

shuffle(FILE_NAMES,VALUES)

Y_test = []

for x in VALUES:
    if float(x) <= 0.2:
        Y_test.append(0)
    elif float(x) <=0.4:   
        Y_test.append(1)
    elif float(x) <= 0.6:
        Y_test.append(2)
    elif float(x) <= 0.8:
        Y_test.append(3)   
    else:
        Y_test.append(4)

Y_test = np.array(Y_test)
y_test = Y_test

del VALUES[:]

folder = "../dataset/testing/"
read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
ims = [read(os.path.join(folder, filename)) for filename in FILE_NAMES]
del FILE_NAMES[:]
X_test = np.array(ims, dtype='float32')



###########################################################
num_train, height, width, depth = X_train.shape
num_test = X_test.shape[0]
num_classes = np.unique(y_train).shape[0]

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= np.max(X_train) # Normalise data to [0, 1] range
X_test /= np.max(X_test) # Normalise data to [0, 1] range

Y_train = np_utils.to_categorical(y_train, num_classes)
Y_test = np_utils.to_categorical(y_test, num_classes)
##########################################################################



##########################################################################
inp = Input(shape=(height, width, depth))
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_1)
drop_1 = Dropout(drop_prob_1)(pool_1)
conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_3)
drop_2 = Dropout(drop_prob_1)(pool_2)
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)
##########################################################################



##########################################################################
model = Model(inputs=inp, outputs=out)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

filepath="model2.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max',save_weights_only=False)
callbacks_list = [checkpoint]
model.fit(X_train, Y_train,
          batch_size=batch_size, epochs=num_epochs,
          verbose=1, validation_split=0.1,callbacks=callbacks_list)


score=model.evaluate(X_test, Y_test, verbose=1)  # Evaluate the trained model on the test set!
print('Test loss:', score[0])
print('Test accuracy:', score[1])