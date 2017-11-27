'''
Author: Abhishek Chowdhry and Pratik Chhajer
Purpose: To predict aesthetic quality of image on a scale of 1 to 5.
How to use: There is a folder named test_images in parent directory of scripts
            Put all your image to test in that folder
            Run this code ie.. python3 main.py
Sample Output: 
farm1_262_20009074919_cdd9c88d5f_b_0_f.jpg 	 3
farm1_551_19542178634_a28b694bb3_b_0_f.jpg 	 2
farm1_546_19894951500_a19ce7092d_b_0_f.jpg 	 2
farm4_3825_20263660105_2e24625702_b_0_f.jpg 	 5
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

'''
This function reads accuracies of three generated models
This function read single value denoting accuracy from file named:-
accuracy1.txt, accuracy2.txt and accuracy3.txt
All the above files are present in folder named accuracy which is present in parent directory of script folder
'''
def get_accuracies():
	filename = "../accuracy/accuracy1.txt"
	with open(filename) as f:
	    accuracy1 = float(f.readline())
	filename = "../accuracy/accuracy1.txt"
	with open(filename) as f:
	    accuracy2 = float(f.readline())
	filename = "../accuracy/accuracy1.txt"
	with open(filename) as f:
	    accuracy3 = float(f.readline())

	return accuracy1, accuracy2, accuracy3

'''
This function read all the images from test_images
It return a list containing name of all the test_images and their matrix representation in numpy array
'''
def read_input():
	folder = "../test_images/"
	FILE_NAMES = os.listdir(folder)
	NEW_FILE_NAMES = []
	for i in range(len(FILE_NAMES)):
		im = Image.open(folder + FILE_NAMES[i])
		im = im.resize((256, 256))
		im.save(folder + "256X256_"+FILE_NAMES[i])
		NEW_FILE_NAMES.append("256X256_"+FILE_NAMES[i])
	read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
	ims = [read(os.path.join(folder, filename)) for filename in NEW_FILE_NAMES]
	for i in NEW_FILE_NAMES:
		os.remove(folder+i)
	X_test = np.array(ims, dtype='float32')
	X_test /= np.max(X_test)
	return FILE_NAMES,X_test


# Main Code begins here
if __name__ == "__main__":

	# Read Accuracies of all the models
	accuracy1, accuracy2, accuracy3 = get_accuracies()

	# Read the input data
	FILE_NAMES,X_test = read_input()

	# Reading Model 1
	model1 = load_model('../model/model1.h5')
	prediction=model1.predict( X_test, batch_size=32, verbose=0)
	y1_classes = prediction.argmax(axis=-1)
	
	# Reading Model 2
	model2 = load_model('../model/model2.h5')
	prediction=model2.predict( X_test, batch_size=32, verbose=0)
	y2_classes = prediction.argmax(axis=-1)
	
	# Reading Model 3
	model3 = load_model('../model/model3.h5')
	prediction=model3.predict( X_test, batch_size=32, verbose=0)
	y3_classes = prediction.argmax(axis=-1)
	

	# Prediction using ensembling
	for i in range(len(y1_classes)):
		y1 = y1_classes[i]
		y2 = y2_classes[i]
		y3 = y3_classes[i]
		Count = [0,0,0,0,0]
		Count[y1] += 1
		Count[y2] += 1
		Count[y3] += 1
		found = False
		for j in range(len(Count)):
			if Count[j] >= 2:
				found = True
				print(FILE_NAMES[i],"\t",j+1)
				break

		if not found:
			if max(accuracy1,accuracy2,accuracy3) == accuracy1:
				print(FILE_NAMES[i],"\t",y1+1)
			elif max(accuracy1,accuracy2,accuracy3) == accuracy2:
				print(FILE_NAMES[i],"\t",y2+1)
			else:
				print(FILE_NAMES[i],"\t",y3+1)