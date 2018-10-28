# import the necessary packages

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
os.environ["CUDA_VISIBLE_DEVICES"]="0" 

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Dense
from keras.utils import np_utils
from keras import regularizers
from keras.layers.core import Dropout
import numpy as np
import argparse


# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to output model file")
# args = vars(ap.parse_args())

 
# initialize the data matrix and labels list
x_train = np.load('x_train_observations_action.npy')
y_train = np.load('y_train_observations_reward.npy')

# define the architecture of the network
model = Sequential()
model.add(Dense(2048, input_dim=x_train.shape[1], 
	init="normal",
	activation="relu", activity_regularizer=regularizers.l1(0.01)))
model.add(Dense(1024, init="normal",
	activation="relu"))
# model.add(Dropout(0.3))
model.add(Dense(512, init="normal",
	activation="relu"))
# model.add(Dropout(0.3))
model.add(Dense(256, kernel_initializer="normal",
	activation="relu"))
# model.add(Dropout(0.3))
model.add(Dense(128, kernel_initializer="normal",
	activation="relu"))
# model.add(Dropout(0.3))
model.add(Dense(64, kernel_initializer="normal",
	activation="relu"))
model.add(Dense(25))


# train the model using SGD
print("[INFO] compiling model...")
# sgd = SGD(lr=0.001)
adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999)
model.compile(loss="mse", optimizer=adam,
	metrics=[])
model.fit(x_train, y_train, epochs=800, batch_size=512,
validation_split=0.2, verbose=2, shuffle=True)

'''
# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(x_train, y_train,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))
'''

# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
#model.save(args["model"])
model.save('model_with_4layer_normalinit_l1regularizer.npy')
