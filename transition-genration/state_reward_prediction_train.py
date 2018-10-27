# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
import numpy as np
import argparse


# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True,
# 	help="path to output model file")
# args = vars(ap.parse_args())

 
# initialize the data matrix and labels list
x_train = np.load('x_train_observations_action.npy')
y_train = np.load('y_train_observation_reward.npy')

# define the architecture of the network
model = Sequential()
model.add(Dense(768, input_dim=x_train.shape[1], init="uniform",
	activation="relu"))
model.add(Dense(384, activation="relu", kernel_initializer="uniform"))
model.add(Dense(25))
model.add(Activation("relu"))


# train the model using SGD
print("[INFO] compiling model...")
sgd = SGD(lr=0.0001)
model.compile(loss="binary_crossentropy", optimizer=sgd,
	metrics=["accuracy"])
model.fit(x_train, y_train, epochs=50, batch_size=128,
    validation_split=0.2, verbose=1)

# show the accuracy on the testing set
print("[INFO] evaluating on testing set...")
(loss, accuracy) = model.evaluate(x_train, y_train,
	batch_size=128, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))
 
# dump the network architecture and weights to file
print("[INFO] dumping architecture and weights to file...")
model.save(args["model"])
model.save('model.npy')