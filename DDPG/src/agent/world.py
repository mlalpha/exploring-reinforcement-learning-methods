import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.models import load_model
import numpy as np


class World():
    def __init__(self):
        self.model = load_model('./model.npy')
        '''
        self.observation = np.array([2.74751475e-03, -5.30353864e-06,  4.12582336e-04, -1.59999633e-02,
                                     9.20776874e-02, -5.44475100e-04,  8.60201955e-01,  1.80557463e-03,
                                     1.00000000e+00,  3.24835218e-02, -5.44430921e-04,  8.53750244e-01,
                                     3.85052020e-04,  1.00000000e+00,  4.40814018e-01,  4.45820123e-01,
                                     4.61422771e-01,  4.89550203e-01,  5.34102798e-01,  6.02461040e-01,
                                     7.09148884e-01,  8.85931849e-01,  1.00000000e+00,  1.00000000e+00])
        '''

    # def set_observation(self, observation):
    #     self.observation = observation

    # def step(self, action):
    #     x = np.concatenate((self.observation, action))
    #     x = x.reshape((1, x.shape[0]))
    #     prediction = self.model.predict(x)
    #     self.observation = prediction[:, :-
    #                                   1].reshape(self.observation.shape[0])
    #     return prediction[:, :-1], prediction[:, -1]

    def predict_batch(self, observation_action):
        prediction = self.model.predict(observation_action)
        return prediction

    def predict(self, observation, action):
        x = np.concatenate((observation, action))
        x = x.reshape((1, x.shape[0]))
        prediction = self.model.predict(x)
        return prediction[:, :-1], prediction[:, -1]
    
    def retrain(self, observation_action, target, epochs = 3):
        self.model.fit(observation_action, target, epochs = epochs, verbose = 0)


if __name__ == '__main__':
    x_train = np.load('x_train_observations_action.npy')
    y_train = np.load('y_train_observations_reward.npy')

    world = World()

    i = 0
    for i in range(x_train.shape[0]):
        print(world.predict(x_train[i, -4:]))
        print(world.model.predict(x_train[i:i+1]))
        if i == 10:
            break
        i += 1
