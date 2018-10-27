import gym
import numpy as np

GAME = 'BipedalWalker-v2'

env = gym.make(GAME)
observation = env.reset()
in_dimen = env.observation_space.shape[0]
out_dimen = env.action_space.shape[0]
obsMin = env.observation_space.low
obsMax = env.observation_space.high
actionMin = env.action_space.low
actionMax = env.action_space.high



max_step = 10000
num_of_episodes = 3000


state_trasition_generation = None

observations_action = []
observations_reward = []


for episode in range(num_of_episodes):
    observation = env.reset()
    for _ in range(max_step):
        # env.render()
        
        # choose a random action
        action = env.action_space.sample()
        # store the observation action pair
        observations_action.append(np.concatenate((observation, action)))
        
        # take a random action
        observation, reward, done, _ = env.step(action) 
        
        # store the observation action pair
        observations_reward.append(np.concatenate((observation, [reward])))

        if done:
            break
    
    print('epside {0}: number of training generated: {1}'.format(
            episode, len(observations_action)))
        
np.save('x_train_observations_action.npy', observations_action)
np.save('y_train_observation_reward.npy', observations_reward)
