from src.agent.world import World
import numpy as np
import gym
from collections import deque

class KStepPlanningAgent:

    def __init__(self, env, k_step = 10):
        self.env = env
        # self.state_size = state_size
        # self.action_size = action_size
        # self.random_seed = random_seed
        self.k_step = k_step
        self.world = World()
        self.extra_observations_actions = deque(maxlen = 10000)
        self.extra_targets = deque(maxlen = 10000)
        self.retrain_step = 100
    
    def retrain(self, epochs = 3):
        self.world.retrain(np.array(self.extra_observations_actions), np.array(self.extra_targets), epochs = epochs)
        return

    def step(self, state, branch = 100):
        first_actions = [self.env.action_space.sample() for _ in range(branch)]
        rewards = np.zeros(branch)
        states = [state for i in range(branch)]
        actions = first_actions
        for i in range(self.k_step):
            state_action_pair = np.concatenate((np.array(states), np.array(actions)), axis=1)
            state_reward_pair = self.world.predict_batch(np.array(state_action_pair))
            #print(state_reward_pair.shape)
            state = [state_reward[:-1] for state_reward in state_reward_pair]
            #print(rewards)
            for j in range(len(state_reward_pair)):
                rewards[j] += state_reward_pair[j][-1]
            actions = [self.env.action_space.sample() for _ in range(branch)]
            
        action_index = np.where(rewards == np.max(rewards))[0][0]
        return first_actions[action_index], np.max(rewards) / self.k_step
        
        

if __name__ == '__main__':


    GAME = 'BipedalWalker-v2'
    env = gym.make(GAME)

    agent = KStepPlanningAgent(env)

    max_step = 300
    num_of_episodes = 20000

    scores_deque = deque(maxlen=100)
    predicted_score_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, num_of_episodes+1):
        state = env.reset()
        score = 0
        predicted_score = 0
        for t in range(max_step):
            # env.render()
            action, predicted_reward = agent.step(state)
            # action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            # get the data
            new_observations_actions = np.concatenate( (np.array(state), np.array(action)) )
            new_target = np.concatenate( (np.array(next_state), np.array([reward])) )
            agent.extra_observations_actions.append(new_observations_actions)
            agent.extra_targets.append(new_target)

            state = next_state
            score += reward
            predicted_score += predicted_reward
            if done:
                break 
        scores_deque.append(score)
        predicted_score_deque.append(predicted_score)
        scores.append(score)
        if score > max_score:
            max_score = score
        # print('Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % 10 == 0:
            print('Episode {}\tAverage Score: {:.2f}\t predicted Score: {:.2f}'.format(i_episode, np.mean(scores_deque), np.mean(predicted_score_deque)))   
            print('Max Score: {:.2f}\n'.format(max_score))
            agent.retrain()

    print('Avg Score: {:.2f}\n'.format(np.mean(scores)))
