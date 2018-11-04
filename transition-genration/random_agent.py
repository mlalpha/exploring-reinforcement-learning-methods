import numpy as np
import gym
from collections import deque


if __name__ == '__main__':


    GAME = 'BipedalWalker-v2'
    env = gym.make(GAME)

    max_step = 300
    num_of_episodes = 20000

    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    for i_episode in range(1, num_of_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_step):
            # env.render()
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
            if done:
                break 
        scores_deque.append(score)
        scores.append(score)
        if score > max_score:
            max_score = score
        # print('Episode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
        if i_episode % 10 == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
            print('Max Score: {:.2f}\n'.format(max_score))

    print('Avg Score: {:.2f}\n'.format(np.mean(scores)))