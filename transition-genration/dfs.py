import gym
import numpy as np
from world import World

def dfs(env, model, current_state, current_depth, sample_size = 10, thershold = 3):

    if current_depth == thershold:
        return 0, None

    actions = [ env.action_space.sample() for _ in range(sample_size)]
    rewards = []
    for action in actions:
        model.set_state(current_state)
        next_state, reward = model.predict(action)
        next_state, reward = next_state[0], reward[0]
        future_reward, _ = dfs(env, model, next_state, current_depth + 1)
        rewards.append(reward + future_reward)
    
    max_id = np.argmax(rewards)

    return rewards[max_id], actions[max_id]
    

if __name__ == '__main__':
    
    GAME = 'BipedalWalker-v2'

    env = gym.make(GAME)
    observation = env.reset()

    model = World()

    done = False
    total_reward = 0

    while (True):
        
        if done:
            break

        _, action = dfs(env, model, observation, 0)
        # take a dfs action
        env.render()
        observation, reward, done, _ = env.step(action) 

        total_reward += reward
        print(total_reward)

    print("Final reward : {0}".format(total_reward))
        