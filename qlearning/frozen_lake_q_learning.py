import gym
import numpy as np
from collections import defaultdict

env = gym.make('FrozenLake-v0')


# 0-left 1-down 2-right 3-up
observations = env.observation_space
actions_dict = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}
actions = env.action_space
print('possible observations: ', observations)
print('possible actions: ', actions)

Q = defaultdict(float)
gamma = 0.99  # Discounting factor
alpha = 0.5  # soft update param
n_episode = 10000
MAX_STEP = 500
epsilon = 0.3  # 10% chances to apply a random action

def act(state):
    if np.random.random() < epsilon:
        # action_space.sample() is a convenient function to get a random action
        # that is compatible with this given action space.
        return env.action_space.sample()

    # Pick the action with highest q value.
    qvals = {a: Q[state, a] for a in range(actions.n)}
    max_q = max(qvals.values())
    # In case multiple actions have the same maximum q value.
    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)

def optimal_act(state):
    # Pick the action with highest q value.
    qvals = {a: Q[state, a] for a in range(actions.n)}
    max_q = max(qvals.values())
    # In case multiple actions have the same maximum q value.
    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)


def update_Q(s, r, a, s_next, done):
    max_q_next = max([Q[s_next, a] for a in range(actions.n)])
    # Do not include the next state's value if currently at the terminal state.
    Q[s, a] += alpha * (r + gamma * max_q_next * (1.0 - done) - Q[s, a])

def reward_transform(reward, done):
    # print(reward, done, 1 if reward == 1 else -100 if done else -1)
    # return 100 if reward == 1 else -100 if done else 0
    return reward

reward_list = []
for episode in range(n_episode):
    observation = env.reset()
    total_reward = 0
    for step in range(MAX_STEP):
        # env.render()
        action = act(observation)
        observation_next, reward, done, info = env.step(action)
        # print(observation_next, reward, done, info)
        reward = reward_transform(reward, done)
        total_reward += reward
        update_Q(observation, reward, action, observation_next, done)
        observation = observation_next
        if done:
            reward_list.append(total_reward)
            # print('-'*10+'end'+'-'*10)
            break
            
            
print('Reward List: ')
n_avg = 250
for i in range(n_avg, n_episode, n_avg):
    print('Avg reward for {:d} {:d} episode: {:f}'.format(i-n_avg, i, np.mean(reward_list[i-n_avg:i])))


test_reward = 0
observation = env.reset()
for step in range(MAX_STEP):
    # env.render()
    action = optimal_act(observation)
    # print('action: ', actions_dict[action])
    observation_next, reward, done, info = env.step(action)
    # print('observation: ', observation_next)
    reward = reward_transform(reward, done) 
    test_reward += reward
    update_Q(observation, reward, action, observation_next, done)
    observation = observation_next
    if done:
        print("Episode finished after {} timesteps".format(step+1))
        print('Test reward: ', test_reward)
        observation = env.reset()
        break


