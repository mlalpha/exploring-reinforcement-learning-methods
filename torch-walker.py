import time, math, random, bisect, copy
import gym
import numpy as np
import torch


device = torch.device('cpu')
torch_float = torch.float

# device = torch.device("cuda:0") # Uncomment this to run on GPU
# torch_float = torch.cuda.FloatTensor # Uncomment this to run on GPU


class FullyConnectedNeuralNet():
    
    def __init__(self, in_dim, out_dim, neuron_per_layer, layers = 2):
        self.init_weight(in_dim, out_dim, neuron_per_layer, layers)
        self.learning_rate = 1e-6

        self.fitness = 0.0

    def init_weight(self, in_dim, out_dim, neuron_per_layer, layers = 2):    
        # Randomly initialize weights
        # print(in_dim, out_dim, neuron_per_layer)
        self.w1 = torch.empty(in_dim, neuron_per_layer, device=device, dtype=torch.float).uniform_(-1, 1)
        self.w2 = torch.empty(neuron_per_layer, out_dim, device=device, dtype=torch.float).uniform_(-1, 1)

    def feed_forword(self, x):
        sigmoid = torch.nn.Sigmoid()

        h_1 = x.mm(self.w1)
        h_relu = h_1.clamp(min=0)
        h_2 = h_relu.mm(self.w2)
        y_pred = (sigmoid(h_2) - 0.5)*2.0
        return y_pred



class Population :
    def __init__(self, populationCount, mutationRate, nodeCount):
        self.nodeCount = nodeCount
        self.popCount = populationCount
        self.m_rate = mutationRate
        self.population = [ FullyConnectedNeuralNet(nodeCount[0], nodeCount[2], nodeCount[1]) for i in range(populationCount)]


    def createChild(self, nn1, nn2):
        
        child = FullyConnectedNeuralNet(self.nodeCount[0], self.nodeCount[2], self.nodeCount[1])


        for i in range(len(child.w1)):
            if random.random() > self.m_rate:
                if random.random() < nn1.fitness / (nn1.fitness+nn2.fitness):
                    child.w1[i] = nn1.w1[i]
                else :
                    child.w1[i] = nn2.w1[i]
                        

        for j in range(len(child.w2)):
            if random.random() > self.m_rate:
                if random.random() < nn1.fitness / (nn1.fitness+nn2.fitness):
                    child.w2[j] = nn1.w2[j]
                else :
                    child.w2[j] = nn2.w2[j]

        return child


    def createNewGeneration(self, bestNN):    
        nextGen = []
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        for i in range(self.popCount):
            if random.random() < float(self.popCount-i)/self.popCount:
                nextGen.append(copy.deepcopy(self.population[i]))

        fitnessSum = [0]
        minFit = min([i.fitness for i in nextGen])
        for i in range(len(nextGen)):
            fitnessSum.append(fitnessSum[i]+(nextGen[i].fitness-minFit)**4)
        

        while(len(nextGen) < self.popCount):
            r1 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            r2 = random.uniform(0, fitnessSum[len(fitnessSum)-1] )
            i1 = bisect.bisect_left(fitnessSum, r1)
            i2 = bisect.bisect_left(fitnessSum, r2)
            if 0 <= i1 < len(nextGen) and 0 <= i2 < len(nextGen) :
                nextGen.append( self.createChild(nextGen[i1], nextGen[i2]) )
            else :
                print("Index Error ")
                print("Sum Array =",fitnessSum)
                print("Randoms = ", r1, r2)
                print("Indices = ", i1, i2)
        self.population.clear()
        self.population = nextGen



def replayBestBots(bestNeuralNets, steps, sleep):  
    choice = input("Do you want to watch the replay ?[Y/E/N] : ")
    if choice=='Y' or choice=='y':
        for i in range(len(bestNeuralNets)):
            if (i+1)%steps == 0 :
                observation = env.reset()
                totalReward = 0
                for step in range(MAX_STEPS):
                    env.render()
                    # time.sleep(sleep)
                    action = bestNeuralNets[i].feed_forword(
                        torch.from_numpy(
                            np.array([observation], dtype=np.float32)
                            ).type(torch_float)
                        ).cpu().numpy()[0]
                    observation, reward, done, info = env.step(action)
                    totalReward += reward
                    if done:
                        observation = env.reset()
                        break
                print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % (i+1, bestNeuralNets[i].fitness, totalReward))
    elif choice=='E':
        for i in range(100,len(bestNeuralNets)):
            if (i+1)%steps == 0 :
                observation = env.reset()
                totalReward = 0
                for step in range(MAX_STEPS):
                    env.render()
                    # time.sleep(sleep)
                    action = bestNeuralNets[i].feed_forword(
                        torch.from_numpy(
                            np.array([observation], dtype=np.float32)
                            ).type(torch_float)
                        ).cpu().numpy()[0]
                    observation, reward, done, info = env.step(action)
                    totalReward += reward
                    if done:
                        observation = env.reset()
                        break
                print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % (i+1, bestNeuralNets[i].fitness, totalReward))
        


def recordBestBots(bestNeuralNets):  
    print("\n Recording Best Bots ")
    print("---------------------")
    # env.monitor.start('Artificial Intelligence/'+GAME, force=True)
    observation = env.reset()
    for i in range(len(bestNeuralNets)):
        totalReward = 0
        for step in range(MAX_STEPS):
            # env.render()
            action = bestNeuralNets[i].feed_forword(
                torch.from_numpy(
                    np.array([observation], dtype=np.float32)
                    ).type(torch_float)
                ).cpu().numpy()[0]
            observation, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                observation = env.reset()
                break
        print("Generation %3d | Expected Fitness of %4d | Actual Fitness = %4d" % (i+1, bestNeuralNets[i].fitness, totalReward))
    env.close()




GAME = 'BipedalWalker-v2'
MAX_STEPS = 1000
MAX_GENERATIONS = 150
POPULATION_COUNT = 500
MUTATION_RATE = 0.02
env = gym.make(GAME)
observation = env.reset()
in_dimen = env.observation_space.shape[0]
out_dimen = env.action_space.shape[0]
obsMin = env.observation_space.low
obsMax = env.observation_space.high
actionMin = env.action_space.low
actionMax = env.action_space.high

D_in, D_out, D_H = in_dimen, out_dimen, 80
pop = Population(POPULATION_COUNT, MUTATION_RATE, [D_in, D_H, D_out])
bestNeuralNets = []

print("\nObservation\n--------------------------------")
print("Shape :", in_dimen, " \n High :", obsMax, " \n Low :", obsMin)
print("\nAction\n--------------------------------")
print("Shape :", out_dimen, " | High :", actionMax, " | Low :", actionMin,"\n")

for gen in range(MAX_GENERATIONS):
    genAvgFit = 0.0
    minFit =  1000000
    maxFit = -1000000
    maxNeuralNet = None
    for nn in pop.population:
        observation = env.reset()
        totalReward = 0
        for step in range(MAX_STEPS):
            # env.render()

            action = nn.feed_forword(
                torch.from_numpy(
                    np.array([observation], dtype=np.float32)
                    ).type(torch_float)
                ).cpu().numpy()[0]
            observation, reward, done, info = env.step(action)
            totalReward += reward
            if done:
                break

        nn.fitness = totalReward
        minFit = min(minFit, nn.fitness)
        genAvgFit += nn.fitness
        if nn.fitness > maxFit :
            maxFit = nn.fitness
            maxNeuralNet = copy.deepcopy(nn)

    bestNeuralNets.append(maxNeuralNet)
    genAvgFit/=pop.popCount
    print("Generation : %3d  |  Min : %5.0f  |  Avg : %5.0f  |  Max : %5.0f  " % (gen+1, minFit, genAvgFit, maxFit) )
    pop.createNewGeneration(maxNeuralNet)


print(bestNeuralNets)


recordBestBots(bestNeuralNets)

print('Please press p to play best bots')
while True:
    print('Please press p to play best bots')
    if input() == 'p':
        replayBestBots(bestNeuralNets, max(1, int(math.ceil(MAX_GENERATIONS/10.0))), 0.0625)