#export LC_ALL=en_US.UTF-8
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
from random import randint
# number of arms
n = 10
# number of bandits
bandits = 2000
# number of plays
plays = 1000
# mean
mean = 0
# variance
variance = 1
#e-greedy factors
eGreedy = [0.1, 0.01, 0]

#Array containing the Rewards
Qstar = np.zeros((bandits,n))
#Fill the array with random numbers of a Gaussian normal distribution
for i in range(n):
    for j in range(bandits):
        Qstar[j][i] = np.random.normal(mean, variance)

#Graphics setup
figure = plt.figure()
plt1 = figure.add_subplot(211)
plt2 = figure.add_subplot(212)

for e in eGreedy:
    #Average of rewards
    avgRwd = np.zeros((bandits,n))
    #Sum of the rewards
    rwdSum = np.zeros((bandits,n))
    #Number of pulls on each arm
    armPulls = np.zeros((bandits,n))
    #History of all the rewards
    rwdHistory = np.zeros((bandits,plays))
    #Keep track of the picked arms
    bestArmPicked = np.zeros((plays))

    # Let the hunger games begin
    for bandit in range(bandits):
        for play in range(plays):
            # Pick a random arm
            if uniform(0,1) <= e:
                arm = randint(0,n-1)
            # Pick the best arm
            else:
                maxRwd = avgRwd[bandit][0]
                arm = 0
                for i in range(n):
                    if maxRwd < avgRwd[bandit][i]:
                        maxRwd = avgRwd[bandit][i]
                        arm = i
            #Get the reward of the chosen arm plus a Gaussian noise
            reward = np.random.normal(Qstar[bandit][arm],variance)
            
            #Check if the current arm was the best one
            QStartMax = Qstar[bandit][0]
            QStartArm = 0
            for i in range(n):
                if QStartMax < Qstar[bandit][i]:
                    QStartMax = Qstar[bandit][i]
                    QStartArm = i
            #Record in case it was the best arm
            if QStartArm == arm:
                bestArmPicked[play] += 1
        
            #Save the current reward
            rwdHistory[bandit][play] = reward
            #Save reward sum
            rwdSum[bandit][arm] += reward
            #Save number of pulls in this specific arm
            armPulls[bandit][arm] += 1
            #Save the rewards average
            avgRwd[bandit][arm] = float(rwdSum[bandit][arm]) / float(armPulls[bandit][arm])
    
    # Get the mean of the rewards history to plot the average reward graph
    rwdMean = np.mean(rwdHistory, axis=0)
    plt1.plot(rwdMean, label="e = "+str(e))
    
    # Get the mean of the correct chosen arms
    for i in range(plays):
        bestArmPicked[i] /= (bandits)
    plt2.plot(bestArmPicked*100, label="e = "+str(e))

#Plot the Average Rewards Graph
plt1.set_xlabel('Plays')
plt1.set_ylabel('Average Reward')

#Plot the Optimal Actions Graph
plt2.set_xlabel('Plays')
plt2.set_ylabel('% Optimal Action')
plt1.legend(loc="lower right", shadow=True, fancybox=True)
plt2.legend(loc="lower right", shadow=True, fancybox=True)

#Show the Graphs
plt.show()