# Q-Learning (off-policy TD control) for estimating pi=pi*
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import time
import os
"""
6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger
    
state space is represented by:
        (taxi_row, taxi_col, passenger_location, destination)
          5 * 5 * 5 * 4 = 500

Rewards:
    per-step : -1,
    delivering the passenger : +20,
    executing "pickup" and "drop-off" actions illegally : -10
    
blue: passenger
magenta: destination
yellow: empty taxi
green: full taxi
"""
env = gym.make('Taxi-v3')
n_states = env.observation_space.n  # 500
n_actions = env.action_space.n      # 6

#Algorithm parameter: step size alpha (0,1], small e > 0
GAMMA = 0.99  # time decay
ALPHA = 0.9  # learning rate
epsilon = 0.7 # exploration start
epsilon_final = 0.1
epsilon_decay = 0.9999
#Initialize Q(s,a) for all s, a arbitrarily except Q(terminal,.)=0
Q = defaultdict(lambda: np.zeros(n_actions))

n_episodes = 1000

scores = []  # agent 가 episode 별로 얻은 score 기록
steps = []  # agent 가 episode 별로 목표를 찾아간 step 수 변화 기록
greedy = [] # epsilon decay history 기록

#Loop for each episode:
for episode in range(n_episodes):
    if episode > n_episodes * 0.995:
        env = gym.make('Taxi-v3', render_mode="human")
    #Initialize S
    s, _ = env.reset()
    step = 0
    score = 0
    #Loop for each step of episode:
    while True:
        step += 1
        # Choose A from S using policy derived from Q (eg. e-greedy)
        # behavior policy : e-greedy
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s])
        
        if epsilon > epsilon_final:
            epsilon *= epsilon_decay
        
        #Take action A, observe R, S'
        s_, r, terminated, truncated, _ = env.step(a)
        score += r
        
        #Q(S,A) <- Q(S,A) + alpha[R + gamma*max_aQ(S',a) - Q(S, A)]
        #자신이 따르는 정책에 상관없이 최적 행동가치함수 q*를 직접 근사
        # target policy : greedy policy
        Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * np.max(Q[s_]) - Q[s][a])
        
        if terminated or truncated:
            break
        
        #S <- S'
        s = s_ 
        
    steps.append(step)
    scores.append(score)
    greedy.append(epsilon)
    
    if episode % 100 == 0:
        print(f"최근 100 episode 평균 score = {np.mean(scores[-100:])}, 평균 step = {np.mean(steps[-100:])}")

plt.bar(np.arange(len(steps)), steps)
plt.title("Steps of Taxi-v3- GAMMA: {}, ALPHA: {}".format(
    GAMMA, ALPHA))
plt.xlabel('episode')
plt.ylabel('steps per episode')
plt.show()

plt.bar(np.arange(len(scores)), scores)
plt.title("Scores of Taxi-v3- GAMMA: {}, ALPHA: {}".format(
                    GAMMA, ALPHA))
plt.xlabel('episode')
plt.ylabel('score per episode')
plt.show()

plt.bar(np.arange(len(greedy)), greedy)
plt.title("Epsilon decay history - epsilon: {}, decay: {}".format(
                    epsilon, epsilon_decay))
plt.xlabel('episode')
plt.ylabel('epsilon per episode')
plt.show()

