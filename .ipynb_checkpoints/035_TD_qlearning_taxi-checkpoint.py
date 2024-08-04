# Q-Learning (off-policy TD control) for estimating pi=pi*
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import time
import os
"""
6개의 이산적인 결정적 행동:
    - 0: 남쪽으로 이동
    - 1: 북쪽으로 이동
    - 2: 동쪽으로 이동
    - 3: 서쪽으로 이동
    - 4: 승객 탑승
    - 5: 승객 하차
    
상태 공간은 다음과 같이 표현됩니다:
        (택시_행, 택시_열, 승객_위치, 목적지)
          5 * 5 * 5 * 4 = 500

보상:
    스텝당: -1,
    승객을 목적지에 배달: +20,
    "pickup"과 "drop-off" 행동을 불법적으로 실행: -10
    
파란색: 승객
자홍색: 목적지
노란색: 빈 택시
녹색: 가득 찬 택시
"""
env = gym.make('Taxi-v3')
n_states = env.observation_space.n  # 500
n_actions = env.action_space.n      # 6

# 알고리즘의 파라미터 설정: 스텝 사이즈 alpha (0, 1], 0 보다 큰 작은 탐색률 e 
GAMMA = 0.99  # time decay
ALPHA = 0.9  # learning rate
epsilon = 0.7 # exploration start
epsilon_final = 0.1
epsilon_decay = 0.9999

# Q(s,a)를 초기화
Q = defaultdict(lambda: np.zeros(n_actions))

n_episodes = 1000

scores = []  # agent 가 episode 별로 얻은 score 기록
steps = []  # agent 가 episode 별로 목표를 찾아간 step 수 변화 기록
greedy = [] # epsilon decay history 기록

#Loop for each episode:
for episode in range(n_episodes):
    if episode > n_episodes * 0.995:
        env = gym.make('Taxi-v3', render_mode="human")
    # 에피소드를 초기화
    s, _ = env.reset()
    step = 0
    score = 0
    # 각 에피소드의 각 스텝에 대한 반복문
    while True:
        step += 1
        # Q에서 유도된 정책(예: e-greedy)을 사용하여 S에서 A를 선택
        # 행동 정책 : e-greedy
        if np.random.rand() < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s])
            
        # epsilon이 epsilon_final보다 크다면 epsilon_decay를 곱하여 감소
        if epsilon > epsilon_final:
            epsilon *= epsilon_decay
        
        # 행동 A를 취하고, R, S'을 관찰
        s_, r, terminated, truncated, _ = env.step(a)
        score += r
        
        # Q(S,A)를 업데이트: Q(S,A) <- Q(S,A) + alpha[R + gamma*max_aQ(S',a) - Q(S, A)]
        # 최적 행동가치함수 q*를 직접 근사
        # 대상 정책 : greedy policy
        Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * np.max(Q[s_]) - Q[s][a])

        # 에피소드가 끝나면 반복문 종료
        if terminated or truncated:
            break
        
        #S <- S'
        s = s_ 
        
    steps.append(step)
    scores.append(score)
    greedy.append(epsilon)
    
    if episode % 100 == 0:
        print(f"최근 100 episode 평균 score = {np.mean(scores[-100:])}, 평균 step = {np.mean(steps[-100:])}")

# 각 에피소드별 단계 수 그래프 그리기
plt.bar(np.arange(len(steps)), steps)
plt.title("Steps of Taxi-v3- GAMMA: {}, ALPHA: {}".format(
    GAMMA, ALPHA))
plt.xlabel('episode')
plt.ylabel('steps per episode')
plt.show()

# 각 에피소드별 점수 그래프 그리기
plt.bar(np.arange(len(scores)), scores)
plt.title("Scores of Taxi-v3- GAMMA: {}, ALPHA: {}".format(
                    GAMMA, ALPHA))
plt.xlabel('episode')
plt.ylabel('score per episode')
plt.show()

# epsilon decay history 그래프 그리기
plt.bar(np.arange(len(greedy)), greedy)
plt.title("Epsilon decay history - epsilon: {}, decay: {}".format(
                    epsilon, epsilon_decay))
plt.xlabel('episode')
plt.ylabel('epsilon per episode')
plt.show()

