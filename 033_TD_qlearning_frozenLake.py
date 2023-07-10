# Q-Learning (off-policy TD control) 알고리즘으로 pi=pi* 구하기
# LEFT - 0, DOWN - 1, RIGHT - 2, UP - 3

# SFFF       (S: 시작 지점, 안전)
# FHFH       (F: 얼어있는 표면, 안전)
# FFFH       (H: 구멍, 추락하면 끝)
# HFFG       (G: 목표 지점, 프리스비가 위치한 곳)

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 알고리즘의 파라미터 설정: 스텝 사이즈 alpha, 작은 e>0
GAMMA = 0.99
ALPHA = 0.9
epsilon = 0.3
n_episodes = 10_000

is_slippery = False

# 'FrozenLake-v1' 게임 환경 생성
env = gym.make('FrozenLake-v1', is_slippery=is_slippery, render_mode=None)

num_actions = env.action_space.n

win_pct = []
scores = []

# Q(s,a)를 초기화(Q(terminal, .)=0 제외)
Q = defaultdict(lambda: np.zeros(num_actions))

# 각 에피소드를 위한 반복문
for episode in range(n_episodes):
    #전체 episode의 99.9%가 지나면 렌더링
    if episode > n_episodes * 0.999:
        env = gym.make('FrozenLake-v1', is_slippery=is_slippery,
                       render_mode="human")
    # 에피소드를 초기화
    s, _ = env.reset()
    # Loop for each step of episode:
    score = 0
    while True:
        # Q에서 유도된 정책(예: e-greedy)을 사용하여 S에서 A를 선택
        # 행동 정책 : e-greedy
        if np.random.rand() < epsilon:
            a = env.action_space.sample()  # 무작위 행동 선택
        else:
            a = np.argmax(Q[s])  # 가장 큰 Q값을 가지는 행동 선택

        # 행동 A를 취하고, R, S'을 관찰
        s_, r, terminated, truncated, _ = env.step(a)
        score += r

        # Q(S,A)를 업데이트: Q(S,A) <- Q(S,A) + alpha[R + gamma*max_aQ(S',a) - Q(S, A)]
        #자신이 따르는 정책에 상관없이 최적 행동가치함수 q*를 직접 근사
        # 대상 정책 : greedy policy
        Q[s][a] = Q[s][a] + ALPHA * (r + GAMMA * np.max(Q[s_]) - Q[s][a])

        # 에피소드가 끝나면 반복문 종료
        if terminated or truncated:
            break

        # 상태 업데이트
        s = s_

    scores.append(score)

    if episode % 1000 and episode > 0.8 * n_episodes:
        average = np.mean(scores[-10:])
        win_pct.append(average)

print("Stochastic" if is_slippery else "Deterministic")
print("GAMMA = {}, epsilon = {}, ALPHA = {}".format(GAMMA, epsilon, ALPHA))

# 학습 결과 시각화
plt.plot(win_pct)
plt.xlabel('episode')
plt.ylabel('success ratio')
plt.title('average success ratio of last 10 games\n - {}'
          .format('Stochastic Env' if is_slippery else 'Deterministic Env'))
plt.show()

# 최적 정책 출력
WIDTH = 4
HEIGHT = 4
GOAL = (3, 3)
actions = ['L', 'D', 'R', 'U']

optimal_policy = []
for i in range(HEIGHT):
    optimal_policy.append([])
    for j in range(WIDTH):
        optimal_action = Q[i*WIDTH+j].argmax()
        if (i, j) == GOAL:
            optimal_policy[i].append("G")
        else:
            optimal_policy[i].append(actions[optimal_action])

for row in optimal_policy:
    print(row)
