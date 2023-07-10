import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# LEFT - 0, DOWN - 1, RIGHT - 2, UP - 3

# SFFF       (S: 시작점, 안전)
# FHFH       (F: 얼어있는 표면, 안전)
# FFFH       (H: 구멍, 추락)
# HFFG       (G: 목표지점, 프리스비 위치)

# 보상
# 목표에 도달: +1
# 구멍에 도달: 0
# 얼어있는 곳에 도달: 0

# 간단한 결정론적 정책
policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1,
          8: 2, 9: 1, 10: 1, 13: 2, 14: 2}

with_policy = True

SLIPPERY = False  # 결정론적 환경
#SLIPPERY = True  # 확률적 환경

env = gym.make('FrozenLake-v1', desc=None,
               map_name="4x4", is_slippery=SLIPPERY)

n_games = 100
win_pct = []
scores = []

# 게임 진행
for i in range(n_games):
    terminated, truncated = False, False
    obs, info = env.reset()
    score = 0
    while not terminated and not truncated:

        if with_policy:  # 간단한 결정론적 정책
            action = policy[obs]
        else:  # 정책이 없는 경우
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        score += reward

    scores.append(score)

    if i % 10:
        average = np.mean(scores[-10:])
        win_pct.append(average)

env.close()

# 그래프 그리기
plt.plot(win_pct)
plt.xlabel('episode')
plt.ylabel('success ratio')
plt.title('With Policy: average success ratio of last 10 games\n - {}'
          .format('Stochastic Env' if SLIPPERY else 'Deterministic Env'))
plt.show()
