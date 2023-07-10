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



# 게임 진행
for i in range(n_games):
    
    
    while not terminated and not truncated:

       
    

        



# 그래프 그리기

