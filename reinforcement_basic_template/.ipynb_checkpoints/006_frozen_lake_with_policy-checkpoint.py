import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# LEFT - 0, DOWN - 1, RIGHT - 2, UP - 3

# SFFF       (S: 시작점, 안전)
# FHFH       (F: 얼어있는 표면, 안전)
# FFFH       (H: 구멍, 추락)
# HFFG       (G: 목표지점, 프리스비 위치)

# position number
# 0  1  2  3
# 4  5  6  7
# 8  9  10 11
# 12 13 14 15

# 보상
# 목표에 도달: +1
# 구멍에 도달: 0
# 얼어있는 곳에 도달: 0

# 간단한 결정론적 정책
policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1,
          8: 2, 9: 1, 10: 1, 13: 2, 14: 2}

with_policy = True  # 간단한 결정론적 정책 사용

SLIPPERY = False  # 결정론적 환경
#SLIPPERY = True  # 확률적 환경

# FrozenLake-v1 환경을 생성합니다.
# desc: None이면 기본 맵을 사용합니다.
# map_name: 사용할 맵의 이름을 지정합니다. 여기서는 "4x4" 맵을 사용합니다.
# is_slippery: True이면 미끄러운 표면(확률적 환경)을 사용하고, False이면 결정론적 환경을 사용합니다.




# 게임 진행
for i in range(n_games):
    
    
    while not terminated and not truncated:

       
    

        



# 그래프 그리기

