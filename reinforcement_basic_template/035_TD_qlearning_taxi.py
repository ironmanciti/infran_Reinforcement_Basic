# Q-Learning (off-policy TD control) 알고리즘으로 pi=pi* 구하기
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


# 알고리즘의 파라미터 설정: 스텝 사이즈 alpha (0,1], 작은 e > 0


# Q(s,a)를 모든 s, a에 대해 임의로 초기화(Q(terminal,.)=0 제외)


scores = []  # 각 에피소드에서 에이전트가 얻은 점수를 기록
steps = []  # 각 에피소드에서 에이전트가 목표를 찾아가는 단계 수를 기록
greedy = []  # epsilon decay history 기록

# 각 에피소드를 위한 반복문
for episode in range(n_episodes):

    # 에피소드를 초기화

    # 각 에피소드의 각 스텝에 대한 반복문
    while True:
      
        # Q에서 유도된 정책(예: e-greedy)을 사용하여 S에서 A를 선택
        # 행동 정책 : e-greedy
       

        # epsilon이 epsilon_final보다 크다면 epsilon_decay를 곱하여 감소
       

        # 행동 A를 취하고, R, S'을 관찰
  

        # Q(S,A)를 업데이트: Q(S,A) <- Q(S,A) + alpha[R + gamma*max_aQ(S',a) - Q(S, A)]
        # 최적 행동가치함수 q*를 직접 근사
        # 대상 정책 : greedy policy
  

        # 에피소드가 끝나면 반복문 종료


        # S <- S'




    if episode % 100 == 0:
        print(
            f"최근 100 episode 평균 score = {np.mean(scores[-100:])}, 평균 step = {np.mean(steps[-100:])}")

# 각 에피소드별 단계 수 그래프 그리기


# 각 에피소드별 점수 그래프 그리기


# epsilon decay history 그래프 그리기

