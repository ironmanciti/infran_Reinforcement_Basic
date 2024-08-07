# Reinforce 알고리즘
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')  # CUDA 장치를 사용 가능하면 사용, 아니면 CPU 사용
print(device)

ENV_NAME = 'CartPole-v1'  # 환경 이름 설정
env = gym.make(ENV_NAME)  # 환경 생성

action_space = np.arange(env.action_space.n)  # 가능한 액션의 수를 설정

# Policy Network를 정의
# 이 신경망은 상태(state)를 입력으로 받아 각 행동에 대한 확률을 출력



# Initialize the parameters theta




total_rewards = []  # 모든 보상을 저장할 리스트 생성

# 각 보상에 대해 감가율을 적용한 값을 반환하는 함수




start_time = time.time()  # 시작 시간 저장
batch_rewards = []  # 배치 보상을 저장할 리스트 생성
batch_actions = []  # 배치 액션을 저장할 리스트 생성
batch_states = []  # 배치 상태를 저장할 리스트 생성
batch_counter = 1  # 배치 카운터 초기화

for episode in range(N):  # N번 에피소드 동안 반복
    
    

    # 에피소드 종료 여부, 에피소드 중단 여부를 저장하는 변수를 False로 초기화
    terminated, truncated = False, False

    while not terminated and not truncated:  # 에피소드가 종료되지 않고 중단되지 않은 동안 반복
        
        

        if done:  # 에피소드가 종료되었다면
            # 보상을 감가율을 적용하여 배치 보상에 추가
            

            # 배치 카운터가 배치 크기와 같다면 (한 배치가 완성되었다면)
            
                # 배치에 있는 모든 에피소드에 대해 정책 손실 계산
                

                # 정책 업데이트
                

    if episode % 100 == 0:  # 매 100 에피소드마다
        avg_score = np.mean(total_rewards[-100:])  # 최근 100 에피소드의 평균 점수 계산
        # 평균 점수 출력
        print(f'episode {episode}, 최근 100 에피소드 평균 reward {avg_score: .2f}')

env.close()  # 환경 종료
print("duration = ", (time.time() - start_time) / 60, "minutes")  # 실행 시간 출력

running_avg = np.zeros(len(total_rewards))  # 누적 평균 점수를 저장할 배열 생성

for i in range(len(running_avg)):  # 각 에피소드에 대해
    # 최근 100 에피소드의 평균 점수 계산
    running_avg[i] = np.mean(total_rewards[max(0, i-100):(i+1)])

plt.plot(running_avg)  # 누적 평균 점수 그래프 그리기
plt.title('Running average of previous 100 rewards')  # 제목 설정
plt.show()  # 그래프 표시
