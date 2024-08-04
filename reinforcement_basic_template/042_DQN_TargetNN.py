import gymnasium as gym
import matplotlib.pyplot as plt
import math
import random
import time
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

env_name = 'MountainCar-v0' 
# env_name = 'CartPole-v1'

env = gym.make(env_name)

# 하이퍼파라미터 설정



# 리플레이 메모리 클래스
class ExperienceReplay:
    def __init__(self, capacity):
       
    # 경험 추가 함수
    def push(self, state, action, new_state, reward, done):
        

    # 경험 샘플링 함수
    def sample(self, batch_size):
        
    def __len__(self):
       
    
# 신경망 클래스
class NeuralNetwork(nn.Module):
    def __init__(self):


    # 순전파 함수
    def forward(self, x):


# 액션 선택 함수
def select_action(state, steps_done):
  
    return action

# 리플레이 메모리 초기화


# 타겟 Q함수 초기화 (랜덤 가중치)


# 리플레이 메모리 초기화


# 타겟 Q함수 초기화 (랜덤 가중치)


# Q함수 초기화 (랜덤 가중치로 신경망 생성)


# 손실 함수 설정 (평균 제곱 오차)


# 최적화 알고리즘 설정 (Adam 옵티마이저)


# 타겟 네트워크 업데이트 카운터 초기화

# 각 에피소드에서 얻은 보상을 저장할 리스트 초기화

# 총 스텝 수 초기화

# 학습 시작 시간 기록



# 에피소드 루프
for episode in range(num_episodes):

    s, _ = env.reset()
    reward = 0
    while True:
        total_steps += 1

        # 액션 선택


        # 환경에서 액션 수행


        # 리플레이 메모리에 경험 저장


        if len(memory) >= batch_size:
            # 리플레이 메모리에서 미니배치 샘플링
            states, actions, new_states, rewards, dones = memory.sample(
                batch_size)

            # 샘플링한 데이터를 텐서로 변환하여 장치에 할당
            
            # 타겟 Q 네트워크로부터 새로운 상태의 Q 값 계산


            # 타겟 값 계산

            # 예측 값 계산


            # 손실 계산 및 역전파


            # 타겟 네트워크 업데이트


        s = s_

        if done:
            reward_history.append(reward)
            print(f"{episode} episode finished after {reward:.2f} rewards")
            break

# 평균 보상 출력


# 마지막 50 에피소드의 평균 보상 출력


# 하이퍼파라미터 정보 출력


# 경과 시간 출력


# 학습 과정의 보상 플롯

