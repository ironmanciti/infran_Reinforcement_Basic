# One-step Actor-Critic(episodic), for estimating pi_theta == pi_*
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time

device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')  # CUDA 장치를 사용 가능하면 사용, 아니면 CPU 사용
print(device)

#env_name = 'CartPole-v1'  # 환경 이름 설정
env_name = 'LunarLander-v2'
env = gym.make(env_name)  # 환경 생성

n_actions = env.action_space.n  # 가능한 액션의 수를 설정
action_space = np.arange(env.action_space.n)  # 가능한 액션의 범위 설정

print(n_actions)

# 정책 네트워크 (pi(a|s,theta))와 가치 네트워크 (v(s,w))를 포함한 Actor-Critic 클래스



# 학습률 0 < alpha < 1 설정

# 감가율 0 < gamma < 1 설정

# 에피소드의 최대 개수 N 설정


batch_size = 32
# theta 매개변수와 가치 가중치 w 초기화



# batch를 만드는 함수




# 각 에피소드에 대해
for episode in range(N):
    if episode > N * 0.995:
        env = gym.make(env_name, render_mode='human')

    # 첫 상태 초기화
    
    
    
    # S가 종료 상태가 아닌 동안 반복
    while not done:
        # A ~ pi(.|S,theta) - 정책 네트워크에서 액션 하나를 샘플링
        
        
        # 액션 A를 취하고, S', R 관찰
        
        
        # S <- S'
        

        done = terminated or truncated

        if done:
            

            # delta <- R + gamma * v(S',w) - v(S,w) (만약 S'가 종료 상태라면 v(S',w) = 0)
            
            
            # advantage = reward + gamma * v(S',w) - v(S,w) --> advantage = delta
            

            # w <- w + alpha * delta * gradient(v(S,w)) - 가치 네트워크 매개변수 업데이트
            # theta <- theta + alpha * I * delta * gradient(pi(A|S,theta)) - 정책 네트워크 매개변수 업데이트
            
            
            # loss = -1 * policy.logprob(action) * advantage + critic loss
            
            
            

    if episode % 100 == 0:
        avg_score = np.mean(total_rewards[-100:])
        print(f'episode {episode},  최근 100 episode 평균 reward {avg_score: .2f}')
