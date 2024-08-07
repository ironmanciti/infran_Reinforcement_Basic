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
class ActorCritic(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorCritic, self).__init__()
        # Fully Connected Layer 1: 입력 차원에서 256개 노드로
        self.fc1 = nn.Linear(*input_dims, 256)
        # Policy Network의 Fully Connected Layer: 256개 노드에서 행동의 수만큼의 노드로
        self.fc_pi = nn.Linear(256, n_actions)
        # Value Network의 Fully Connected Layer: 256개 노드에서 1개 노드로
        self.fc_v = nn.Linear(256, 1)

    # 정책 네트워크의 순전파 함수
    def pi(self, state):
        x = F.relu(self.fc1(state))  # ReLU 활성화 함수를 사용하여 FC1 계산
        x = self.fc_pi(x)  # FC_pi 계산
        prob = F.softmax(x, dim=-1)  # FC_pi의 출력에 소프트맥스 함수 적용하여 각 행동에 대한 확률 계산
        return prob

    # 가치 네트워크의 순전파 함수
    def v(self, state):
        x = F.relu(self.fc1(state))  # ReLU 활성화 함수를 사용하여 FC1 계산
        v = self.fc_v(x)  # FC_v 계산
        return v


# 학습률 0 < alpha < 1 설정
alpha = 0.001
# 감가율 0 < gamma < 1 설정
gamma = 0.98
# 에피소드의 최대 개수 N 설정
N = 1000

batch_size = 32
# theta 매개변수와 가치 가중치 w 초기화
model = ActorCritic(env.observation_space.shape, n_actions).to(device)

optimizer = optim.Adam(model.parameters(), lr=alpha)

rendering = True

total_rewards = []

# batch를 만드는 함수
def make_batch(memory):
    batch_states, batch_actions, batch_rewards, batch_next_state, batch_done = [], [], [], [], []
    for transition in memory:
        s, a, r, s_, done = transition
        batch_states.append(s)
        batch_actions.append([a])
        batch_rewards.append([r])
        batch_next_state.append(s_)
        done_mask = 0 if done else 1
        batch_done.append([done_mask])
    return torch.FloatTensor(batch_states).to(device), torch.LongTensor(batch_actions).to(device), \
        torch.FloatTensor(batch_rewards).to(device), torch.FloatTensor(batch_next_state).to(device), \
        torch.FloatTensor(batch_done).to(device)


# 각 에피소드에 대해
for episode in range(N):
    if episode > N * 0.995:
        env = gym.make(env_name, render_mode='human')

    # 첫 상태 초기화
    s, _ = env.reset()

    done = False
    memory = []
    # S가 종료 상태가 아닌 동안 반복
    while not done:
        # A ~ pi(.|S,theta) - 정책 네트워크에서 액션 하나를 샘플링
        probs = model.pi(torch.tensor(
            s, dtype=torch.float).to(device)).detach().numpy()
        a = np.random.choice(action_space, p=probs)
        # 액션 A를 취하고, S', R 관찰
        s_, r, terminated, truncated, _ = env.step(a)
        memory.append((s, a, r, s_, done))
        # S <- S'
        s = s_

        done = terminated or truncated

        if done:
            s_batch, a_batch, r_batch, s_next_batch, done_batch = make_batch(
                memory)

            # delta <- R + gamma * v(S',w) - v(S,w) (만약 S'가 종료 상태라면 v(S',w) = 0)
            td_target = r_batch + gamma * \
                model.v(s_next_batch) * (1-done_batch)
            # advantage = reward + gamma * v(S',w) - v(S,w) --> advantage = delta
            delta = td_target - model.v(s_batch)

            # w <- w + alpha * delta * gradient(v(S,w)) - 가치 네트워크 매개변수 업데이트
            # theta <- theta + alpha * I * delta * gradient(pi(A|S,theta)) - 정책 네트워크 매개변수 업데이트
            pi = model.pi(s_batch)
            pi_a = pi.gather(1, a_batch)
            # loss = -1 * policy.logprob(action) * advantage + critic loss
            loss = -1 * torch.log(pi_a) * delta + \
                F.smooth_l1_loss(model.v(s_batch), td_target)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            total_rewards.append(sum(r_batch.detach().numpy()))

    if episode % 100 == 0:
        avg_score = np.mean(total_rewards[-100:])
        print(f'episode {episode},  최근 100 episode 평균 reward {avg_score: .2f}')
