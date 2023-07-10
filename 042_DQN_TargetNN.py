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
num_episodes = 300
GAMMA = 0.99  # 감마 (discount factor)
learning_rate = 0.001  # 학습률
hidden_layer = 120  # 은닉층 노드 수
replay_memory_size = 50_000  # 리플레이 메모리 크기
batch_size = 128  # 배치 크기

e_start = 0.9  # 입실론 초기값
e_end = 0.05  # 입실론 최종값
e_decay = 200  # 입실론 감소율

target_nn_update_frequency = 10  # 타겟 네트워크 업데이트 주기
clip_error = False  # 오차 클리핑 여부

device = "cpu"

n_inputs = env.observation_space.shape[0]  # 입력 차원 수 (상태 수)
n_outputs = env.action_space.n  # 출력 차원 수 (액션 수)

# 리플레이 메모리 클래스
class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # 경험 추가 함수
    def push(self, state, action, new_state, reward, done):
        transition = (state, action, new_state, reward, done)

        if self.position >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    # 경험 샘플링 함수
    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)

# 신경망 클래스
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(n_inputs, hidden_layer)
        self.linear2 = nn.Linear(hidden_layer, hidden_layer//2)
        self.linear3 = nn.Linear(hidden_layer//2, n_outputs)

    # 순전파 함수
    def forward(self, x):
        a1 = torch.relu(self.linear1(x))
        a2 = torch.relu(self.linear2(a1))
        output = self.linear3(a2)
        return output

# 액션 선택 함수
def select_action(state, steps_done):
    e_threshold = e_end + (e_start - e_end) * \
        math.exp(-1. * steps_done/e_decay)

    if random.random() > e_threshold:
        with torch.no_grad():
            state = torch.Tensor(state).to(device)
            action_values = Q(state)
            action = torch.argmax(action_values).item()
    else:
        action = env.action_space.sample()

    return action

# 리플레이 메모리 초기화
memory = ExperienceReplay(replay_memory_size)

# 타겟 Q함수 초기화 (랜덤 가중치)
target_Q = NeuralNetwork().to(device)

# Q함수 초기화 (랜덤 가중치)
Q = NeuralNetwork().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(Q.parameters(), lr=learning_rate)

update_target_counter = 0
reward_history = []
total_steps = 0
start_time = time.time()

# 에피소드 루프
for episode in range(num_episodes):
    if episode > num_episodes * 0.98:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    s, _ = env.reset()
    reward = 0
    while True:
        total_steps += 1

        # 액션 선택
        a = select_action(s, total_steps)

        # 환경에서 액션 수행
        s_, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        reward += r

        # 리플레이 메모리에 경험 저장
        memory.push(s, a, s_, r, done)

        if len(memory) >= batch_size:
            # 리플레이 메모리에서 미니배치 샘플링
            states, actions, new_states, rewards, dones = memory.sample(
                batch_size)

            states = torch.Tensor(states).to(device)
            actions = torch.LongTensor(actions).to(device)
            new_states = torch.Tensor(new_states).to(device)
            rewards = torch.Tensor([rewards]).to(device)
            dones = torch.Tensor(dones).to(device)

            new_action_values = target_Q(new_states).detach()

            y_target = rewards + \
                (1 - dones) * GAMMA * torch.max(new_action_values, 1)[0]
            y_pred = Q(states).gather(1, actions.unsqueeze(1))

            # 손실 계산 및 역전파
            loss = criterion(y_pred.squeeze(), y_target.squeeze())
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # 타겟 네트워크 업데이트
            if update_target_counter % target_nn_update_frequency == 0:
                target_Q.load_state_dict(Q.state_dict())

            update_target_counter += 1

        s = s_

        if done:
            reward_history.append(reward)
            print(f"{episode} episode finished after {reward:.2f} rewards")
            break

print("Average rewards: %.2f" % (sum(reward_history)/num_episodes))
print("Average of last 100 episodes: %.2f" % (sum(reward_history[-50:])/50))
print("---------------------- Hyper parameters --------------------------------------")
print(
    f"GAMMA:{GAMMA}, learning rate: {learning_rate}, hidden layer: {hidden_layer}")
print(f"replay_memory: {replay_memory_size}, batch size: {batch_size}")
print(f"epsilon_start: {e_start}, epsilon_end: {e_end}, " +
      f"epsilon_decay: {e_decay}")
print(
    f"update frequency: {target_nn_update_frequency}, clipping: {clip_error}")
elapsed_time = time.time() - start_time
print(f"Time Elapsed : {elapsed_time//60} min {elapsed_time%60:.0} sec")

# 학습 과정의 보상 플롯
plt.bar(torch.arange(len(reward_history)).numpy(), reward_history)
plt.xlabel("episodes")
plt.ylabel("rewards")
plt.title("DQN - Target Network")
plt.show()
