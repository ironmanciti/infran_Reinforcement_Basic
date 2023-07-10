import gymnasium as gym
import numpy as np

map = "4x4"

SLIPPERY = False  # 결정론적 환경
#SLIPPERY = True  # 확률적 환경

env = gym.make('FrozenLake-v1', desc=None,
               map_name=map, is_slippery=SLIPPERY)

GAMMA = 0.95
THETA = 1e-5
num_states = env.observation_space.n
num_actions = env.action_space.n
transitions = env.P

# 1. 모든 s에 대해 V(s) = 0으로 초기화
V = np.zeros(num_states)

#루프
while True:
    #delta <- 0
    delta = 0
    #각 s에 대해 반복
    for s in range(num_states):
        # v <- V(s)
        old_value = V[s]
        new_action_values = np.zeros(num_actions)
        #V(s) = max_a(sum(p(s,a)*[r + gamma*v(s')]))
        for a in range(num_actions):
            # s', r에 대해 합산
            for prob, s_, r, _ in transitions[s][a]:
                new_action_values[a] += prob * (r + GAMMA * V[s_])
        v_max = max(new_action_values)
        #delta <-max(delta|v - V(s)|)
        delta = max(delta, np.abs(v_max - old_value))
        # 상태-가치 함수 업데이트
        V[s] = v_max
    #delta < theta까지
    if delta < THETA:
        break

# 액션 가치를 사용해 결정론적 최적 정책 추출
# 결정론적 정책 생성
pi = np.zeros((num_states, num_actions))

for s in range(num_states):
    #pi(s) = argmax_a(sum(p(s,a)*[r + gamma*v(s')]))
    action_values = np.zeros(num_actions)

    for a in range(num_actions):
        # s', r에 대해 합산
        for prob, s_, r, _ in transitions[s][a]:
            action_values[a] += prob * (r + GAMMA * V[s_])

    #pi(s) <- argmax_a(action_values)
    # 각 상태에서 최대 행동 가치를 가지는 행동 선택
    new_action = np.argmax(action_values)
    pi[s] = np.eye(num_actions)[new_action]

print("Optimal Value Function = \n", V.reshape(
    4, 4) if map == "4x4" else V.reshape(8, 8))
print("Optimal Policy = \n", pi)
print("Optimal Action = \n", np.argmax(pi, axis=1).reshape(4, 4) if map == "4x4"
      else np.argmax(pi, axis=1).reshape(8, 8))
