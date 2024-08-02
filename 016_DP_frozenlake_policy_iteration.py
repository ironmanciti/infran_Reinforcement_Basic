import gymnasium as gym
import numpy as np
import copy

# LEFT - 0, DOWN - 1, RIGHT - 2, UP - 3

# SFFF       (S: 시작점, 안전)
# FHFH       (F: 얼어있는 표면, 안전)
# FFFH       (H: 구멍, 추락)
# HFFG       (G: 목표, 프리스비 위치)

map = "4x4"

SLIPPERY = False  # 결정론적 환경
#SLIPPERY = True  # 확률적 환경

# FrozenLake-v1 환경을 생성합니다.
# desc: None이면 기본 맵을 사용합니다.
# map_name: 사용할 맵의 이름을 지정합니다. 
# is_slippery: True이면 미끄러운 표면(확률적 환경)을 사용하고, False이면 결정론적 환경을 사용합니다.
env = gym.make('FrozenLake-v1', desc=None, map_name=map, is_slippery=SLIPPERY)

GAMMA = 0.9
THETA = 1e-7
num_states = env.observation_space.n
num_actions = env.action_space.n
transitions = env.P

# 1. 모든 s에 대해 V(s) = 0과 임의의 pi(s)를 초기화
V = np.zeros(num_states)
pi = np.ones([num_states, num_actions]) * 0.25

policy_stable = False  # 정책이 안정적인지 여부를 나타내는 변수

while not policy_stable:
    #2. 정책 평가
    while True:
        # delta 초기화 : delta <- 0
        delta = 0
        #각 s에 대해 반복
        for s in range(num_states):
            old_value = V[s]  # 현재 가치 저장
            new_value = 0
            # 업데이트 규칙: V(s) = sum(pi(a|s)*sum(p(s,a)*[r + gamma*v(s')]))
            for a, prob_a in enumerate(pi[s]):
                # s', r에 대해 합산 (각 행동 a에 대한 상태 전이 확률을 반영한 보상 합산)
                for prob, s_, r, _ in transitions[s][a]:
                    new_value += prob_a * prob * (r + GAMMA * V[s_])
            V[s] = new_value   # 상태 s의 가치를 새로운 값으로 업데이트
            # delta 갱신: delta <-- max(delta|v - V(s)|)
            delta = max(delta, np.abs(old_value - V[s]))
        # delta가 THETA보다 작으면 정책 평가 종료
        if delta < THETA:
            break

    #3. 정책 개선
    old_pi = copy.deepcopy(pi)

    #각 s에 대해:
    for s in range(num_states):
        # pi_s <- argmax_a(sum(p(s',r|s,a)*[r + gamma*V(s')]))
        # 새로운 행동 가치 초기화
        new_action_values = np.zeros(num_actions)
        for a in range(num_actions):
            # 각 행동 a에 대해 상태 전이 확률과 보상을 반영한 가치 합산
            for prob, s_, r, _ in transitions[s][a]:
                new_action_values[a] += prob * (r + GAMMA * V[s_])

        # 새로운 정책 결정: 가치가 최대가 되는 행동 선택
        new_action = np.argmax(new_action_values)
        pi[s] = np.eye(num_actions)[new_action]  # 새로운 정책 갱신

    # 이전 정책과 새로운 정책이 동일한지 확인하여 정책 안정성 판단
    if np.array_equal(old_pi, pi):
        policy_stable = True
    else:
        policy_stable = False
        
    #정책이 안정화 되면 V와 pi를 반환하고 종료, 아니면 2로 돌아감.

print("Optimal Value Function = \n", V.reshape(4, 4) if map == "4x4" else V.reshape(8, 8))
print("Optimal Policy = \n", pi)
print("Optimal Action = \n", np.argmax(pi, axis=1).reshape(4, 4) if map == "4x4"
      else np.argmax(pi, axis=1).reshape(8, 8))
print("LEFT - 0, DOWN - 1, RIGHT - 2, UP - 3")