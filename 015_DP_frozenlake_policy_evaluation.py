import gymnasium as gym
import numpy as np
import pprint

# SFFF       (S: 시작점, 안전)
# FHFH       (F: 얼어있는 표면, 안전)
# FFFH       (H: 구멍, 추락)
# HFFG       (G: 목표, 프리스비 위치)

# LEFT = 0
# DOWN = 1
# RIGHT = 2
# UP = 3

map = "4x4"
 
SLIPPERY = False # 결정론적 환경
#SLIPPERY = True  # 확률적 환경

env = gym.make('FrozenLake-v1', desc=None,
                map_name=map, is_slippery=SLIPPERY)

GAMMA = 1.0
THETA = 1e-5
num_states = env.observation_space.n
num_actions = env.action_space.n
transitions = env.P
# 환경의 동역학 출력
#pprint.pprint(transitions)

# 평가할 정책인 pi를 입력
# 무작위 정책으로 초기화
policy = np.ones([num_states, num_actions]) * 0.25

# 모든 s에 대해 V(s) = 0 배열을 초기화
V = np.zeros(num_states)

#반복
while True:
    #delta <- 0
    delta = 0
    #각 s에 대해 반복:
    for s in range(num_states):
        #v <- V(s)
        old_value = V[s]
        new_value = 0
        #업데이트 규칙 : V(s) = sum(pi(a|s)*sum(p(s,a)*[r + gamma*v(s')]))
        for a, prob_action in enumerate(policy[s]):
            # s', r에 대해 합산
            for prob, s_, reward, _ in transitions[s][a]:
                new_value += prob_action * prob * (reward + GAMMA * V[s_])
        V[s] = new_value
        #delta <- max(delta|v - V(s)|)
        delta = max(delta, np.abs(old_value - V[s]))

    #delta < theta까지
    if delta < THETA:
        break
#V는 v_pi에 수렴
print("수렴한 Optimal Value = \n", V.reshape(4, 4) if map == "4x4" else V.reshape(8, 8))
